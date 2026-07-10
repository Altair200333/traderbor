"""Phase C bake-off model zoo for scanner-v3 (see docs/notes/2026-07-07/
scanner-v3-data-model-plan.md, "Phase C -- Model bake-off").

Uniform adapter interface, drop-in compatible with train_eval.fit_predict:

    fit_predict_<name>(train_df, test_df, features) -> np.ndarray  # scores, len(test_df)

Conventions mirrored from train_eval.fit_predict:
  * binary target        y = (df["outcome"] == "tp").astype(int)
  * sample weights       df["sw"]  (per-symbol 24h uniqueness, produced by prep())
  * higher score = better (rank / p(tp)); AUC + top-decile net R are ordering-only,
    so rankers may return uncalibrated scores.
  * train_df is expected pre-filtered to resolved rows (outcome in tp/sl) by the
    caller, exactly as train_eval.main / harness_v3.walk_forward_dev do.

Missing-value policy (per spec):
  * GBDTs (lgbm/catboost/xgboost, and the LGBMRanker) consume native NaN.
  * torch / tabpfn / logreg get median-impute + a missing-indicator column for the
    perp (funding/OI/top-trader) features -- those are the columns with real NaN gaps.

Harness wiring (one line, later, by whoever owns harness_v3.py):
    from models_v3 import REGISTRY
    # inside _fit_fold, before falling back to te.fit_predict:
    if model in REGISTRY:  return REGISTRY[model](train, test, feats)

Everything heavy is imported lazily inside each adapter so importing this module is
cheap and never pulls torch/tabpfn unless used. CPU-only: torch threads capped at 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# shared config
# ----------------------------------------------------------------------------
SEED = 13
BAR_KEY = "as_of"          # per-scan-bar candidate pool: events sharing as_of are the
                           # simultaneous candidates the selector fills 3 slots from --
                           # this is the actual decision unit (not symbol-day).
COST_BPS = 25.0            # only used by internal helpers if needed; smoke owns metrics

# perp / positioning columns that carry genuine NaN gaps -> get missing-indicators.
PERP_FEATURES = [
    "funding_rate_last", "funding_z_30d", "funding_cum_3d", "funding_pctile_90d",
    "oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_z_7d",
    "toptrader_ls", "toptrader_ls_z_7d", "taker_ratio_24h",
]

# LGBM champion params (train_eval.py:106-110) -- mirrored philosophy for catboost/xgb.
_LGBM_PARAMS = dict(
    num_leaves=15, min_child_samples=60, learning_rate=0.05, n_estimators=500,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1,
)


def _y(df: pd.DataFrame) -> np.ndarray:
    return (df["outcome"].to_numpy() == "tp").astype(int)


def _sw(df: pd.DataFrame) -> np.ndarray:
    return df["sw"].to_numpy() if "sw" in df.columns else np.ones(len(df))


def _tail_val(n: int, frac: float = 0.15, floor: int = 100) -> int:
    """positional tail-val size, mirroring train_eval.fit_predict."""
    return max(int(n * frac), floor)


# ----------------------------------------------------------------------------
# dense matrix builder: median-impute + missing-indicator (perp cols)
# ----------------------------------------------------------------------------
def _dense(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]):
    """Return (Xtr, Xte) float32 with non-finite -> train-median, plus a
    missing-indicator column for every perp feature present in `features`."""
    feats = list(features)
    ind_cols = [c for c in feats if c in PERP_FEATURES]

    def _mat(df):
        m = df[feats].to_numpy(dtype=np.float64, copy=True)
        m[~np.isfinite(m)] = np.nan
        return m

    Xtr, Xte = _mat(train_df), _mat(test_df)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    tr_isnan, te_isnan = np.isnan(Xtr), np.isnan(Xte)
    Xtr = np.where(tr_isnan, med, Xtr)
    Xte = np.where(te_isnan, med, Xte)

    if ind_cols:
        idx = [feats.index(c) for c in ind_cols]
        Xtr = np.hstack([Xtr, tr_isnan[:, idx].astype(np.float64)])
        Xte = np.hstack([Xte, te_isnan[:, idx].astype(np.float64)])
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def _cap_torch_threads():
    import torch
    try:
        torch.set_num_threads(4)
    except Exception:
        pass
    torch.manual_seed(SEED)
    np.random.seed(SEED)


# ----------------------------------------------------------------------------
# 1. logreg -- regularized logistic-regression floor
# ----------------------------------------------------------------------------
def fit_predict_logreg(train_df, test_df, features) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte = _dense(train_df, test_df, features)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(sc.transform(Xtr), _y(train_df), sample_weight=_sw(train_df))
    return clf.predict_proba(sc.transform(Xte))[:, 1]


# ----------------------------------------------------------------------------
# 2a. catboost -- shallow/regularized, champion-comparable (native NaN)
# ----------------------------------------------------------------------------
def fit_predict_catboost(train_df, test_df, features) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool
    feats = list(features)
    Xtr = train_df[feats].to_numpy(dtype=np.float64)
    Xte = test_df[feats].to_numpy(dtype=np.float64)
    y, sw = _y(train_df), _sw(train_df)
    nv = _tail_val(len(train_df))
    pool_tr = Pool(Xtr[:-nv], y[:-nv], weight=sw[:-nv])
    pool_val = Pool(Xtr[-nv:], y[-nv:], weight=sw[-nv:])
    clf = CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=5.0,
        subsample=0.8, bootstrap_type="Bernoulli", rsm=0.8,
        loss_function="Logloss", eval_metric="AUC", random_seed=SEED,
        nan_mode="Min", od_type="Iter", od_wait=50, verbose=False,
    )
    clf.fit(pool_tr, eval_set=pool_val, verbose=False)
    return clf.predict_proba(Xte)[:, 1]


# ----------------------------------------------------------------------------
# 2b. xgboost -- shallow/regularized, champion-comparable (native NaN)
# ----------------------------------------------------------------------------
def fit_predict_xgboost(train_df, test_df, features) -> np.ndarray:
    from xgboost import XGBClassifier
    feats = list(features)
    # xgboost QuantileDMatrix rejects +/-inf even with missing=nan -> map to NaN
    Xtr = train_df[feats].to_numpy(dtype=np.float32)
    Xte = test_df[feats].to_numpy(dtype=np.float32)
    Xtr[~np.isfinite(Xtr)] = np.nan
    Xte[~np.isfinite(Xte)] = np.nan
    y, sw = _y(train_df), _sw(train_df)
    nv = _tail_val(len(train_df))
    clf = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=5.0,
        objective="binary:logistic", eval_metric="auc",
        early_stopping_rounds=50, tree_method="hist", n_jobs=4,
        random_state=SEED, missing=np.nan,
    )
    clf.fit(Xtr[:-nv], y[:-nv], sample_weight=sw[:-nv],
            eval_set=[(Xtr[-nv:], y[-nv:])], sample_weight_eval_set=[sw[-nv:]],
            verbose=False)
    return clf.predict_proba(Xte)[:, 1]


# ----------------------------------------------------------------------------
# 3. tabpfn -- TabPFN v2 subsample ensemble (CPU)
#    N members, each on a random row-subsample within TabPFN's limits; avg proba.
# ----------------------------------------------------------------------------
def fit_predict_tabpfn(train_df, test_df, features, n_members: int = 4,
                       subsample: int = 3000) -> np.ndarray:
    import os
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")   # never hang on interactive login
    from tabpfn import TabPFNClassifier
    Xtr_full, Xte = _dense(train_df, test_df, features)   # tabpfn wants no NaN
    y = _y(train_df)
    rng = np.random.default_rng(SEED)
    n = len(Xtr_full)
    probs = np.zeros(len(Xte))
    ok = 0
    for m in range(n_members):
        if n > subsample:
            idx = rng.choice(n, size=subsample, replace=False)
        else:
            idx = np.arange(n)
        ytr = y[idx]
        if ytr.max() == ytr.min():          # degenerate subsample -> skip
            continue
        clf = TabPFNClassifier(
            n_estimators=2, device="cpu", random_state=SEED + m,
            ignore_pretraining_limits=True, fit_mode="fit_preprocessors",
        )
        try:
            clf.fit(Xtr_full[idx], ytr)
        except Exception as e:
            # TabPFN v2 weights are gated: first use requires accepting the license
            # to obtain TABPFN_TOKEN, then the checkpoint downloads from HuggingFace.
            # Neither token nor cached weights exist in this offline env -> documented
            # stub. The subsample-ensemble logic above is correct; only the gated
            # download is blocked. To enable: set env TABPFN_TOKEN=<token from license>.
            raise RuntimeError(
                "TabPFN v2 unavailable: gated model weights need TABPFN_TOKEN "
                "(accept license once, then checkpoint auto-downloads). "
                f"underlying: {type(e).__name__}: {e}") from e
        p = clf.predict_proba(Xte)
        cls = list(clf.classes_)
        probs += p[:, cls.index(1)] if 1 in cls else np.zeros(len(Xte))
        ok += 1
    return probs / ok if ok else np.full(len(Xte), 0.5)


# ----------------------------------------------------------------------------
# 4a. xranker_gbm -- LGBMRanker (lambdarank), groups = per-bar candidate pool
#     relevance = graded realized-R buckets (loss<0 ->0, 0-1 ->1, 1-2 ->2, >2 ->3)
# ----------------------------------------------------------------------------
def _graded_relevance(df: pd.DataFrame) -> np.ndarray:
    r = df["r_mkt_eval"].to_numpy() if "r_mkt_eval" in df.columns else df["r_market"].to_numpy()
    return np.digitize(r, bins=[0.0, 1.0, 2.0]).astype(int)   # -> 0,1,2,3


def fit_predict_xranker_gbm(train_df, test_df, features) -> np.ndarray:
    import lightgbm as lgb
    feats = list(features)
    key = train_df[BAR_KEY].to_numpy()
    order = np.argsort(key, kind="stable")            # lambdarank needs contiguous groups
    Xtr = train_df[feats].to_numpy(dtype=np.float32)[order]
    rel = _graded_relevance(train_df)[order]
    _, counts = np.unique(key[order], return_counts=True)
    rk = lgb.LGBMRanker(
        objective="lambdarank", num_leaves=15, min_child_samples=60,
        learning_rate=0.05, n_estimators=300, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, label_gain=list(range(4)),
        random_state=SEED, verbose=-1,
    )
    rk.fit(Xtr, rel, group=counts)
    return rk.predict(test_df[feats].to_numpy(dtype=np.float32))


# ----------------------------------------------------------------------------
# 4b. xranker_torch -- shared MLP encoder + 1 attention layer across the candidate
#     axis + listwise (ListNet top-1) softmax loss per bar. Padded/masked, <100k params.
# ----------------------------------------------------------------------------
def fit_predict_xranker_torch(train_df, test_df, features, hidden: int = 32,
                              epochs: int = 20, bar_batch: int = 48) -> np.ndarray:
    import torch
    import torch.nn as nn
    _cap_torch_threads()
    from sklearn.preprocessing import StandardScaler

    Xtr_np, Xte_np = _dense(train_df, test_df, features)
    sc = StandardScaler().fit(Xtr_np)
    Xtr_np, Xte_np = sc.transform(Xtr_np).astype(np.float32), sc.transform(Xte_np).astype(np.float32)
    D = Xtr_np.shape[1]
    rel = _graded_relevance(train_df).astype(np.float32)

    class BarRanker(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h))
            self.attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(h)
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(h, 1))

        def forward(self, x, key_pad):  # x:[B,S,D] key_pad:[B,S] True=pad
            z = self.enc(x)
            a, _ = self.attn(z, z, z, key_padding_mask=key_pad)
            z = self.norm(z + a)
            return self.head(z).squeeze(-1)  # [B,S]

    model = BarRanker(D, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # group train rows by bar
    tr_bars = list(pd.DataFrame({"k": train_df[BAR_KEY].to_numpy()}).groupby("k").indices.values())
    Xtr_t = torch.from_numpy(Xtr_np)
    rel_t = torch.from_numpy(rel)

    def _collate(bar_idx_list):
        smax = max(len(b) for b in bar_idx_list)
        B = len(bar_idx_list)
        x = torch.zeros(B, smax, D)
        tgt = torch.zeros(B, smax)
        pad = torch.ones(B, smax, dtype=torch.bool)  # True=pad
        for i, b in enumerate(bar_idx_list):
            s = len(b)
            x[i, :s] = Xtr_t[b]
            tgt[i, :s] = rel_t[b]
            pad[i, :s] = False
        return x, tgt, pad

    model.train()
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        rng.shuffle(tr_bars)
        for j in range(0, len(tr_bars), bar_batch):
            batch = tr_bars[j:j + bar_batch]
            x, tgt, pad = _collate(batch)
            scores = model(x, pad)
            scores = scores.masked_fill(pad, -1e9)
            logp = torch.log_softmax(scores, dim=1)
            tgt = tgt.masked_fill(pad, 0.0)
            denom = tgt.sum(dim=1, keepdim=True)
            valid = (denom.squeeze(1) > 0)
            if valid.sum() == 0:
                continue
            p_tgt = torch.where(denom > 0, tgt / denom.clamp(min=1e-9), torch.zeros_like(tgt))
            loss = -(p_tgt[valid] * logp[valid]).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # predict: attention within each TEST bar
    model.eval()
    Xte_t = torch.from_numpy(Xte_np)
    te_groups = pd.DataFrame({"k": test_df[BAR_KEY].to_numpy()}).groupby("k").indices
    out = np.zeros(len(test_df), dtype=np.float32)
    with torch.no_grad():
        for _, idx in te_groups.items():
            x = Xte_t[idx].unsqueeze(0)                      # [1,S,D]
            pad = torch.zeros(1, len(idx), dtype=torch.bool)
            out[idx] = model(x, pad).squeeze(0).numpy()
    return out


# ----------------------------------------------------------------------------
# 5. moe -- regime mixture-of-experts, HARD regime gating (simpler-first fallback;
#    learned softmax gate is the torch upgrade, noted but not the default).
#    3 experts (bull/bear/chop) = separate champion-param LGBMs, gated by BTC state.
# ----------------------------------------------------------------------------
def _regime(df: pd.DataFrame) -> np.ndarray:
    above = df["btc_above_ema50"].to_numpy() if "btc_above_ema50" in df.columns else np.ones(len(df))
    z = df["btc_z20"].to_numpy() if "btc_z20" in df.columns else np.zeros(len(df))
    above = np.nan_to_num(above, nan=1.0) > 0.5
    z = np.nan_to_num(z, nan=0.0)
    reg = np.where(above & (z >= 0), "bull", np.where((~above) & (z < 0), "bear", "chop"))
    return reg


def fit_predict_moe(train_df, test_df, features) -> np.ndarray:
    import lightgbm as lgb
    feats = list(features)
    y, sw = _y(train_df), _sw(train_df)
    reg_tr, reg_te = _regime(train_df), _regime(test_df)
    Xtr_all = train_df[feats].to_numpy(dtype=np.float32)
    Xte_all = test_df[feats].to_numpy(dtype=np.float32)

    def _fit(mask):
        clf = lgb.LGBMClassifier(**_LGBM_PARAMS, random_state=SEED)
        clf.fit(Xtr_all[mask], y[mask], sample_weight=sw[mask])
        return clf

    glob = _fit(np.ones(len(train_df), dtype=bool))     # fallback expert
    experts = {}
    for r in ("bull", "bear", "chop"):
        m = reg_tr == r
        experts[r] = _fit(m) if m.sum() >= 300 else glob

    out = np.zeros(len(test_df))
    for r in ("bull", "bear", "chop"):
        m = reg_te == r
        if m.any():
            out[m] = experts[r].predict_proba(Xte_all[m])[:, 1]
    return out


# ----------------------------------------------------------------------------
# 6. multitask -- torch MLP trunk + heads: p(tp) [main], mfe_r quantile (pinball),
#    t_exit_min regression (masked for NaN). Returns main-head p(tp).
# ----------------------------------------------------------------------------
def fit_predict_multitask(train_df, test_df, features, hidden: int = 64,
                          epochs: int = 40, batch: int = 512) -> np.ndarray:
    import torch
    import torch.nn as nn
    _cap_torch_threads()
    from sklearn.preprocessing import StandardScaler

    Xtr_np, Xte_np = _dense(train_df, test_df, features)
    sc = StandardScaler().fit(Xtr_np)
    Xtr_np = sc.transform(Xtr_np).astype(np.float32)
    Xte_np = sc.transform(Xte_np).astype(np.float32)
    D = Xtr_np.shape[1]

    y = _y(train_df).astype(np.float32)
    sw = _sw(train_df).astype(np.float32)
    # aux targets, standardized; t_exit has NaN (unresolved) -> masked
    mfe = train_df["mfe_r"].to_numpy(dtype=np.float64)
    tex = train_df["t_exit_min"].to_numpy(dtype=np.float64)
    mfe_m = float(np.nanmean(mfe)); mfe_s = float(np.nanstd(mfe) + 1e-9)
    tex_m = float(np.nanmean(tex)); tex_s = float(np.nanstd(tex) + 1e-9)
    mfe_z = np.nan_to_num((mfe - mfe_m) / mfe_s, nan=0.0).astype(np.float32)
    tex_z = ((tex - tex_m) / tex_s).astype(np.float32)
    tex_mask = np.isfinite(tex_z).astype(np.float32)
    tex_z = np.nan_to_num(tex_z, nan=0.0)

    class MT(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                                       nn.Linear(h, h), nn.ReLU())
            self.h_tp = nn.Linear(h, 1)      # main
            self.h_mfe = nn.Linear(h, 1)     # quantile (median) regression
            self.h_tex = nn.Linear(h, 1)     # time-to-exit
        def forward(self, x):
            z = self.trunk(x)
            return self.h_tp(z).squeeze(-1), self.h_mfe(z).squeeze(-1), self.h_tex(z).squeeze(-1)

    model = MT(D, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def pinball(pred, target, q=0.5):
        e = target - pred
        return torch.maximum(q * e, (q - 1) * e)

    Xt = torch.from_numpy(Xtr_np)
    yt = torch.from_numpy(y); swt = torch.from_numpy(sw)
    mfet = torch.from_numpy(mfe_z); text = torch.from_numpy(tex_z); texm = torch.from_numpy(tex_mask)
    n = len(Xt)
    rng = np.random.default_rng(SEED)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for j in range(0, n, batch):
            b = perm[j:j + batch]
            xb = Xt[b]
            o_tp, o_mfe, o_tex = model(xb)
            l_tp = (bce(o_tp, yt[b]) * swt[b]).mean()
            l_mfe = pinball(o_mfe, mfet[b]).mean()
            mt = texm[b]
            l_tex = ((pinball(o_tex, text[b], q=0.5) * mt).sum() / mt.sum().clamp(min=1.0))
            loss = l_tp + 0.3 * l_mfe + 0.3 * l_tex
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        o_tp, _, _ = model(torch.from_numpy(Xte_np))
        return torch.sigmoid(o_tp).numpy()


# ----------------------------------------------------------------------------
# adapter registry -- one-line wire-in point for harness_v3.py
# ----------------------------------------------------------------------------
REGISTRY = {
    "logreg": fit_predict_logreg,
    "catboost": fit_predict_catboost,
    "xgboost": fit_predict_xgboost,
    "tabpfn": fit_predict_tabpfn,
    "xranker_gbm": fit_predict_xranker_gbm,
    "xranker_torch": fit_predict_xranker_torch,
    "moe": fit_predict_moe,
    "multitask": fit_predict_multitask,
}
