"""Scanner-v3 Phase B training/evaluation harness (vaults + regime + adversarial + stability).

Layered on top of train_eval.py (imported, never mutated on disk). Adds:

  1. VAULT DISCIPLINE  -- hard-coded backward/forward vaults; dev runs drop vault
     rows (+24h embargo) from every training fold and never emit a test fold that
     overlaps a vault. Loud banner + assertions. Vault EVALUATION (Phase D) is
     implemented but guarded behind --open-vault (burns the vault; logs opening).
  2. REGIME STRATIFICATION -- bull/bear/chop per calendar month from BTC 1h klines
     (close vs 50d EMA + 30d drawdown); per-regime walk-forward breakdown.
  3. ADVERSARIAL VALIDATION -- LGBM classifies epoch membership on FEATURES(+flow);
     AUC >> 0.5 + a dominating feature == drift/epoch carrier.
  4. FEATURE-STABILITY PRUNING -- per-fold gain-importance stability -> pruned
     feature list; base-vs-pruned walk-forward comparison.

Every mechanic takes the dataset path as a parameter (the 4y set does not exist
yet). Default CLI runs DEVELOPMENT mode only (vaults excluded, never opened).

Usage:
  python harness_v3.py --dataset .../events_v3a1.parquet --with-flow
  python harness_v3.py --open-vault forward --dataset <4y set>   # burns a vault
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval as te  # noqa: E402  (imported, never modified on disk)
from train_eval import FEATURES, FLOW_FEATURES, HORIZON_MS, money, prep  # noqa: E402
from slot_sim import report as slot_report, run_slots  # noqa: E402
from universe import REPO_ROOT  # noqa: E402
from models_v3 import REGISTRY  # noqa: E402  (Phase C model zoo; internals never modified)

from sklearn.metrics import roc_auc_score  # noqa: E402

# --- Vault windows (UTC, hard-coded per Phase B spec) ------------------------
# (name, start_date, end_date) -- end_date is INCLUSIVE (whole day).
VAULTS = [
    ("backward", "2023-07-01", "2023-12-31"),
    ("forward",  "2026-04-07", "2026-07-05"),
]
EMBARGO_MS = 24 * 3_600_000
BTC_KLINES = REPO_ROOT / "research" / "data" / "klines" / "1h" / "BTCUSDT.parquet"
VAULT_LOG = REPO_ROOT / "research" / "data" / "vault_openings.log"

# scanner-v3 A2/A3 perp features (opt-in; default runs are byte-identical).
FUNDING_FEATURES = [
    "funding_rate_last", "funding_z_30d", "funding_cum_3d", "funding_pctile_90d",
]
METRIC_FEATURES = [
    "oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_z_7d",
    "toptrader_ls", "toptrader_ls_z_7d", "taker_ratio_24h",
]

# --- Drift-hardening (opt-in via --harden) -----------------------------------
# Market-level carriers: value is CONSTANT across all rows sharing an as_of
# (verified const_frac==1.0). Hardened variant = trailing 30d z-score over the
# feature's own UNIQUE as_of history, causal (window ends at the current as_of,
# inclusive of the current bar -> the convention used by the raw z features).
HARDEN_MARKET = [
    "breadth_ema50", "breadth_roc24_pos", "btc_roc_4h", "btc_roc_24h",
    "btc_z20", "btc_vol_24h", "median_roc24", "median_roc4", "altseason",
]
# Per-symbol carrier (VERIFIED NOT market-level: const_frac~0.30, per-symbol
# median range). Hardened = trailing 30d z over the SYMBOL's own past event rows.
HARDEN_PERSYM = ["med_range24"]
# Cross-sectional rank carriers dropped from the hardened set (see investigation):
# per-bar coverage is a median of 2-3 candidates and events are a high-vol_ratio
# SELECTED subsample, so the pct-rank encodes trigger-selectivity/universe drift,
# not NaN-coverage -> "rank scaled by coverage" cannot fix it.
DROP_RANKS = ["rank_volratio", "rank_roc4"]
# Funding: funding_z_30d is the already-self-normalized clean representative
# (adversarial rank #21); the raw level/pctile/cum carriers are dropped.
HARDENED_FUNDING = ["funding_z_30d"]
DROP_FUNDING_RAW = ["funding_rate_last", "funding_pctile_90d", "funding_cum_3d"]
CALENDAR = ["hour", "dow"]
HZ_WINDOW_DAYS = 30
HZ_MIN_MARKET = 20
HZ_MIN_PERSYM = 3


def _trailing_z_market(ev: pd.DataFrame, feat: str, window_days: int = HZ_WINDOW_DAYS,
                       min_periods: int = HZ_MIN_MARKET) -> pd.Series:
    """Causal trailing z of a MARKET-LEVEL feature over its unique-as_of history.

    Value is constant per as_of, so collapse to one point per as_of, roll a
    time-based `window_days` window (pandas closed='right' == inclusive of the
    current as_of, excludes the point exactly window_days ago), z-score with
    population std, then broadcast back to every row via as_of. Causal because
    the rolling window never looks past the current as_of.
    """
    u = ev.groupby("as_of", sort=True)[feat].first()
    s = pd.Series(u.to_numpy(), index=pd.to_datetime(u.index, unit="ms", utc=True))
    r = s.rolling(f"{window_days}D", min_periods=min_periods)
    z = (s - r.mean()) / r.std(ddof=0)
    return ev["as_of"].map(pd.Series(z.to_numpy(), index=u.index))


def _trailing_z_persym(ev: pd.DataFrame, feat: str, window_days: int = HZ_WINDOW_DAYS,
                       min_periods: int = HZ_MIN_PERSYM) -> pd.Series:
    """Causal trailing z of a PER-SYMBOL feature over the symbol's own past event
    rows (time-based `window_days` window, inclusive of the current bar)."""
    out = pd.Series(np.nan, index=ev.index, dtype=float)
    for _, g in ev.sort_values("as_of").groupby("symbol", sort=False):
        s = pd.Series(g[feat].to_numpy(),
                      index=pd.to_datetime(g["as_of"].to_numpy(), unit="ms", utc=True))
        r = s.rolling(f"{window_days}D", min_periods=min_periods)
        out.loc[g.index] = ((s - r.mean()) / r.std(ddof=0)).to_numpy()
    return out


def add_hardened_features(ev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add every *_hz hardened column IN-MEMORY (no parquet write). Returns
    (ev_with_hz, mapping raw_name -> hardened_name)."""
    ev = ev.copy()
    mapping: dict[str, str] = {}
    for f in HARDEN_MARKET:
        ev[f + "_hz"] = _trailing_z_market(ev, f)
        mapping[f] = f + "_hz"
    for f in HARDEN_PERSYM:
        ev[f + "_hz"] = _trailing_z_persym(ev, f)
        mapping[f] = f + "_hz"
    return ev, mapping


def harden_feature_list(base_feats: list[str], mapping: dict) -> list[str]:
    """Return the base-hardened feature list: raw carriers replaced by their
    hardened variant, cross-sectional rank carriers dropped."""
    out = []
    for f in base_feats:
        if f in DROP_RANKS:
            continue
        out.append(mapping.get(f, f))
    return out


def verify_hardening_causality(ev: pd.DataFrame, feat: str = "breadth_ema50",
                               n: int = 3, seed: int = 0) -> list[tuple]:
    """Recompute a hardened MARKET feature at n random as_of using only rows
    truncated at that as_of; assert equality with the full-data value."""
    full = _trailing_z_market(ev, feat)
    u = ev.groupby("as_of")[feat].first().sort_index()
    cand = u.index.to_numpy()[HZ_MIN_MARKET + 5:]  # ensure window is populated
    picks = np.random.default_rng(seed).choice(cand, size=n, replace=False)
    rows = []
    for T in picks:
        sub = ev[ev["as_of"] <= T]
        z_sub = _trailing_z_market(sub, feat)
        r_full = ev.index[ev["as_of"] == T][0]
        r_sub = sub.index[sub["as_of"] == T][0]
        vf, vs = float(full.loc[r_full]), float(z_sub.loc[r_sub])
        rows.append((int(T), vf, vs, bool(np.isclose(vf, vs, equal_nan=True))))
    return rows


def _ms(date_str: str, end_inclusive: bool = False) -> int:
    ts = pd.Timestamp(date_str, tz="UTC")
    if end_inclusive:
        ts = ts + pd.Timedelta(days=1)  # inclusive end -> start of next day
    return int(ts.timestamp() * 1000)


def vault_bounds() -> list[dict]:
    """Core + embargoed ms bounds for each vault."""
    out = []
    for name, s, e in VAULTS:
        vs, ve = _ms(s), _ms(e, end_inclusive=True)
        out.append({"name": name, "start_str": s, "end_str": e,
                    "vs": vs, "ve": ve, "vs_emb": vs - EMBARGO_MS, "ve_emb": ve + EMBARGO_MS})
    return out


def in_vault_or_embargo(as_of: np.ndarray) -> np.ndarray:
    """Boolean mask: True where as_of falls in ANY vault +/- 24h embargo (train-drop set)."""
    m = np.zeros(len(as_of), dtype=bool)
    for v in vault_bounds():
        m |= (as_of >= v["vs_emb"]) & (as_of < v["ve_emb"])
    return m


def month_overlaps_vault(month: str) -> str | None:
    """Return vault name if the calendar month overlaps a vault CORE window, else None."""
    m_s = _ms(month + "-01")
    m_e = int((pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthBegin(1)).timestamp() * 1000)
    for v in vault_bounds():
        if m_s < v["ve"] and v["vs"] < m_e:  # interval overlap
            return v["name"]
    return None


def print_vault_banner(ev: pd.DataFrame) -> None:
    n_drop = int(in_vault_or_embargo(ev["as_of"].to_numpy()).sum())
    print("=" * 78)
    print("  VAULT DISCIPLINE ACTIVE  (development mode -- vaults NEVER opened here)")
    print("  Excluded windows (rows dropped from ALL training folds, +/-24h embargo):")
    for v in vault_bounds():
        print(f"    - {v['name']:8s} vault: {v['start_str']} .. {v['end_str']} UTC "
              f"(embargo {EMBARGO_MS//3_600_000}h each edge)")
    print(f"  Vault rows present in this dataset (excluded from every train fold): {n_drop}")
    print("  Test folds overlapping any vault are SKIPPED (never scored).")
    print("=" * 78)


# --- Regime labelling --------------------------------------------------------
def btc_month_regime(klines_path: Path = BTC_KLINES, ema_days: int = 50,
                     dd_days: int = 30, dd_thresh: float = -0.15) -> tuple[dict, pd.DataFrame]:
    """Label each calendar month bull/bear/chop from BTC 1h closes.

    daily close (UTC), EMA50 (daily-equivalent), 30d rolling-max drawdown:
      bull = close > EMA50 and dd_30 > -15%
      bear = close < EMA50 and dd_30 <= -15%
      else = chop
    Month label = majority of its daily states. Source: BTC 1h klines cache.
    """
    k = pd.read_parquet(klines_path, columns=["open_time", "close"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    daily = k.set_index("ts").sort_index()["close"].resample("1D").last().dropna()
    ema = daily.ewm(span=ema_days, adjust=False).mean()
    dd = daily / daily.rolling(dd_days, min_periods=1).max() - 1.0
    state = pd.Series("chop", index=daily.index)
    state[(daily > ema) & (dd > dd_thresh)] = "bull"
    state[(daily < ema) & (dd <= dd_thresh)] = "bear"
    det = pd.DataFrame({"month": daily.index.strftime("%Y-%m"), "state": state.values})
    month_regime = det.groupby("month")["state"].agg(lambda s: s.value_counts().idxmax())
    return month_regime.to_dict(), det


# --- Walk-forward (development) ---------------------------------------------
def _fit_registry_cal(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
                      model: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Phase C REGISTRY adapter and Platt-calibrate it EXACTLY the way
    train_eval.fit_predict calibrates LGBM: single fit on the tr-85% slice; the
    15% time-tail val slice + the test rows are scored in ONE predict call
    (concat so the adapter fits once); a LogisticRegression maps (val_score ->
    y_val) and is applied to the test scores. Returns (p_calibrated, raw_scores).
    raw_scores are the uncalibrated model scores on the test rows (ordering-only
    metrics -- AUC / quintile lift / ranker p@3 -- and ranker-native top-k use
    these; EV>0 selection uses the calibrated probabilities)."""
    from sklearn.linear_model import LogisticRegression
    fn = REGISTRY[model]
    n_val = max(int(len(train) * 0.15), 100)
    tr, val = train.iloc[:-n_val], train.iloc[-n_val:]
    scores = np.asarray(fn(tr, pd.concat([val, test], axis=0), feats), dtype=float)
    val_raw, test_raw = scores[:len(val)], scores[len(val):]
    yv = (val["outcome"].to_numpy() == "tp").astype(int)
    m = np.isfinite(val_raw)
    if m.sum() < 10 or len(np.unique(yv[m])) < 2:
        return test_raw, test_raw  # degenerate val -> leave uncalibrated
    cal = LogisticRegression(max_iter=1000)
    cal.fit(val_raw[m].reshape(-1, 1), yv[m])
    fill = float(np.nanmedian(val_raw[m]))
    p_cal = cal.predict_proba(np.nan_to_num(test_raw, nan=fill).reshape(-1, 1))[:, 1]
    return p_cal, test_raw


def _sanitize_metric_infs(train: pd.DataFrame, test: pd.DataFrame,
                          feats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metric change ratios can be +/-inf when an old OI denominator is zero.
    Treat that as missing only for runs that actually select metric features."""
    if not any(f in METRIC_FEATURES for f in feats):
        return train, test
    cols = [f for f in feats if f in METRIC_FEATURES]
    train = train.copy()
    test = test.copy()
    train[cols] = train[cols].replace([np.inf, -np.inf], np.nan)
    test[cols] = test[cols].replace([np.inf, -np.inf], np.nan)
    return train, test


def _fit_fold(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], model: str) -> np.ndarray:
    """Fit one fold. lgbm/lr -> train_eval.fit_predict (feature list swapped, then
    restored). Any Phase C REGISTRY model -> Platt-calibrated adapter (see
    _fit_registry_cal). Raw uncalibrated test scores are stashed on
    _fit_fold.last_raw for ordering metrics and ranker-native policies."""
    train, test = _sanitize_metric_infs(train, test, feats)
    if model in REGISTRY:
        p_cal, raw = _fit_registry_cal(train, test, feats, model)
        _fit_fold.last_raw = raw
        return p_cal
    saved = te.FEATURES
    te.FEATURES = feats
    try:
        p = te.fit_predict(train, test, model)
        _fit_fold.last_raw = np.asarray(p, dtype=float)
        return p
    finally:
        te.FEATURES = saved


def walk_forward_dev(ev: pd.DataFrame, feats: list[str], first_test: str, last_test: str,
                     model: str, cost_bps: float, regime_map: dict,
                     train_lookback_ms: int | None = None,
                     recency_halflife_ms: int | None = None) -> dict:
    """Purged expanding walk-forward with vault discipline. Returns folds/preds/importances.

    Opt-in training-window variants (both default None == unchanged expanding-window
    behaviour, existing callers byte-identical):
      * train_lookback_ms   -- rolling window: keep only train rows with
                               as_of >= t_start - train_lookback_ms.
      * recency_halflife_ms -- exponential recency weights multiplied into the
                               existing per-symbol uniqueness sample weight
                               (w *= 0.5 ** (age / halflife), age measured from
                               the fold's test-start).
    """
    ev = ev.copy()
    ev["_vault"] = in_vault_or_embargo(ev["as_of"].to_numpy())
    te.fit_predict.best_iters = []
    fold_rows, preds, imps, skipped = [], [], [], []
    resolved = ev["outcome"].isin(["tp", "sl"])

    for month in te.month_starts(first_test, last_test):
        vn = month_overlaps_vault(month)
        if vn is not None:
            skipped.append((month, vn))
            continue
        t_start = _ms(month + "-01")
        t_end = int((pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthBegin(1)).timestamp() * 1000)
        # training: purged (24h before test) + resolved + NOT in any vault/embargo
        train = ev[(ev["as_of"] < t_start - HORIZON_MS) & resolved & ~ev["_vault"]]
        if train_lookback_ms is not None:                       # opt-in rolling window
            train = train[train["as_of"] >= t_start - train_lookback_ms]
        if recency_halflife_ms is not None:                     # opt-in recency weights
            train = train.copy()
            age = (t_start - train["as_of"].to_numpy()).astype(float)
            train["sw"] = train["sw"].to_numpy() * (0.5 ** (age / recency_halflife_ms))
        test = ev[(ev["as_of"] >= t_start) & (ev["as_of"] < t_end)]
        # Folds overlapping a vault CORE are skipped above, but a fold ADJACENT to a
        # vault edge can still contain +/-24h EMBARGO rows (e.g. 2023-06-30 in fold
        # 2023-06, 2024-01-01..02 in fold 2024-01). Spec: vault rows stay out of test
        # folds AND training rows -> drop embargo rows from the test fold. No-op for
        # any fold month not touching a vault edge (all runs with first-test >= 2025-04).
        if test["_vault"].any():
            test = test[~test["_vault"]]
        # ASSERTION: a produced test fold must never contain vault rows
        assert not test["_vault"].any(), f"BUG: test fold {month} contains vault rows"
        if len(train) < 300 or len(test) == 0:
            continue
        p = _fit_fold(train, test, feats, model)
        test = test.copy()
        test["p_hat"] = p
        test["p_raw"] = getattr(_fit_fold, "last_raw", p)   # uncalibrated (ordering/ranker use)
        test["regime"] = regime_map.get(month, "chop")
        preds.append(test)
        if model == "lgbm" and hasattr(te.fit_predict, "last_model"):
            imps.append(pd.Series(
                te.fit_predict.last_model.booster_.feature_importance(importance_type="gain"),
                index=feats))

        res = test["outcome"].isin(["tp", "sl"])
        auc = (roc_auc_score((test.loc[res, "outcome"] == "tp").astype(int),
                             test.loc[res, "p_hat"])
               if res.sum() >= 10 and test.loc[res, "outcome"].nunique() > 1 else np.nan)
        ev_val = test["p_hat"] * test["tp_rr"] - (1 - test["p_hat"]) - cost_bps / 1e4 / test["d_final"]
        sel = test[ev_val > 0]
        m_sel = money(sel, cost_bps)
        fold_rows.append({"month": month, "regime": regime_map.get(month, "chop"),
                          "auc": auc, "n_test": len(test), "n_train": len(train),
                          "ev_n": m_sel["n"], "ev_Rm": round(m_sel["R_mkt"], 2),
                          "ev_avgR": round(m_sel["avg_mkt"], 4) if m_sel["n"] else np.nan})
    return {"folds": pd.DataFrame(fold_rows),
            "preds": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
            "importances": imps, "skipped": skipped}


def regime_table(folds: pd.DataFrame) -> pd.DataFrame:
    """Per-regime breakdown: fold AUC (mean), EV>0 avg R, summed R, n."""
    rows = []
    for reg in ["bull", "bear", "chop"]:
        g = folds[folds["regime"] == reg]
        if len(g) == 0:
            continue
        ev_n = int(g["ev_n"].sum())
        ev_Rm = float(g["ev_Rm"].sum())
        rows.append({"regime": reg, "n_folds": len(g),
                     "mean_fold_auc": round(g["auc"].mean(), 4),
                     "n_test": int(g["n_test"].sum()),
                     "ev_sel_n": ev_n, "ev_sel_Rm": round(ev_Rm, 2),
                     "ev_sel_avgR": round(ev_Rm / ev_n, 4) if ev_n else np.nan})
    return pd.DataFrame(rows)


def pooled_metrics(preds: pd.DataFrame, cost_bps: float) -> dict:
    """Pooled-over-folds AUC + Q5-Q1 tp-rate lift + EV>0 stream money.

    AUC / lift on RESOLVED (tp|sl) rows; lift = tp_rate(top p_hat quintile) -
    tp_rate(bottom quintile). EV stream = money() on the p_hat*rr>cost selection.
    """
    if preds.empty:
        return {"pooled_auc": np.nan, "q5_q1_lift": np.nan,
                "ev_n": 0, "ev_Rm": 0.0, "ev_avgR": np.nan}
    res = preds[preds["outcome"].isin(["tp", "sl"])]
    y = (res["outcome"] == "tp").astype(int)
    auc = roc_auc_score(y, res["p_hat"]) if y.nunique() > 1 else np.nan
    try:
        q = pd.qcut(res["p_hat"], 5, labels=False, duplicates="drop")
        grp = y.groupby(q).mean()
        lift = float(grp.iloc[-1] - grp.iloc[0]) if len(grp) >= 2 else np.nan
    except Exception:
        lift = np.nan
    ev_val = preds["p_hat"] * preds["tp_rr"] - (1 - preds["p_hat"]) - cost_bps / 1e4 / preds["d_final"]
    m = money(preds[ev_val > 0], cost_bps)
    return {"pooled_auc": auc, "q5_q1_lift": lift,
            "ev_n": m["n"], "ev_Rm": m["R_mkt"], "ev_avgR": m["avg_mkt"]}


# --- Adversarial validation --------------------------------------------------
def adversarial_validation(ev: pd.DataFrame, feats: list[str], split_ms: int | None = None,
                           seed: int = 0) -> dict:
    """Train LGBM to classify epoch (early vs late by as_of median). AUC>>0.5 == drift."""
    import lightgbm as lgb

    if split_ms is None:
        split_ms = int(ev["as_of"].median())
    y = (ev["as_of"].to_numpy() >= split_ms).astype(int)
    X = ev[feats]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ev))
    cut = int(len(ev) * 0.7)
    tr, va = idx[:cut], idx[cut:]
    clf = lgb.LGBMClassifier(num_leaves=31, min_child_samples=60, learning_rate=0.05,
                             n_estimators=300, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr])
    auc = roc_auc_score(y[va], clf.predict_proba(X.iloc[va])[:, 1])
    gain = pd.Series(clf.booster_.feature_importance(importance_type="gain"),
                     index=feats).sort_values(ascending=False)
    return {"auc": float(auc), "split_ms": split_ms,
            "split_utc": str(pd.to_datetime(split_ms, unit="ms", utc=True)),
            "n_early": int((y == 0).sum()), "n_late": int((y == 1).sum()),
            "top15": gain.head(15), "gain": gain}


def adversarial_validation_full(ev: pd.DataFrame, feats: list[str], seed: int = 0) -> pd.Series:
    """Same epoch-classifier as adversarial_validation, but return the FULL
    gain-importance ranking (descending) so out-of-top-15 feature ranks are
    locatable."""
    import lightgbm as lgb
    split_ms = int(ev["as_of"].median())
    y = (ev["as_of"].to_numpy() >= split_ms).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ev))
    cut = int(len(ev) * 0.7)
    clf = lgb.LGBMClassifier(num_leaves=31, min_child_samples=60, learning_rate=0.05,
                             n_estimators=300, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    clf.fit(ev[feats].iloc[idx[:cut]], y[idx[:cut]])
    return pd.Series(clf.booster_.feature_importance(importance_type="gain"),
                     index=feats).sort_values(ascending=False)


# --- Feature-stability pruning ----------------------------------------------
def stability_prune(importances: list[pd.Series], keep_frac: float = 0.5) -> dict:
    """Per-feature stability = fraction of folds where gain-importance > that fold's median.

    Keep features stable in >= keep_frac of folds. Also reports rank-CV.
    """
    imp = pd.concat(importances, axis=1)  # rows=features, cols=folds
    med = imp.median(axis=0)               # per-fold median importance
    above = imp.gt(med, axis=1)            # feature > fold median?
    stability = above.mean(axis=1)         # fraction of folds
    ranks = imp.rank(axis=0, ascending=False)
    rank_cv = (ranks.std(axis=1) / ranks.mean(axis=1)).round(3)
    kept = stability[stability >= keep_frac].index.tolist()
    dropped = stability[stability < keep_frac].index.tolist()
    summary = pd.DataFrame({"stability": stability.round(3), "rank_cv": rank_cv,
                            "mean_gain": imp.mean(axis=1).round(1)}).sort_values(
        "stability", ascending=False)
    return {"kept": kept, "dropped": dropped, "summary": summary}


# --- Phase D: VAULT EVALUATION (guarded; burns the vault) --------------------
def open_vault_eval(ev: pd.DataFrame, which: str, feats: list[str], model: str,
                    cost_bps: float, regime_map: dict | None = None
                    ) -> tuple[dict, pd.DataFrame]:
    """PHASE D. Burns a vault. Only call via --open-vault. Logs the opening.

    forward : train on ALL dev data with as_of <= 2026-04-06, test on forward vault.
    backward: train on dev data with as_of >= 2024-01-01 only (mirrors backward-OOS),
              test on backward vault.
    Training always drops vault/embargo rows so the other vault stays sealed.
    Returns (pre-registered c1/c2/c3 metrics, test predictions frame).
    """
    v = next(x for x in vault_bounds() if x["name"] == which)
    ts = datetime.now(timezone.utc).isoformat()
    VAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(VAULT_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{ts}\tOPENED {which} vault {v['start_str']}..{v['end_str']} "
                 f"dataset_rows={len(ev)}\n")
    print("!" * 78)
    print(f"!!  VAULT BURN: opening the {which.upper()} vault. This vault is now SPENT.")
    print(f"!!  Logged to {VAULT_LOG} at {ts}")
    print("!" * 78)

    resolved = ev["outcome"].isin(["tp", "sl"])
    not_vault = ~in_vault_or_embargo(ev["as_of"].to_numpy())  # the OTHER vault stays sealed
    test = ev[(ev["as_of"] >= v["vs"]) & (ev["as_of"] < v["ve"])].copy()
    if which == "forward":
        train = ev[(ev["as_of"] < v["vs"] - EMBARGO_MS) & resolved & not_vault]
    else:  # backward -> train only on data AFTER 2024-01-01 (later regime), test earlier
        cut = _ms("2024-01-01")
        train = ev[(ev["as_of"] >= cut) & resolved & not_vault]
    if len(train) < 300 or len(test) == 0:
        return ({"which": which, "error": f"insufficient data (train={len(train)}, test={len(test)})"},
                test)
    test["p_hat"] = _fit_fold(train, test, feats, model)
    test["month"] = pd.to_datetime(test["as_of"], unit="ms", utc=True).dt.strftime("%Y-%m")
    test["regime"] = test["month"].map(regime_map or {}).fillna("?")
    res = test["outcome"].isin(["tp", "sl"])
    auc = (roc_auc_score((test.loc[res, "outcome"] == "tp").astype(int), test.loc[res, "p_hat"])
           if res.sum() >= 10 and test.loc[res, "outcome"].nunique() > 1 else np.nan)
    out = {"which": which, "model": model, "n_feat": len(feats), "n_train": len(train),
           "n_test": len(test), "n_resolved": int(res.sum()), "auc": auc}
    # c1 (tp_rate = precision on selected) + c2 (avg_mkt) at 10/25/50 bps
    for cb in (10.0, 25.0, 50.0):
        sel_cb = test[test["p_hat"] * test["tp_rr"] - (1 - test["p_hat"])
                      - cb / 1e4 / test["d_final"] > 0]
        out[f"sel_{int(cb)}bps"] = money(sel_cb, cb)
    # c3: slot-3 capacity sim on the EV>0 stream at headline cost (sym-day dedup)
    test["ev_val"] = (test["p_hat"] * test["tp_rr"] - (1 - test["p_hat"])
                      - cost_bps / 1e4 / test["d_final"])
    test["selected_ev"] = test["ev_val"] > 0
    sel = test[test["selected_ev"]].copy()
    sel["rm"] = sel["r_mkt_eval"] - cost_bps / 1e4 / sel["d_final"]
    if "t_exit_min" not in sel.columns:
        sel["t_exit_min"] = np.nan  # run_slots then holds the slot the full 24h
    sel["day"] = pd.to_datetime(sel["as_of"], unit="ms", utc=True).dt.date
    stream = (sel.sort_values("as_of").groupby(["symbol", "day"], as_index=False).head(1)
              .sort_values("as_of"))
    taken = run_slots(stream, 3)
    out["slot3"] = slot_report(taken, f"{which} slots=3")
    out["slot3_by_regime"] = {str(k): round(float(x), 2)
                              for k, x in taken.groupby("regime")["rm"].sum().items()}
    return out, test


def _pooled_row(name: str, run: dict, feats: list[str], cost_bps: float) -> dict:
    pm = pooled_metrics(run["preds"], cost_bps)
    fa = run["folds"]["auc"]
    preds = run["preds"]
    if preds.empty:
        sel_n = sel_resolved = sel_tp = sel_sl = 0
        sel_precision = np.nan
    else:
        ev_val = preds["p_hat"] * preds["tp_rr"] - (1 - preds["p_hat"]) - cost_bps / 1e4 / preds["d_final"]
        sel = preds[ev_val > 0]
        sel_res = sel[sel["outcome"].isin(["tp", "sl"])]
        sel_n = int(len(sel))
        sel_resolved = int(len(sel_res))
        sel_tp = int((sel_res["outcome"] == "tp").sum())
        sel_sl = int((sel_res["outcome"] == "sl").sum())
        sel_precision = sel_tp / sel_resolved if sel_resolved else np.nan
    return {"variant": name, "n_feat": len(feats),
            "mean_fold_auc": round(fa.mean(), 4),
            "pooled_auc": round(pm["pooled_auc"], 4),
            "q5_q1_lift": round(pm["q5_q1_lift"], 4),
            "ev_n": pm["ev_n"], "ev_sumR": round(pm["ev_Rm"], 2),
            "ev_avgR": round(pm["ev_avgR"], 4) if pm["ev_n"] else np.nan,
            "selected_n": sel_n, "selected_resolved": sel_resolved,
            "selected_tp": sel_tp, "selected_sl": sel_sl,
            "selected_precision": round(sel_precision, 4) if np.isfinite(sel_precision) else np.nan}


def _json_clean(x):
    """Convert pandas/numpy objects plus NaN/inf into strict JSON values."""
    if isinstance(x, dict):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, pd.Series):
        return _json_clean(x.to_dict())
    if isinstance(x, pd.DataFrame):
        return _json_clean(x.to_dict(orient="records"))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _dataset_meta(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": int(st.st_size),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


def _feature_coverage(ev: pd.DataFrame, feats: list[str]) -> dict:
    cols = [c for c in feats if c in ev.columns]
    if not cols:
        return {}
    x = ev[cols].replace([np.inf, -np.inf], np.nan)
    return {c: round(float(v), 6) for c, v in x.notna().mean().items()}


def _coverage_rows(ev: pd.DataFrame, rfeats: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall, by_month, by_symbol = [], [], []
    for variant, feats in rfeats.items():
        cols = [c for c in feats if c in ev.columns]
        if not cols:
            continue
        raw = ev[cols]
        clean = raw.replace([np.inf, -np.inf], np.nan)
        for c in cols:
            overall.append({"variant": variant, "feature": c,
                            "non_null_frac": float(clean[c].notna().mean()),
                            "inf_frac": float(np.isinf(raw[c].to_numpy(dtype=float)).mean())})
        if "month" in ev.columns:
            for month, idx in ev.groupby("month").groups.items():
                sub_raw = raw.loc[idx]
                sub_clean = clean.loc[idx]
                for c in cols:
                    by_month.append({"variant": variant, "month": month, "feature": c,
                                     "non_null_frac": float(sub_clean[c].notna().mean()),
                                     "inf_frac": float(np.isinf(sub_raw[c].to_numpy(dtype=float)).mean())})
        if "symbol" in ev.columns:
            core = ev["symbol"].isin(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            for sym, idx in ev[core].groupby("symbol").groups.items():
                sub_raw = raw.loc[idx]
                sub_clean = clean.loc[idx]
                for c in cols:
                    by_symbol.append({"variant": variant, "symbol": sym, "feature": c,
                                      "non_null_frac": float(sub_clean[c].notna().mean()),
                                      "inf_frac": float(np.isinf(sub_raw[c].to_numpy(dtype=float)).mean())})
    return pd.DataFrame(overall), pd.DataFrame(by_month), pd.DataFrame(by_symbol)


def _adversarial_rows(extras: dict | None) -> pd.DataFrame:
    rows = []

    def visit(prefix: str, obj) -> None:
        if isinstance(obj, dict) and "gain" in obj:
            gain = obj["gain"]
            for rank, (feat, val) in enumerate(gain.items(), 1):
                rows.append({"report": prefix, "adv_auc": obj.get("auc"),
                             "split_utc": obj.get("split_utc"),
                             "n_early": obj.get("n_early"), "n_late": obj.get("n_late"),
                             "feature": feat, "rank": rank, "gain": val})
            return
        if isinstance(obj, pd.Series):
            for rank, (feat, val) in enumerate(obj.items(), 1):
                rows.append({"report": prefix, "feature": feat, "rank": rank, "gain": val})
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(f"{prefix}.{k}" if prefix else str(k), v)

    visit("", extras or {})
    return pd.DataFrame(rows)


def save_run_artifacts(args, ev: pd.DataFrame, runs: dict, rfeats: dict,
                       extras: dict | None = None) -> Path | None:
    """Write raw artifacts for audit/replay. Opt-in via --artifact-dir."""
    if not args.artifact_dir:
        return None
    root = Path(args.artifact_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.run_name)
    out = root / f"{ts}_{safe_name}"
    out.mkdir(parents=True, exist_ok=False)

    rows = []
    for name, run in runs.items():
        feats = rfeats[name]
        rows.append(_pooled_row(name, run, feats, args.cost_bps))
        (out / f"{name}.features.txt").write_text("\n".join(feats) + "\n", encoding="utf-8")
        run["folds"].to_csv(out / f"{name}.folds.csv", index=False)
        if not run["preds"].empty:
            pred = run["preds"].copy()
            pred["ev_val"] = pred["p_hat"] * pred["tp_rr"] - (1 - pred["p_hat"]) - args.cost_bps / 1e4 / pred["d_final"]
            pred["selected_ev"] = pred["ev_val"] > 0
            pred_cols = [c for c in [
                "as_of", "month", "regime", "symbol", "outcome", "p_hat", "p_raw",
                "tp_rr", "d_final", "r_mkt_eval", "r_ret_eval", "retest_filled",
                "ev_val", "selected_ev", "is_hard", "is_marginal",
            ] if c in pred.columns]
            pred[pred_cols].to_parquet(out / f"{name}.preds.parquet", index=False)
        if run.get("importances"):
            pd.concat(run["importances"], axis=1).to_csv(out / f"{name}.importances.csv")

    variant_summary = pd.DataFrame(rows)
    variant_summary.to_csv(out / "variant_summary.csv", index=False)
    cov_all, cov_month, cov_symbol = _coverage_rows(ev, rfeats)
    cov_all.to_csv(out / "coverage_overall.csv", index=False)
    cov_month.to_csv(out / "coverage_by_month.csv", index=False)
    cov_symbol.to_csv(out / "coverage_core_symbols.csv", index=False)
    adv_rows = _adversarial_rows(extras or {})
    if not adv_rows.empty:
        adv_rows.to_csv(out / "adversarial_full.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "mode": "vault" if args.open_vault else "dev",
        "vault_log_exists": VAULT_LOG.exists(),
        "dataset": _dataset_meta(args.dataset),
        "klines": _dataset_meta(args.klines),
        "model": args.model,
        "cost_bps": args.cost_bps,
        "first_test": args.first_test,
        "last_test": args.last_test,
        "with_flow": bool(args.with_flow),
        "with_perp_funding": bool(args.with_perp_funding),
        "with_perp_metrics": bool(args.with_perp_metrics),
        "harden": bool(args.harden),
        "stability_thresh": args.stability_thresh,
        "event_rows": int(len(ev)),
        "resolved_rows": int(ev["outcome"].isin(["tp", "sl"]).sum()) if "outcome" in ev.columns else None,
        "outcome_counts": ev["outcome"].value_counts(dropna=False).to_dict() if "outcome" in ev.columns else {},
        "columns": list(ev.columns),
        "as_of_min_utc": str(pd.to_datetime(ev["as_of"].min(), unit="ms", utc=True)) if "as_of" in ev.columns else None,
        "as_of_max_utc": str(pd.to_datetime(ev["as_of"].max(), unit="ms", utc=True)) if "as_of" in ev.columns else None,
        "variants": rows,
        "feature_coverage": {name: _feature_coverage(ev, feats) for name, feats in rfeats.items()},
        "extras": extras or {},
    }
    (out / "summary.json").write_text(json.dumps(_json_clean(summary), indent=2, allow_nan=False),
                                      encoding="utf-8")
    print(f"\n=== ARTIFACTS WRITTEN ===\n{out}")
    return out


def _adv_report(ev: pd.DataFrame, feats: list[str], label: str) -> dict:
    adv = adversarial_validation(ev, feats)
    print(f"\n[{label}]  epoch-classifier AUC = {adv['auc']:.4f}   "
          f"(n_feat={len(feats)}, split {adv['split_utc']})")
    print("  top-15 carriers (gain):")
    for i, (f, g) in enumerate(adv["top15"].items(), 1):
        print(f"    {i:2d}. {f:22s} {g:12.1f}")
    return adv


def run_harden(ev_base: pd.DataFrame, base_feats: list[str], args, regime_map: dict) -> None:
    """Drift-hardening experiment: hardened variants REPLACE raw carriers.
    Adversarial before/after + WF runs (a) base-hardened, (b) +funding_z_30d,
    (c) pruned-hardened (+/-funding_z_30d / +/-metrics), (d) per-fold AUC +
    per-regime best."""
    cb = args.cost_bps
    ev, mapping = add_hardened_features(ev_base)
    hard_feats = harden_feature_list(base_feats, mapping)
    has_fund = "funding_z_30d" in ev.columns
    metric_missing = [c for c in METRIC_FEATURES if c not in ev.columns]
    use_metrics = bool(args.with_perp_metrics)
    if use_metrics:
        assert not metric_missing, f"metric features absent from dataset: {metric_missing} (rebuild with --perp-dir)"

    print("\n" + "#" * 78)
    print("  DRIFT-HARDENING EXPERIMENT (opt-in --harden; parquet NOT modified)")
    print("#" * 78)
    print(f"hardened MARKET-level (trailing {HZ_WINDOW_DAYS}d z over unique as_of): "
          f"{[m + '_hz' for m in HARDEN_MARKET]}")
    print(f"hardened PER-SYMBOL (trailing {HZ_WINDOW_DAYS}d z over symbol's own past): "
          f"{[m + '_hz' for m in HARDEN_PERSYM]}")
    print(f"DROPPED rank carriers: {DROP_RANKS}")
    print(f"funding representative (hardened runs): {HARDENED_FUNDING}  "
          f"| dropped raw funding: {DROP_FUNDING_RAW}")
    if use_metrics:
        cov = ev[METRIC_FEATURES].replace([np.inf, -np.inf], np.nan).notna().mean()
        print("metrics features (A3 opt-in): " + ", ".join(f"{c}={cov[c]:.3f}" for c in METRIC_FEATURES))
    print(f"base_feats={len(base_feats)}  ->  hardened_feats={len(hard_feats)}")

    # (2) CAUSALITY -----------------------------------------------------------
    print("\n=== CAUSALITY CHECK (recompute at 3 random as_of on truncated data) ===")
    for T, vf, vs, ok in verify_hardening_causality(ev, "breadth_ema50"):
        ts = pd.to_datetime(T, unit="ms", utc=True)
        print(f"  breadth_ema50_hz @ {ts}  full={vf:.6f}  truncated={vs:.6f}  equal={ok}")
    assert all(ok for *_, ok in verify_hardening_causality(ev, "breadth_ema50")), \
        "CAUSALITY FAIL: hardened market feature differs on truncation"
    print("  ASSERT PASS: hardened market feature is causal (truncation-invariant).")

    # (3) ADVERSARIAL before/after --------------------------------------------
    print("\n=== ADVERSARIAL RE-CHECK (epoch membership, split @ median as_of) ===")
    adv_reports = {}
    adv_reports["before_raw_base"] = _adv_report(ev, base_feats, "BEFORE  raw base")
    if has_fund:
        adv_reports["before_raw_base_raw_funding"] = _adv_report(
            ev, base_feats + FUNDING_FEATURES, "BEFORE  raw base + raw funding")
    adv_reports["after_hardened"] = _adv_report(ev, hard_feats, "AFTER   hardened")
    adv_reports["after_hardened_no_calendar"] = _adv_report(
        ev, [f for f in hard_feats if f not in CALENDAR], "AFTER   hardened, hour/dow removed")
    if has_fund:
        adv_reports["after_hardened_fundz"] = _adv_report(
            ev, hard_feats + HARDENED_FUNDING, "AFTER   hardened + funding_z_30d")
    if use_metrics:
        adv_reports["after_hardened_metrics"] = _adv_report(
            ev, hard_feats + METRIC_FEATURES, "AFTER   hardened + metrics")

    # (4) WALK-FORWARD --------------------------------------------------------
    print("\n=== WALK-FORWARD (vault-disciplined dev, %gbps) ===" % cb)
    runs, rfeats = {}, {}
    rfeats["base-hardened"] = hard_feats
    runs["base-hardened"] = walk_forward_dev(ev, hard_feats, args.first_test,
                                             args.last_test, args.model, cb, regime_map)
    if has_fund:
        rfeats["base-hardened+fundz"] = hard_feats + HARDENED_FUNDING
        runs["base-hardened+fundz"] = walk_forward_dev(ev, rfeats["base-hardened+fundz"],
                                                       args.first_test, args.last_test,
                                                       args.model, cb, regime_map)
    if use_metrics:
        rfeats["base-hardened+metrics"] = hard_feats + METRIC_FEATURES
        runs["base-hardened+metrics"] = walk_forward_dev(ev, rfeats["base-hardened+metrics"],
                                                         args.first_test, args.last_test,
                                                         args.model, cb, regime_map)

    # (4c) stability pruning ON the hardened set
    stab = stability_prune(runs["base-hardened"]["importances"], args.stability_thresh)
    print(f"\n--- STABILITY PRUNING on hardened set (keep stability>={args.stability_thresh}) ---")
    print(f"KEPT ({len(stab['kept'])}): {stab['kept']}")
    print(f"DROPPED ({len(stab['dropped'])}): {stab['dropped']}")
    rfeats["pruned-hardened"] = stab["kept"]
    runs["pruned-hardened"] = walk_forward_dev(ev, stab["kept"], args.first_test,
                                               args.last_test, args.model, cb, regime_map)
    if has_fund:
        rfeats["pruned-hardened+fundz"] = stab["kept"] + HARDENED_FUNDING
        runs["pruned-hardened+fundz"] = walk_forward_dev(ev, rfeats["pruned-hardened+fundz"],
                                                         args.first_test, args.last_test,
                                                         args.model, cb, regime_map)
    if use_metrics:
        rfeats["pruned-hardened+metrics"] = stab["kept"] + METRIC_FEATURES
        runs["pruned-hardened+metrics"] = walk_forward_dev(ev, rfeats["pruned-hardened+metrics"],
                                                           args.first_test, args.last_test,
                                                           args.model, cb, regime_map)

    order = [k for k in ["base-hardened", "base-hardened+fundz", "base-hardened+metrics",
                         "pruned-hardened", "pruned-hardened+fundz",
                         "pruned-hardened+metrics"] if k in runs]
    tbl = pd.DataFrame([_pooled_row(k, runs[k], rfeats[k], cb) for k in order])
    print("\n--- WF COMPARISON (runs a-d) ---")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(tbl.to_string(index=False))

    # (4d) best combo by pooled EV sumR: per-fold AUC + per-regime
    best = max(order, key=lambda k: pooled_metrics(runs[k]["preds"], cb)["ev_Rm"])
    print(f"\n--- BEST combo: {best} ---")
    print("per-fold AUC:")
    with pd.option_context("display.width", 240):
        print(runs[best]["folds"][["month", "regime", "auc", "n_test",
                                   "ev_n", "ev_Rm"]].to_string(index=False))
    print(f"\n--- PER-REGIME breakdown ({best}) ---")
    print(regime_table(runs[best]["folds"]).to_string(index=False))
    # also print per-regime for base-hardened (does chop still bleed w/o funding?)
    if best != "base-hardened":
        print("\n--- PER-REGIME breakdown (base-hardened) ---")
        print(regime_table(runs["base-hardened"]["folds"]).to_string(index=False))

    extras = {
        "adversarial": adv_reports,
        "stability": {
            "kept": stab["kept"],
            "dropped": stab["dropped"],
            "summary": stab["summary"],
        },
        "regime_best": regime_table(runs[best]["folds"]),
        "best_variant": best,
    }
    save_run_artifacts(args, ev, runs, rfeats, extras)


# --- CLI --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Scanner-v3 Phase B harness")
    ap.add_argument("--dataset", default=str(REPO_ROOT / "research" / "data" / "events_v3a1.parquet"))
    ap.add_argument("--klines", default=str(BTC_KLINES))
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "lr"] + list(REGISTRY.keys()))
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--first-test", default="2025-04")
    ap.add_argument("--last-test", default="2026-07")
    ap.add_argument("--with-flow", action="store_true", help="append A1 taker-flow aggregates")
    ap.add_argument("--with-perp-funding", action="store_true",
                    help="A2: evaluate the 4 funding features (base/+funding/pruned/pruned+funding)")
    ap.add_argument("--with-perp-metrics", action="store_true",
                    help="A3: evaluate the 7 OI/positioning metrics features (opt-in)")
    ap.add_argument("--harden", action="store_true",
                    help="drift-hardening: replace raw epoch-carrier features with self-normalized "
                         "trailing-z variants; adversarial re-check + WF re-runs (a-d)")
    ap.add_argument("--stability-thresh", type=float, default=0.5)
    ap.add_argument("--artifact-dir", default="",
                    help="optional directory to write summary JSON, folds CSV, predictions parquet, features, importances")
    ap.add_argument("--run-name", default="scanner-v3-run",
                    help="artifact subdirectory suffix when --artifact-dir is set")
    ap.add_argument("--eval-features-file", default="",
                    help="evaluate exactly this newline-delimited feature list; useful for Phase C model replay")
    ap.add_argument("--open-vault", choices=["forward", "backward"], default=None,
                    help="PHASE D: BURN a vault and evaluate on it (logs the opening)")
    args = ap.parse_args()

    feats = list(FEATURES) + (list(FLOW_FEATURES) if args.with_flow else [])
    ev = prep(pd.read_parquet(args.dataset))
    print(f"dataset: {args.dataset}")
    print(f"eligible events: {len(ev)}  resolved: {int(ev['outcome'].isin(['tp','sl']).sum())}  "
          f"features: {len(feats)} ({'+flow' if args.with_flow else 'base'})  model: {args.model}")
    print(f"as_of span: {pd.to_datetime(ev['as_of'].min(),unit='ms',utc=True)} .. "
          f"{pd.to_datetime(ev['as_of'].max(),unit='ms',utc=True)}")

    if args.open_vault:
        if args.eval_features_file:
            fpath = Path(args.eval_features_file)
            eval_feats = [ln.strip() for ln in fpath.read_text(encoding="utf-8").splitlines()
                          if ln.strip()]
            missing = [f for f in eval_feats if f not in ev.columns]
            if missing and any(f.endswith("_hz") for f in missing):
                ev, _ = add_hardened_features(ev)
                missing = [f for f in eval_feats if f not in ev.columns]
            assert not missing, f"features absent from dataset/in-memory hardening: {missing}"
            feats = eval_feats
            print(f"frozen feature list: {fpath}  n_feat={len(feats)}")
        regime_map, _ = btc_month_regime(Path(args.klines))
        r, vpreds = open_vault_eval(ev, args.open_vault, feats, args.model, args.cost_bps,
                                    regime_map)
        print(f"\n=== VAULT EVAL ({args.open_vault}) ===")
        print(json.dumps(_json_clean(r), indent=2))
        if args.artifact_dir and "error" not in r:
            root = Path(args.artifact_dir)
            vts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.run_name)
            outd = root / f"{vts}_{safe}"
            outd.mkdir(parents=True, exist_ok=False)
            (outd / "vault_result.json").write_text(
                json.dumps(_json_clean({"argv": sys.argv, "result": r}), indent=2),
                encoding="utf-8")
            (outd / "features.txt").write_text("\n".join(feats) + "\n", encoding="utf-8")
            vcols = [c for c in ["as_of", "month", "regime", "symbol", "outcome", "p_hat",
                                 "tp_rr", "d_final", "r_mkt_eval", "r_ret_eval",
                                 "retest_filled", "ev_val", "selected_ev"]
                     if c in vpreds.columns]
            vpreds[vcols].to_parquet(outd / f"vault_{args.open_vault}.preds.parquet",
                                     index=False)
            print(f"\n=== ARTIFACTS WRITTEN ===\n{outd}")
        return

    print_vault_banner(ev)

    # regime labels
    regime_map, det = btc_month_regime(Path(args.klines))
    daily_counts = det["state"].value_counts().to_dict()
    month_counts = pd.Series(regime_map).value_counts().to_dict()
    print("\n=== REGIME LABELS (source: BTC 1h klines -> daily close vs EMA50 + 30d drawdown) ===")
    print(f"daily-state counts: {daily_counts}")
    print(f"month-class counts: {month_counts}")
    print(pd.Series(regime_map).to_string())

    if args.eval_features_file:
        fpath = Path(args.eval_features_file)
        eval_feats = [ln.strip() for ln in fpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        missing = [f for f in eval_feats if f not in ev.columns]
        if missing and any(f.endswith("_hz") for f in missing):
            ev, _ = add_hardened_features(ev)
            missing = [f for f in eval_feats if f not in ev.columns]
        assert not missing, f"features absent from dataset/in-memory hardening: {missing}"
        print("\n=== FIXED FEATURE-LIST EVALUATION ===")
        print(f"features file: {fpath}  n_feat={len(eval_feats)}  model={args.model}")
        run = walk_forward_dev(ev, eval_feats, args.first_test, args.last_test,
                               args.model, args.cost_bps, regime_map)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(run["folds"].to_string(index=False))
        print(f"skipped (vault) folds: {run['skipped']}")
        pm = pooled_metrics(run["preds"], args.cost_bps)
        print(f"POOLED fixed: mean_fold_auc={run['folds']['auc'].mean():.4f}  "
              f"pooled_auc={pm['pooled_auc']:.4f}  ev_sel_n={pm['ev_n']}  "
              f"ev_sel_Rm={pm['ev_Rm']:+.1f}")
        print("\n--- per-regime breakdown ---")
        print(regime_table(run["folds"]).to_string(index=False))
        save_run_artifacts(args, ev, {"fixed-features": run}, {"fixed-features": eval_feats},
                           {"eval_features_file": str(fpath)})
        return

    if args.harden:
        run_harden(ev, feats, args, regime_map)
        return

    # deliverable-1 proof: normal vs vault-excluded test months
    normal_months = te.month_starts(args.first_test, args.last_test)
    dev_months = [m for m in normal_months if month_overlaps_vault(m) is None]
    disappeared = [(m, month_overlaps_vault(m)) for m in normal_months if month_overlaps_vault(m)]
    print("\n=== VAULT EXCLUSION PROOF (test folds) ===")
    print(f"normal-run test months ({len(normal_months)}): {normal_months}")
    print(f"dev-run test months  ({len(dev_months)}): {dev_months}")
    print(f"DISAPPEARED (month, vault): {disappeared}")

    # walk-forward BASE
    print("\n=== WALK-FORWARD (base features, vault-disciplined) ===")
    base = walk_forward_dev(ev, feats, args.first_test, args.last_test, args.model,
                            args.cost_bps, regime_map)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(base["folds"].to_string(index=False))
    print(f"skipped (vault) folds: {base['skipped']}")
    fb = base["folds"]
    print(f"POOLED: mean_fold_auc={fb['auc'].mean():.4f}  ev_sel_n={int(fb['ev_n'].sum())}  "
          f"ev_sel_Rm={fb['ev_Rm'].sum():+.1f}")
    print("\n--- per-regime breakdown ---")
    print(regime_table(fb).to_string(index=False))

    # adversarial validation
    print("\n=== ADVERSARIAL VALIDATION (epoch membership, split at median as_of) ===")
    adv = adversarial_validation(ev, feats)
    print(f"split @ {adv['split_utc']}  n_early={adv['n_early']}  n_late={adv['n_late']}")
    print(f"epoch-classifier AUC: {adv['auc']:.4f}   (>>0.5 => drift carriers present)")
    flow_set = set(FLOW_FEATURES)
    top = adv["top15"]
    print("top-15 drift carriers (gain):")
    for f, g in top.items():
        tag = "  <-- FLOW" if f in flow_set else ""
        print(f"  {f:20s} {g:12.1f}{tag}")
    flow_in_top = [f for f in top.index if f in flow_set]
    print(f"flow features in top-15: {flow_in_top if flow_in_top else 'NONE'}")

    # stability pruning
    print("\n=== FEATURE-STABILITY PRUNING ===")
    if not base["importances"]:
        print("no LGBM importances collected (need --model lgbm); skipping.")
        return
    stab = stability_prune(base["importances"], args.stability_thresh)
    print(f"folds used: {len(base['importances'])}  threshold: stability >= {args.stability_thresh}")
    print(f"KEPT ({len(stab['kept'])}): {stab['kept']}")
    print(f"DROPPED ({len(stab['dropped'])}): {stab['dropped']}")
    print("--- stability summary (all features) ---")
    with pd.option_context("display.max_rows", 200):
        print(stab["summary"].to_string())

    print("\n=== WALK-FORWARD (pruned features) ===")
    pruned = walk_forward_dev(ev, stab["kept"], args.first_test, args.last_test, args.model,
                              args.cost_bps, regime_map)
    fp = pruned["folds"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(fp.to_string(index=False))
    print(f"POOLED pruned: mean_fold_auc={fp['auc'].mean():.4f}  ev_sel_n={int(fp['ev_n'].sum())}  "
          f"ev_sel_Rm={fp['ev_Rm'].sum():+.1f}")

    print("\n=== BASE vs PRUNED ===")
    cmp = pd.DataFrame([
        {"variant": "base", "n_feat": len(feats), "mean_fold_auc": round(fb["auc"].mean(), 4),
         "pooled_ev_n": int(fb["ev_n"].sum()), "pooled_ev_Rm": round(fb["ev_Rm"].sum(), 2),
         "ev_avgR": round(fb["ev_Rm"].sum() / max(int(fb["ev_n"].sum()), 1), 4)},
        {"variant": "pruned", "n_feat": len(stab["kept"]), "mean_fold_auc": round(fp["auc"].mean(), 4),
         "pooled_ev_n": int(fp["ev_n"].sum()), "pooled_ev_Rm": round(fp["ev_Rm"].sum(), 2),
         "ev_avgR": round(fp["ev_Rm"].sum() / max(int(fp["ev_n"].sum()), 1), 4)},
    ])
    print(cmp.to_string(index=False))
    artifact_runs = {"base": base, "pruned": pruned}
    artifact_feats = {"base": feats, "pruned": stab["kept"]}
    artifact_extras = {
        "adversarial_base": adv,
        "stability": {
            "kept": stab["kept"],
            "dropped": stab["dropped"],
            "summary": stab["summary"],
        },
    }

    # ---- A2 FUNDING EVALUATION (opt-in) --------------------------------------
    if args.with_perp_funding:
        miss = [c for c in FUNDING_FEATURES if c not in ev.columns]
        assert not miss, f"funding features absent from dataset: {miss} (rebuild with --perp-dir)"
        cov = ev[FUNDING_FEATURES[0]].notna().mean()
        print("\n" + "=" * 78)
        print(f"  A2 FUNDING EVALUATION  (funding populated frac={cov:.3f}, cost {args.cost_bps}bps)")
        print("=" * 78)
        variant_feats = {
            "base": feats,
            "base+funding": feats + FUNDING_FEATURES,
            "pruned": stab["kept"],
            "pruned+funding": stab["kept"] + FUNDING_FEATURES,
        }
        # reuse already-computed base/pruned runs; only run the +funding variants
        runs = {"base": base, "pruned": pruned}
        for name in ("base+funding", "pruned+funding"):
            runs[name] = walk_forward_dev(ev, variant_feats[name], args.first_test,
                                          args.last_test, args.model, args.cost_bps, regime_map)
            artifact_runs[name] = runs[name]
            artifact_feats[name] = variant_feats[name]

        rows = []
        for name in ("base", "base+funding", "pruned", "pruned+funding"):
            r = runs[name]
            pm = pooled_metrics(r["preds"], args.cost_bps)
            rows.append({"variant": name, "n_feat": len(variant_feats[name]),
                         "mean_fold_auc": round(r["folds"]["auc"].mean(), 4),
                         "pooled_auc": round(pm["pooled_auc"], 4),
                         "q5_q1_lift": round(pm["q5_q1_lift"], 4),
                         "ev_n": pm["ev_n"], "ev_sumR": round(pm["ev_Rm"], 2),
                         "ev_avgR": round(pm["ev_avgR"], 4) if pm["ev_n"] else np.nan})
        cmp_f = pd.DataFrame(rows)
        print("\n--- WALK-FORWARD COMPARISON (base / base+funding / pruned / pruned+funding) ---")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(cmp_f.to_string(index=False))

        print("\n--- per-fold AUC (base+funding vs pruned+funding) ---")
        merged = runs["base+funding"]["folds"][["month", "regime", "auc"]].rename(
            columns={"auc": "auc_base+fund"}).merge(
            runs["pruned+funding"]["folds"][["month", "auc"]].rename(
                columns={"auc": "auc_pruned+fund"}), on="month", how="outer")
        with pd.option_context("display.width", 220):
            print(merged.to_string(index=False))

        # best funding variant by pooled EV sum R
        fund_variants = ["base+funding", "pruned+funding"]
        best = max(fund_variants, key=lambda n: pooled_metrics(runs[n]["preds"], args.cost_bps)["ev_Rm"])
        print(f"\n--- PER-REGIME breakdown for BEST funding variant: {best} ---")
        print(regime_table(runs[best]["folds"]).to_string(index=False))

        # adversarial including the 4 funding features -> are they epoch carriers?
        print("\n--- ADVERSARIAL VALIDATION incl. funding (epoch membership) ---")
        adv_f = adversarial_validation(ev, feats + FUNDING_FEATURES)
        artifact_extras["adversarial_funding"] = adv_f
        print(f"epoch-classifier AUC: {adv_f['auc']:.4f}  "
              f"(split {adv_f['split_utc']}, n_early={adv_f['n_early']}, n_late={adv_f['n_late']})")
        fund_set = set(FUNDING_FEATURES)
        print("top-15 drift carriers (gain):")
        for i, (f, g) in enumerate(adv_f["top15"].items(), 1):
            tag = "  <== FUNDING" if f in fund_set else ""
            print(f"  {i:2d}. {f:22s} {g:12.1f}{tag}")
        # positions of ALL 4 funding features in the full ranking (top15 may omit them)
        adv_full = adversarial_validation_full(ev, feats + FUNDING_FEATURES)
        artifact_extras["adversarial_funding_full"] = adv_full
        ranks = {f: r for r, f in enumerate(adv_full.index, 1)}
        print(f"funding feature ranks in FULL {len(adv_full)}-feature adversarial ranking "
              f"(1=strongest epoch/drift carrier):")
        for f in FUNDING_FEATURES:
            print(f"  {f:22s} rank {ranks[f]:3d}/{len(adv_full)}  gain={adv_full[f]:.1f}")

    # ---- A3 OI / POSITIONING METRICS EVALUATION (opt-in) --------------------
    if args.with_perp_metrics:
        miss = [c for c in METRIC_FEATURES if c not in ev.columns]
        assert not miss, f"metric features absent from dataset: {miss} (rebuild with --perp-dir)"
        cov = ev[METRIC_FEATURES].replace([np.inf, -np.inf], np.nan).notna().mean()
        print("\n" + "=" * 78)
        print(f"  A3 OI/METRICS EVALUATION  (cost {args.cost_bps}bps)")
        print("  coverage: " + ", ".join(f"{c}={cov[c]:.3f}" for c in METRIC_FEATURES))
        print("=" * 78)
        variant_feats = {
            "base": feats,
            "base+metrics": feats + METRIC_FEATURES,
            "pruned": stab["kept"],
            "pruned+metrics": stab["kept"] + METRIC_FEATURES,
        }
        runs = {"base": base, "pruned": pruned}
        for name in ("base+metrics", "pruned+metrics"):
            runs[name] = walk_forward_dev(ev, variant_feats[name], args.first_test,
                                          args.last_test, args.model, args.cost_bps, regime_map)
            artifact_runs[name] = runs[name]
            artifact_feats[name] = variant_feats[name]

        rows = []
        for name in ("base", "base+metrics", "pruned", "pruned+metrics"):
            r = runs[name]
            pm = pooled_metrics(r["preds"], args.cost_bps)
            rows.append({"variant": name, "n_feat": len(variant_feats[name]),
                         "mean_fold_auc": round(r["folds"]["auc"].mean(), 4),
                         "pooled_auc": round(pm["pooled_auc"], 4),
                         "q5_q1_lift": round(pm["q5_q1_lift"], 4),
                         "ev_n": pm["ev_n"], "ev_sumR": round(pm["ev_Rm"], 2),
                         "ev_avgR": round(pm["ev_avgR"], 4) if pm["ev_n"] else np.nan})
        cmp_m = pd.DataFrame(rows)
        print("\n--- WALK-FORWARD COMPARISON (base / base+metrics / pruned / pruned+metrics) ---")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(cmp_m.to_string(index=False))

        print("\n--- ADVERSARIAL VALIDATION incl. metrics (epoch membership) ---")
        adv_m = adversarial_validation(ev, feats + METRIC_FEATURES)
        artifact_extras["adversarial_metrics"] = adv_m
        print(f"epoch-classifier AUC: {adv_m['auc']:.4f}  "
              f"(split {adv_m['split_utc']}, n_early={adv_m['n_early']}, n_late={adv_m['n_late']})")
        metric_set = set(METRIC_FEATURES)
        print("top-15 drift carriers (gain):")
        for i, (f, g) in enumerate(adv_m["top15"].items(), 1):
            tag = "  <== METRIC" if f in metric_set else ""
            print(f"  {i:2d}. {f:22s} {g:12.1f}{tag}")
        adv_full = adversarial_validation_full(ev, feats + METRIC_FEATURES)
        artifact_extras["adversarial_metrics_full"] = adv_full
        ranks = {f: r for r, f in enumerate(adv_full.index, 1)}
        print(f"metric feature ranks in FULL {len(adv_full)}-feature adversarial ranking "
              f"(1=strongest epoch/drift carrier):")
        for f in METRIC_FEATURES:
            print(f"  {f:22s} rank {ranks[f]:3d}/{len(adv_full)}  gain={adv_full[f]:.1f}")

    save_run_artifacts(args, ev, artifact_runs, artifact_feats, artifact_extras)


if __name__ == "__main__":
    main()
