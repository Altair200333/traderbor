"""Liqrev ML — STAGE 2: modeling + portfolio + SINGLE holdout shot.

PRE-REGISTERED SPEC (frozen 2026-07-09 BEFORE the first run; every cell below is
reported, nothing is hidden; exactly ONE holdout evaluation happens).

INPUT: research/data/liqrev/ml_dataset.parquet (stage 1, lookahead-checked).
Modeling universe = FILLED events only (unfilled carry no label and the frozen
config skips them). DEV = ts < 2025-01-01, HOLDOUT = ts >= 2025-01-01.

FEATURES (stage-1 pruned 13) + basis_bps_missing flag (14 columns):
  btc_ret_6h, btc_regime, dist_low_30d, oi_turnover, vol_ratio,
  taker_buy_share, wick_frac, ret_6h, doi6, funding_last, mkt_events_24h,
  basis_bps, m1_low_pos, basis_bps_missing.
Preprocessing per training set (never test): NaN -> train median, then z-score
with train mean/std (flag column included, harmless).

EXIT-TIME CONVENTION (dataset stores no exit_ts): frozen config exits at the
close of bar i+24 => outcome KNOWN at ts + 25h (embargo uses this,
conservative); slot frees at ts + 24h (open of exit bar, matching
liqrev_v2.simulate's recorded exit_ts).

DECLARED MODELS (score = "bigger is better"):
  M0 baseline    equal slots on all events (benchmark; == any model x R0).
  M1 dumb rule   score = -btc_ret_6h (no fit).
  M2 ridge       on net_ret; alpha in {0.1,1,10} by inner TimeSeriesSplit(3),
                 selected by mean inner-val Spearman IC vs net_ret.
  M3 logistic    on win; C in {0.1,1,10} by inner TimeSeriesSplit(3), selected
                 by mean inner-val ROC-AUC; score = P(win).
  M4 shallow LGBM on net_ret; HARD CAPS max_depth=3, num_leaves=8,
                 min_child_samples=50, lr=0.05, subsample=0.8 (freq 1),
                 n_estimators<=500, early stopping 30 rounds on the
                 chronological LAST 20% of the training set (model kept is the
                 one fit on the first 80%); seed 42.

CV PROTOCOL (DEV only): 5 chronological equal-count folds over DEV filled
events. For fold k: train = DEV filled events with exit_known < fold_start-7d
(7d embargo); predict the fold. Fold 1 has no train -> excluded. OOS = folds
2..5 pooled. Score normalisation for pooling/rules: every score (train or
test) is mapped to its PERCENTILE IN THAT FOLD'S TRAIN-SCORE DISTRIBUTION
(causal; test never informs thresholds).
Metrics per model: day-clustered bootstrap (1000 iters, resample calendar
days, seed 11) rank IC of OOS percentile-scores vs net_ret; decile lift =
mean net_ret(top decile) - mean net_ret(bottom decile) of OOS scores.

PORTFOLIO RULES (per fold, thresholds/percentiles from TRAIN scores only):
  R0 all-equal: keep all, w=1.
  R1 skip bottom 20%: keep iff train-percentile >= 0.20, w=1.
  R2 skip bottom 40%: keep iff train-percentile >= 0.40, w=1.
  R3 rank-proportional: keep all, w = min(2.0, 2*train_percentile)
     (E[w]=1 under uniform percentiles => same expected total exposure, 2x cap).
Portfolio sim = liqrev_v2 15-slot machinery generalized to weights/filters:
events in ts order; slot busy until ts+24h; taken iff kept & free slot;
eq *= (1 + w*net_ret/15). Report n_kept, net_mean(kept), win(kept), CAGR
(over evaluated span), maxDD. 16 combos = M1..M4 x R0..R3, all on the SAME
pooled OOS events; baseline M0 = the R0 row.

SELECTION (pre-registered): choose ONE model x rule = max DEV OOS CAGR
subject to maxDD <= baseline_maxDD + 2pp AND n_kept >= 60% of OOS events.
MULTIPLE-COMPARISONS COST: 16 combos are compared on DEV; DEV numbers are
optimistically biased and only the single holdout shot is confirmatory.

HOLDOUT SHOT (exactly one): train chosen model on ALL DEV filled events with
exit_known < 2024-12-25 (7d-equivalent embargo before holdout start); compute
rule threshold from THAT training score distribution; apply STATICALLY to all
holdout filled events; run portfolio. DIAGNOSTIC (labelled, non-verdict):
quarterly retrain — for each quarter Q starting 2025-01-01..2026-07-01, train
on all filled events with exit_known < Q start, predict Q's events, threshold
from each train distribution. Also report M1 (chosen rule, DEV-derived
threshold) and M0 baseline on the same holdout events, with by-year tables.

VERDICT RULE (pre-registered): ADOPT iff static holdout CAGR > baseline
holdout CAGR AND holdout maxDD <= baseline maxDD + 2pp AND n_kept >= 60%.
Otherwise REJECT (equal slots stand). If M1 beats the chosen model on holdout
we say so plainly.

Artifacts: research/data/liqrev/results_ml.json.
Usage: python liqrev_ml_model.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liqrev_ml_features import (  # noqa: E402
    ART_DIR, HOLDOUT_START, _spearman,
)

SEED = 42
N_BOOT = 1000
SLOTS = 15
EMBARGO = pd.Timedelta("7D")
EXIT_KNOWN = pd.Timedelta("25h")   # outcome known (close of bar i+24)
SLOT_BUSY = pd.Timedelta("24h")    # slot frees (open of exit bar)
HOLDOUT_TRAIN_CUT = pd.Timestamp("2024-12-25", tz="UTC")

PRUNED = ["btc_ret_6h", "btc_regime", "dist_low_30d", "oi_turnover",
          "vol_ratio", "taker_buy_share", "wick_frac", "ret_6h", "doi6",
          "funding_last", "mkt_events_24h", "basis_bps", "m1_low_pos"]
XCOLS = PRUNED + ["basis_bps_missing"]
MODELS = ["M1_dumb", "M2_ridge", "M3_logit", "M4_lgbm"]
RULES = ["R0_all", "R1_skip20", "R2_skip40", "R3_rankw"]


# ------------------------------------------------------------- preprocessing -
def make_xy(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["filled"]].copy().sort_values("ts").reset_index(drop=True)
    d["basis_bps_missing"] = d["basis_bps"].isna().astype(float)
    d["exit_known"] = d["ts"] + EXIT_KNOWN
    d["exit_slot"] = d["ts"] + SLOT_BUSY
    return d


class Prep:
    """train-only median impute + z-score."""

    def fit(self, X: pd.DataFrame) -> "Prep":
        self.med = X.median()
        Xi = X.fillna(self.med)
        self.mu, self.sd = Xi.mean(), Xi.std().replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return ((X.fillna(self.med) - self.mu) / self.sd).to_numpy()


# ------------------------------------------------------------------- models --
def _inner_splits(n: int):
    return TimeSeriesSplit(n_splits=3).split(np.arange(n))


def fit_score(model: str, tr: pd.DataFrame, te: pd.DataFrame
              ) -> tuple[np.ndarray, np.ndarray]:
    """returns (train_scores, test_scores), bigger = better."""
    if model == "M1_dumb":
        return -tr["btc_ret_6h"].to_numpy(), -te["btc_ret_6h"].to_numpy()
    prep = Prep().fit(tr[XCOLS])
    Xtr, Xte = prep.transform(tr[XCOLS]), prep.transform(te[XCOLS])
    y_ret = tr["net_ret"].to_numpy()
    y_win = tr["win"].to_numpy()
    if model == "M2_ridge":
        best, best_ic = None, -np.inf
        for a in (0.1, 1.0, 10.0):
            ics = []
            for itr, iva in _inner_splits(len(tr)):
                m = Ridge(alpha=a).fit(Xtr[itr], y_ret[itr])
                ics.append(_spearman(m.predict(Xtr[iva]), y_ret[iva]))
            ic = float(np.nanmean(ics))
            if ic > best_ic:
                best_ic, best = ic, a
        m = Ridge(alpha=best).fit(Xtr, y_ret)
        return m.predict(Xtr), m.predict(Xte)
    if model == "M3_logit":
        best, best_auc = None, -np.inf
        for c in (0.1, 1.0, 10.0):
            aucs = []
            for itr, iva in _inner_splits(len(tr)):
                if len(np.unique(y_win[itr])) < 2 or len(np.unique(y_win[iva])) < 2:
                    continue
                m = LogisticRegression(C=c, max_iter=2000).fit(Xtr[itr], y_win[itr])
                aucs.append(roc_auc_score(y_win[iva], m.predict_proba(Xtr[iva])[:, 1]))
            auc = float(np.nanmean(aucs)) if aucs else np.nan
            if np.isfinite(auc) and auc > best_auc:
                best_auc, best = auc, c
        m = LogisticRegression(C=best or 1.0, max_iter=2000).fit(Xtr, y_win)
        return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]
    if model == "M4_lgbm":
        cut = max(1, int(len(tr) * 0.8))
        m = lgb.LGBMRegressor(
            max_depth=3, num_leaves=8, min_child_samples=50,
            learning_rate=0.05, subsample=0.8, subsample_freq=1,
            n_estimators=500, random_state=SEED, verbosity=-1)
        m.fit(Xtr[:cut], y_ret[:cut],
              eval_set=[(Xtr[cut:], y_ret[cut:])] if cut < len(tr) else None,
              callbacks=[lgb.early_stopping(30, verbose=False)]
              if cut < len(tr) else None)
        return m.predict(Xtr), m.predict(Xte)
    raise ValueError(model)


def pctl(train_scores: np.ndarray, s: np.ndarray) -> np.ndarray:
    """percentile of each s in the train-score distribution (causal)."""
    ts = np.sort(train_scores[np.isfinite(train_scores)])
    if len(ts) == 0:
        return np.full(len(s), 0.5)
    return np.searchsorted(ts, s, side="right") / len(ts)


# ---------------------------------------------------------------- portfolio --
def run_portfolio(ev: pd.DataFrame, keep: np.ndarray, w: np.ndarray) -> dict:
    """15-slot sequential sim (liqrev_v2 machinery + weights/filters)."""
    order = np.argsort(ev["ts"].to_numpy())
    eq, busy, curve = 1.0, [], []
    kept_ret, n_taken = [], 0
    ts_arr = ev["ts"].to_numpy()[order]
    ex_arr = ev["exit_slot"].to_numpy()[order]
    r_arr = ev["net_ret"].to_numpy()[order]
    k_arr = keep[order]
    w_arr = w[order]
    for i in range(len(ev)):
        busy = [b for b in busy if b > ts_arr[i]]
        if k_arr[i] and len(busy) < SLOTS:
            eq *= (1 + w_arr[i] * r_arr[i] / SLOTS)
            busy.append(ex_arr[i])
            n_taken += 1
            kept_ret.append(r_arr[i])
        curve.append((ts_arr[i], eq))
    c = pd.Series({t: v for t, v in curve})
    years = max((c.index[-1] - c.index[0]).days, 1) / 365.25
    kr = np.array(kept_ret)
    n_kept = int(keep.sum())
    return {"n_events": int(len(ev)), "n_kept": n_kept,
            "kept_frac": round(n_kept / len(ev), 3),
            "n_taken": n_taken,
            "net_mean": round(float(kr.mean()), 5) if len(kr) else None,
            "win": round(float((kr > 0).mean()), 4) if len(kr) else None,
            "total": round(float(eq - 1), 4),
            "cagr": round(float(eq ** (1 / years) - 1), 4),
            "maxDD": round(float((c / c.cummax() - 1).min()), 4)}


def apply_rule(rule: str, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rule == "R0_all":
        return np.ones(len(p), bool), np.ones(len(p))
    if rule == "R1_skip20":
        return p >= 0.20, np.ones(len(p))
    if rule == "R2_skip40":
        return p >= 0.40, np.ones(len(p))
    if rule == "R3_rankw":
        return np.ones(len(p), bool), np.minimum(2.0, 2.0 * p)
    raise ValueError(rule)


# ---------------------------------------------------------------- metrics ----
def clustered_ic(ts: pd.Series, x: np.ndarray, y: np.ndarray,
                 seed: int = 11) -> dict:
    ic = _spearman(x, y)
    days = ts.dt.floor("1D").to_numpy()
    uniq = np.unique(days)
    groups = {d: np.where(days == d)[0] for d in uniq}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(N_BOOT):
        sel = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([groups[uniq[j]] for j in sel])
        bs.append(_spearman(x[idx], y[idx]))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return {"ic": round(ic, 4) if np.isfinite(ic) else None,
            "ci_lo": round(float(np.percentile(bs, 2.5)), 4) if len(bs) else None,
            "ci_hi": round(float(np.percentile(bs, 97.5)), 4) if len(bs) else None}


def decile_lift(p: np.ndarray, y: np.ndarray) -> float | None:
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() < 50:
        return None
    q = pd.qcut(pd.Series(p[m]).rank(method="first"), 10, labels=False)
    g = pd.Series(y[m]).groupby(q).mean()
    return round(float(g.iloc[-1] - g.iloc[0]), 5)


# ------------------------------------------------------------------ main -----
def main() -> None:
    df = make_xy(pd.read_parquet(ART_DIR / "ml_dataset.parquet"))
    dev = df[df["ts"] < HOLDOUT_START].reset_index(drop=True)
    hold = df[df["ts"] >= HOLDOUT_START].reset_index(drop=True)
    assert (dev["ts"] < HOLDOUT_START).all()
    print(f"filled events: dev={len(dev)} holdout={len(hold)}")

    # ---- walk-forward CV on DEV --------------------------------------------
    fold_id = pd.qcut(np.arange(len(dev)), 5, labels=False)
    oos = {m: [] for m in MODELS}         # rows: (index, pctl_score, fold)
    for k in range(1, 5):
        te = dev[fold_id == k]
        fold_start = te["ts"].min()
        tr = dev[dev["exit_known"] < fold_start - EMBARGO]
        print(f"fold {k+1}/5: train={len(tr)} test={len(te)} "
              f"(train exit < {fold_start - EMBARGO:%Y-%m-%d %H:%M})")
        for m in MODELS:
            s_tr, s_te = fit_score(m, tr, te)
            oos[m].append(pd.DataFrame(
                {"idx": te.index, "p": pctl(s_tr, s_te), "fold": k + 1}))
    model_stats, oos_p = [], {}
    for m in MODELS:
        o = pd.concat(oos[m], ignore_index=True)
        sub = dev.loc[o["idx"]]
        p = o["p"].to_numpy()
        oos_p[m] = (sub, p)
        st = clustered_ic(sub["ts"], p, sub["net_ret"].to_numpy())
        st.update({"model": m,
                   "decile_lift": decile_lift(p, sub["net_ret"].to_numpy()),
                   "n_oos": len(sub)})
        model_stats.append(st)
    print("\n=== DEV OOS model metrics (day-clustered IC, folds 2..5) ===")
    print(pd.DataFrame(model_stats)[
        ["model", "n_oos", "ic", "ci_lo", "ci_hi", "decile_lift"]].to_string(index=False))

    # ---- 16-combo DEV portfolio grid ---------------------------------------
    any_sub = oos_p[MODELS[0]][0]
    baseline = run_portfolio(any_sub, np.ones(len(any_sub), bool),
                             np.ones(len(any_sub)))
    print(f"\nM0 baseline (equal slots, OOS events): {json.dumps(baseline)}")
    grid = []
    for m in MODELS:
        sub, p = oos_p[m]
        for r in RULES:
            keep, w = apply_rule(r, p)
            res = run_portfolio(sub, keep, w)
            res.update({"model": m, "rule": r})
            grid.append(res)
    gt = pd.DataFrame(grid)[["model", "rule", "n_kept", "kept_frac",
                             "net_mean", "win", "cagr", "maxDD"]]
    print("\n=== 16-combo DEV grid (NOTE: 16 comparisons -> optimistic) ===")
    print(gt.to_string(index=False))

    # ---- selection -----------------------------------------------------------
    ok = gt[(gt["maxDD"] >= baseline["maxDD"] - 0.02 - 1e-12) &
            (gt["kept_frac"] >= 0.60)]
    chosen = ok.sort_values("cagr", ascending=False).iloc[0]
    print(f"\nCHOSEN: {chosen['model']} x {chosen['rule']} "
          f"(dev cagr={chosen['cagr']}, maxDD={chosen['maxDD']}, "
          f"kept={chosen['kept_frac']})")

    # ---- HOLDOUT SHOT (single) ----------------------------------------------
    tr_hold = dev[dev["exit_known"] < HOLDOUT_TRAIN_CUT]
    print(f"\nholdout static train: {len(tr_hold)} events "
          f"(exit_known < {HOLDOUT_TRAIN_CUT:%Y-%m-%d})")

    def hold_eval(model: str, rule: str) -> tuple[dict, dict]:
        s_tr, s_te = fit_score(model, tr_hold, hold)
        p = pctl(s_tr, s_te)
        keep, w = apply_rule(rule, p)
        res = run_portfolio(hold, keep, w)
        kept = hold[keep]
        by_year = (kept.groupby("year")["net_ret"]
                   .agg(n="count", mean="mean",
                        win=lambda x: float((x > 0).mean())).round(4))
        return res, {str(y): dict(r) for y, r in by_year.iterrows()}

    static_res, static_by = hold_eval(str(chosen["model"]), str(chosen["rule"]))
    m1_res, m1_by = hold_eval("M1_dumb", str(chosen["rule"]))
    base_res = run_portfolio(hold, np.ones(len(hold), bool), np.ones(len(hold)))
    base_by = (hold.groupby("year")["net_ret"]
               .agg(n="count", mean="mean",
                    win=lambda x: float((x > 0).mean())).round(4))
    base_by = {str(y): dict(r) for y, r in base_by.iterrows()}

    # diagnostic quarterly retrain (verdict number is STATIC)
    qs = pd.date_range("2025-01-01", "2026-07-01", freq="QS", tz="UTC")
    keep_q = np.zeros(len(hold), bool)
    w_q = np.ones(len(hold))
    for q0 in qs:
        q1 = q0 + pd.offsets.QuarterBegin(startingMonth=1)
        m_q = (hold["ts"] >= q0) & (hold["ts"] < q1)
        if not m_q.any():
            continue
        tr_q = df[df["exit_known"] < q0]
        s_tr, s_te = fit_score(str(chosen["model"]), tr_q, hold[m_q])
        p = pctl(s_tr, s_te)
        k, w = apply_rule(str(chosen["rule"]), p)
        keep_q[m_q.to_numpy()] = k
        w_q[m_q.to_numpy()] = w
    diag_res = run_portfolio(hold, keep_q, w_q)

    print("\n=== HOLDOUT (2025-26) ===")
    cmp = pd.DataFrame([
        {"who": "M0_baseline", **base_res},
        {"who": f"{chosen['model']}x{chosen['rule']}_STATIC", **static_res},
        {"who": f"{chosen['model']}x{chosen['rule']}_QRETRAIN(diag)", **diag_res},
        {"who": f"M1_dumb x {chosen['rule']}", **m1_res},
    ])[["who", "n_kept", "kept_frac", "net_mean", "win", "cagr", "maxDD"]]
    print(cmp.to_string(index=False))
    print("\nby-year baseline:", json.dumps(base_by))
    print("by-year static:", json.dumps(static_by))
    print("by-year M1:", json.dumps(m1_by))

    adopt = (static_res["cagr"] > base_res["cagr"]
             and static_res["maxDD"] >= base_res["maxDD"] - 0.02 - 1e-12
             and static_res["kept_frac"] >= 0.60)
    verdict = "ADOPT" if adopt else "REJECT (equal slots stand)"
    print(f"\nVERDICT (pre-registered rule): {verdict}")

    out = {"run_utc": datetime.now(timezone.utc).isoformat(),
           "spec": "see module docstring (pre-registered)",
           "n_dev_filled": len(dev), "n_holdout_filled": len(hold),
           "model_ic_table": model_stats,
           "dev_baseline_M0": baseline,
           "dev_grid_16": grid,
           "chosen": {"model": str(chosen["model"]), "rule": str(chosen["rule"]),
                      "dev_cagr": float(chosen["cagr"]),
                      "note": "16 combos compared on DEV -> optimistic bias"},
           "holdout": {"baseline_M0": base_res, "static": static_res,
                       "quarterly_retrain_DIAGNOSTIC": diag_res,
                       "M1_dumb_same_rule": m1_res,
                       "by_year": {"baseline": base_by, "static": static_by,
                                   "m1": m1_by}},
           "verdict": verdict}
    (ART_DIR / "results_ml.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nartifacts -> {ART_DIR / 'results_ml.json'}")


if __name__ == "__main__":
    main()
