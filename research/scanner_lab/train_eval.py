"""Walk-forward training + policy evaluation for the scanner meta-model.

Design (per research-ml-meta-labeling.md):
- binary target: tp-before-sl on resolved events only
- LR = honesty bar; shallow regularized LightGBM = candidate
- expanding walk-forward, monthly test folds, 24h purge before test start
- per-symbol uniqueness sample weights (overlapping 24h horizons)
- Platt calibration on time-tail validation slice
- decision: EV = p*rr - (1-p) - cost_r > 0, cost_r = cost_bps/1e4/d_final
- money metrics on ALL outcomes of selected events (ambiguous counted as sl —
  conservative), vs baselines incl. flow-matched top-N

Usage: python train_eval.py [--cost-bps 25] [--first-test 2025-04] [--model lgbm]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402

warnings.filterwarnings("ignore")

EVENTS = REPO_ROOT / "research" / "data" / "events.parquet"
PRED_OUT = REPO_ROOT / "research" / "data" / "predictions.parquet"

FEATURES = [
    "roc_1h", "roc_4h", "roc_12h", "roc_24h", "roc_72h", "roc_168h",
    "vol_ratio", "rvol_hod", "qvol_z168", "trades_z168",
    "taker_share", "taker_share_4h",
    "rsi14", "atr_pct", "atr_ratio_72", "ema20_ext_atr", "breakout_dist_atr",
    "z20", "rexp72", "med_range24", "dist_hh168_atr", "bbw_pctile_720",
    "ma_align", "ema50_slope_24h",
    "d_final", "tp_rr", "p1h_age", "hour", "dow",
    "tier", "uni_adr_pct", "uni_log_spot30",
    "breadth_ema50", "breadth_roc24_pos", "median_roc24", "median_roc4",
    "btc_roc_4h", "btc_roc_24h", "btc_z20", "btc_vol_24h", "btc_above_ema50",
    "altseason", "rank_roc4", "rank_roc24", "rank_volratio",
    "pat_P1", "pat_P1H", "pat_P2", "pat_P3", "is_long",
]
# scanner-v3 A1 multi-horizon taker-flow aggregates (opt-in via --with-flow)
FLOW_FEATURES = [
    "taker_ema_24h", "taker_ema_72h", "taker_ema_168h", "taker_z168",
    "flow_z_24h", "flow_z_72h", "flow_div_24h", "flow_div_72h",
    "flow_z24_rank",
]
HORIZON_MS = 24 * 3_600_000
RETEST_P = 0.4


def prep(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[ev["stop_feasible"] & ~ev["outcome"].isin(["missing_5m", "invalid_plan"])].copy()
    for p in ("P1", "P1H", "P2", "P3"):
        ev[f"pat_{p}"] = (ev["pattern"] == p).astype(int)
    ev["is_long"] = (ev["side"] == "long").astype(int)
    # conservative money accounting
    ev["r_mkt_eval"] = np.where(ev["outcome"] == "ambiguous", -1.0, ev["r_market"])
    ev["r_ret_eval"] = np.where(ev["retest_outcome"] == "ambiguous",
                                -(1.0 - RETEST_P), ev["r_retest"])
    # per-symbol uniqueness weights: 1 / #overlapping events (same symbol, 24h)
    ev = ev.sort_values(["symbol", "as_of"]).reset_index(drop=True)
    w = np.ones(len(ev))
    for _, g in ev.groupby("symbol", sort=False):
        t = g["as_of"].to_numpy()
        lo = np.searchsorted(t, t - HORIZON_MS, side="left")
        hi = np.searchsorted(t, t + HORIZON_MS, side="right")
        w[g.index] = 1.0 / (hi - lo)
    ev["sw"] = w
    ev["month"] = pd.to_datetime(ev["as_of"], unit="ms", utc=True).dt.strftime("%Y-%m")
    return ev


def month_starts(first: str, last: str) -> list[str]:
    out, cur = [], pd.Timestamp(first + "-01")
    end = pd.Timestamp(last + "-01")
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        cur += pd.offsets.MonthBegin(1)
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    X_tr, y_tr = train[FEATURES], (train["outcome"] == "tp").astype(int)
    X_te = test[FEATURES]
    sw = train["sw"].to_numpy()
    if model == "lr":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000))
        pipe.fit(X_tr, y_tr, logisticregression__sample_weight=sw)
        return pipe.predict_proba(X_te)[:, 1]
    # lgbm with time-tail val for early stop + Platt calibration
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    n_val = max(int(len(train) * 0.15), 100)
    tr, val = train.iloc[:-n_val], train.iloc[-n_val:]
    clf = lgb.LGBMClassifier(
        num_leaves=15, min_child_samples=60, learning_rate=0.05,
        n_estimators=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, verbose=-1,
    )
    clf.fit(tr[FEATURES], (tr["outcome"] == "tp").astype(int),
            sample_weight=tr["sw"].to_numpy(),
            eval_set=[(val[FEATURES], (val["outcome"] == "tp").astype(int))],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    fit_predict.last_model = clf
    fit_predict.best_iters = getattr(fit_predict, "best_iters", [])
    fit_predict.best_iters.append(clf.best_iteration_ or clf.n_estimators)
    p_val = clf.predict_proba(val[FEATURES])[:, 1]
    cal = LogisticRegression(max_iter=1000)
    cal.fit(p_val.reshape(-1, 1), (val["outcome"] == "tp").astype(int))
    p_raw = clf.predict_proba(X_te)[:, 1]
    return cal.predict_proba(p_raw.reshape(-1, 1))[:, 1]


def money(sub: pd.DataFrame, cost_bps: float) -> dict:
    if len(sub) == 0:
        return {"n": 0, "R_mkt": 0.0, "R_ret": 0.0, "avg_mkt": np.nan, "tp_rate": np.nan}
    cost_r = cost_bps / 1e4 / sub["d_final"]
    r_m = sub["r_mkt_eval"] - cost_r
    filled = sub["retest_filled"]
    r_r = sub["r_ret_eval"] - cost_r * filled  # unfilled retest pays no fees
    res = sub[sub["outcome"].isin(["tp", "sl"])]
    return {
        "n": len(sub), "R_mkt": r_m.sum(), "R_ret": r_r.sum(),
        "avg_mkt": r_m.mean(),
        "tp_rate": (res["outcome"] == "tp").mean() if len(res) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--first-test", default="2025-04")
    ap.add_argument("--last-test", default="2026-07")
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "lr", "both"])
    ap.add_argument("--events", default=str(EVENTS))
    ap.add_argument("--drop-static", action="store_true",
                    help="drop tier/uni_* snapshot features (2026-07 leak risk)")
    ap.add_argument("--with-flow", action="store_true",
                    help="append A1 multi-horizon taker-flow aggregate features")
    ap.add_argument("--train-longp1", action="store_true",
                    help="train only on long P1 events (specialist)")
    ap.add_argument("--pred-out", default=str(PRED_OUT))
    args = ap.parse_args()
    global FEATURES
    if args.drop_static:
        FEATURES = [f for f in FEATURES if f not in ("tier", "uni_adr_pct", "uni_log_spot30")]
    if args.with_flow:
        FEATURES = FEATURES + FLOW_FEATURES

    from sklearn.metrics import roc_auc_score

    ev = prep(pd.read_parquet(args.events))
    print(f"eligible events: {len(ev)}, resolved: {ev['outcome'].isin(['tp', 'sl']).sum()}, "
          f"base tp rate: {(ev['outcome'] == 'tp').sum() / max(ev['outcome'].isin(['tp', 'sl']).sum(), 1):.3f}")

    models = ["lgbm", "lr"] if args.model == "both" else [args.model]
    all_rows = []
    preds_frames = []
    importances = []
    for model in models:
        fold_rows = []
        for month in month_starts(args.first_test, args.last_test):
            t_start = int(pd.Timestamp(month + "-01", tz="UTC").timestamp() * 1000)
            t_end = int((pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthBegin(1)).timestamp() * 1000)
            train = ev[(ev["as_of"] < t_start - HORIZON_MS) & ev["outcome"].isin(["tp", "sl"])]
            if args.train_longp1:
                train = train[(train["side"] == "long") & (train["pattern"] == "P1")]
            test = ev[(ev["as_of"] >= t_start) & (ev["as_of"] < t_end)]
            if len(train) < 300 or len(test) == 0:
                continue
            p = fit_predict(train, test, model)
            test = test.copy()
            test["p_hat"] = p
            test["model"] = model
            preds_frames.append(test)
            if model == "lgbm" and hasattr(fit_predict, "last_model"):
                importances.append(pd.Series(
                    fit_predict.last_model.feature_importances_, index=FEATURES))

            res_mask = test["outcome"].isin(["tp", "sl"])
            auc = (roc_auc_score((test.loc[res_mask, "outcome"] == "tp").astype(int),
                                 test.loc[res_mask, "p_hat"])
                   if res_mask.sum() >= 10 and test.loc[res_mask, "outcome"].nunique() > 1
                   else np.nan)
            ev_val = test["p_hat"] * test["tp_rr"] - (1 - test["p_hat"]) \
                - args.cost_bps / 1e4 / test["d_final"]
            sel = test[ev_val > 0]
            base_flow = test[test["is_hard"] | test["is_marginal"]]
            topn = test.nlargest(len(base_flow), "p_hat") if len(base_flow) else test.iloc[:0]
            row = {"model": model, "month": month, "auc": auc,
                   "n_test": len(test), "n_train": len(train)}
            for name, sub in [("ml_ev", sel), ("hard", test[test["is_hard"]]),
                              ("hardmarg", base_flow), ("topn_flow", topn),
                              ("all", test)]:
                m = money(sub, args.cost_bps)
                row[f"{name}_n"] = m["n"]
                row[f"{name}_Rm"] = round(m["R_mkt"], 2)
                row[f"{name}_Rr"] = round(m["R_ret"], 2)
            fold_rows.append(row)
        df = pd.DataFrame(fold_rows)
        all_rows.append(df)
        print(f"\n===== {model} walk-forward (cost {args.cost_bps}bps) =====")
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(df.to_string(index=False))
        print(f"\nOOS mean AUC: {df['auc'].mean():.4f}")
        for name in ["ml_ev", "hard", "hardmarg", "topn_flow", "all"]:
            print(f"{name}: n={df[f'{name}_n'].sum()} R_mkt={df[f'{name}_Rm'].sum():+.1f} "
                  f"R_retest={df[f'{name}_Rr'].sum():+.1f}")

    if preds_frames:
        pd.concat(preds_frames, ignore_index=True).to_parquet(args.pred_out, index=False)
        print(f"\npredictions -> {args.pred_out}")
    if importances:
        imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        print("\n=== LGBM importance (mean gain-splits, top 25) ===")
        print(imp.head(25).to_string())
    if hasattr(fit_predict, "best_iters"):
        print(f"\nper-fold best_iter: {fit_predict.best_iters}")


if __name__ == "__main__":
    main()
