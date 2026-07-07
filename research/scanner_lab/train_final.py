"""Train the deployable scanner-v2 model on ALL data and export artifacts.

Procedure identical to the walk-forward folds (validated in train_eval.py):
resolved events only, uniqueness weights, shallow LGBM, Platt calibration on
the last-15% time slice. Artifacts -> research/artifacts/scanner_v2/.

Usage: python train_final.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eval import FEATURES, prep  # noqa: E402
from universe import REPO_ROOT  # noqa: E402

EVENTS = REPO_ROOT / "research" / "data" / "events.parquet"
ART = REPO_ROOT / "research" / "artifacts" / "scanner_v2"


def main() -> None:
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # keep prep() order ([symbol, as_of]) — exact fold-procedure parity:
    # the fold val slice was a symbol-tail split, and folds deployed ~500 trees
    ev = prep(pd.read_parquet(EVENTS))
    train = ev[ev["outcome"].isin(["tp", "sl"])]
    print(f"training on {len(train)} resolved events "
          f"({pd.Timestamp(train['as_of'].min(), unit='ms')} .. "
          f"{pd.Timestamp(train['as_of'].max(), unit='ms')})")

    n_val = max(int(len(train) * 0.15), 100)
    tr, val = train.iloc[:-n_val], train.iloc[-n_val:]
    y = lambda d: (d["outcome"] == "tp").astype(int)  # noqa: E731
    clf = lgb.LGBMClassifier(
        num_leaves=15, min_child_samples=60, learning_rate=0.05,
        n_estimators=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, verbose=-1,
    )
    # fixed 500 trees: per-fold best_iter was 497-500/500, early stop never bound
    clf.fit(tr[FEATURES], y(tr), sample_weight=tr["sw"].to_numpy())
    p_val = clf.predict_proba(val[FEATURES])[:, 1]
    print(f"val AUC (calibration slice): {roc_auc_score(y(val), p_val):.4f}")
    cal = LogisticRegression(max_iter=1000)
    cal.fit(p_val.reshape(-1, 1), y(val))

    ART.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(ART / "model.txt"))
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "calibration": {"type": "platt", "coef": float(cal.coef_[0][0]),
                        "intercept": float(cal.intercept_[0])},
        "label": "tp(2.5R plan tp_rr) before sl(1.0x d_final), 24h horizon, 5m paths",
        "training_events_resolved": int(len(train)),
        "training_window": [
            pd.Timestamp(train["as_of"].min(), unit="ms").isoformat(),
            pd.Timestamp(train["as_of"].max(), unit="ms").isoformat()],
        "selection_policy": {
            "side": "long", "pattern": "P1", "regime": "btc_close>ema50_1h",
            "rule": "EV = p*tp_rr - (1-p) - cost_bps/1e4/d_final > 0",
            "cost_bps_rt": 25, "dedup": "one signal per symbol per UTC day",
            "note": "EV stays in label units (tp_rr from plan, 2.0-2.5)"},
        "execution_recommendation": {
            "entry": "market/next_open (NOT retest - retest hurts these picks)",
            "stop": "1.0x d_final (production noise-floor plan)",
            "tp_rr": "3.0-3.5 (exit sweep: rr3-4 @ 24h plateau, slot3_R +55..+77 vs +38 at 2.5)",
            "max_hold_h": 24, "slots": "3-4 concurrent", "symbol_cooldown_h": 24},
        "status": "EXPERIMENTAL - no deployable edge demonstrated (see below)",
        "walk_forward_oos": {
            "months": "2025-04..2026-07", "auc": 0.542,
            "note": ("after fixing the breadth epoch-leak, per-event money lift "
                     "of the ML filter is ~0 at 25bps: slots=3 long-P1-bull EV>0 "
                     "+17R/16mo (6/16 months positive) vs no-ML control -83R; "
                     "causally-calibrated p quintiles are FLAT (0.235..0.244). "
                     "Use p_hat as advisory rank only, not as an edge claim."),
            "pre_fix_artifact_warning": (
                "an earlier build showed +38R slots=3; traced to breadth_ema50 "
                "counting unlisted coins as below-EMA (epoch proxy). Fixed in "
                "features.py; numbers here are post-fix."),
            "no_ml_control_slot3_R": -83.0,
            "cost_sensitivity": "breakeven moves to ~10bps RT; costs dominate"},
    }
    (ART / "meta.json").write_text(json.dumps(meta, indent=1))
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    (ART / "feature_importance.txt").write_text(imp.to_string())
    print(f"artifacts -> {ART}")
    print(imp.head(12).to_string())


if __name__ == "__main__":
    main()
