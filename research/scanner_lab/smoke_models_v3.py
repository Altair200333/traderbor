"""Smoke test for the Phase C model zoo (models_v3.py).

Train  : events_v3a2 rows with as_of in [2025-04, 2025-10)  (resolved only)
Test   : [2025-10, 2025-11)
For every model: test AUC, top-decile avg net R (r_mkt_eval - 25bps/d_final),
wall-clock fit+predict seconds. For rankers also: per-bar precision@3 and NDCG@3
vs random and vs LGBM-scores-as-ranker.

Run: F:/projects/traderbor/.venv/Scripts/python.exe research/scanner_lab/smoke_models_v3.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from sklearn.metrics import roc_auc_score  # noqa: E402

import train_eval as te  # noqa: E402
from train_eval import FEATURES, FLOW_FEATURES, prep  # noqa: E402
import models_v3 as mv  # noqa: E402
from models_v3 import PERP_FEATURES, REGISTRY, BAR_KEY  # noqa: E402

DATASET = LAB.parent / "data" / "events_v3a2.parquet"
COST_BPS = 25.0


def net_r(df: pd.DataFrame) -> np.ndarray:
    return df["r_mkt_eval"].to_numpy() - COST_BPS / 1e4 / df["d_final"].to_numpy()


def point_metrics(test: pd.DataFrame, scores: np.ndarray) -> tuple[float, float]:
    res = test["outcome"].isin(["tp", "sl"]).to_numpy()
    y = (test["outcome"].to_numpy() == "tp").astype(int)
    auc = (roc_auc_score(y[res], scores[res])
           if res.sum() >= 10 and len(np.unique(y[res])) > 1 else float("nan"))
    k = max(int(np.ceil(0.10 * len(test))), 1)
    top = np.argsort(-scores)[:k]
    dec_r = float(net_r(test.iloc[top]).mean())
    return auc, dec_r


def rank_metrics(test: pd.DataFrame, scores: np.ndarray) -> tuple[float, float]:
    """per-bar precision@3 (tp fraction in top-3) and NDCG@3 (graded R)."""
    tp = (test["outcome"].to_numpy() == "tp").astype(float)
    rel = mv._graded_relevance(test).astype(float)
    groups = pd.DataFrame({"k": test[BAR_KEY].to_numpy()}).groupby("k").indices
    precs, ndcgs = [], []
    disc = 1.0 / np.log2(np.arange(2, 5))  # positions 1..3
    for _, idx in groups.items():
        if len(idx) < 3:
            continue
        idx = np.asarray(idx)
        order = idx[np.argsort(-scores[idx])][:3]
        precs.append(tp[order].mean())
        dcg = (rel[order] * disc).sum()
        ideal = np.sort(rel[idx])[::-1][:3]
        idcg = (ideal * disc).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return (float(np.mean(precs)) if precs else float("nan"),
            float(np.mean(ndcgs)) if ndcgs else float("nan"))


def lgbm_floor(train, test, feats) -> np.ndarray:
    saved = te.FEATURES
    te.FEATURES = feats
    try:
        return te.fit_predict(train, test, "lgbm")
    finally:
        te.FEATURES = saved


def main() -> None:
    ev = prep(pd.read_parquet(DATASET))
    dt = pd.to_datetime(ev["as_of"], unit="ms", utc=True)
    tr_mask = (dt >= "2025-04-01") & (dt < "2025-10-01") & ev["outcome"].isin(["tp", "sl"])
    te_mask = (dt >= "2025-10-01") & (dt < "2025-11-01")
    train, test = ev[tr_mask].copy(), ev[te_mask].copy()
    feats = list(FEATURES) + list(FLOW_FEATURES) + list(PERP_FEATURES)
    feats = [f for f in feats if f in ev.columns]

    base_tp = (test["outcome"] == "tp").mean()
    print(f"train {len(train)} resolved | test {len(test)} (tp base {base_tp:.3f}) | "
          f"features {len(feats)} (+{len([f for f in feats if f in PERP_FEATURES])} perp)")
    print("=" * 92)

    # --- LGBM floor (train_eval champion) + its ranker baseline ---------------
    rankers = {"xranker_gbm", "xranker_torch"}
    t0 = time.perf_counter()
    lgbm_scores = lgbm_floor(train, test, feats)
    lgbm_sec = time.perf_counter() - t0
    lgbm_auc, lgbm_dec = point_metrics(test, lgbm_scores)
    lgbm_p3, lgbm_ndcg = rank_metrics(test, lgbm_scores)

    # random-ranker baseline (avg over seeds)
    rng = np.random.default_rng(0)
    rp = [rank_metrics(test, rng.random(len(test)))[0] for _ in range(20)]
    rand_p3 = float(np.mean(rp))

    hdr = f"{'model':<15}{'AUC':>8}{'decileR':>10}{'p@3':>8}{'ndcg@3':>9}{'sec':>9}"
    print(hdr); print("-" * 92)
    print(f"{'lgbm(floor)':<15}{lgbm_auc:>8.4f}{lgbm_dec:>10.4f}"
          f"{lgbm_p3:>8.3f}{lgbm_ndcg:>9.3f}{lgbm_sec:>9.1f}")

    results = [("lgbm(floor)", lgbm_auc, lgbm_dec, lgbm_p3, lgbm_ndcg, lgbm_sec)]
    order = ["logreg", "catboost", "xgboost", "tabpfn",
             "xranker_gbm", "xranker_torch", "moe", "multitask"]
    for name in order:
        try:
            t0 = time.perf_counter()
            s = REGISTRY[name](train, test, feats)
            sec = time.perf_counter() - t0
            s = np.asarray(s, dtype=float)
            auc, dec = point_metrics(test, s)
            if name in rankers:
                p3, ndcg = rank_metrics(test, s)
            else:
                p3, ndcg = float("nan"), float("nan")
            print(f"{name:<15}{auc:>8.4f}{dec:>10.4f}"
                  f"{p3:>8.3f}{ndcg:>9.3f}{sec:>9.1f}")
            results.append((name, auc, dec, p3, ndcg, sec))
        except Exception as e:  # a crash is reported, not silently skipped
            import traceback
            print(f"{name:<15}  CRASH: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((name, float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan")))

    print("-" * 92)
    print(f"ranker baselines  ->  random p@3 = {rand_p3:.3f} (~tp base) | "
          f"LGBM-as-ranker p@3 = {lgbm_p3:.3f} ndcg@3 = {lgbm_ndcg:.3f}")


if __name__ == "__main__":
    main()
