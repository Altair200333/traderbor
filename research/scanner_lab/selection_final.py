"""Final selection design on fixed-feature predictions.

Principled rules only (no threshold shopping):
1. OOS-causal calibration: for month k, fit Platt on pooled OOS predictions of
   months < k (true out-of-sample calibration), then EV>0.
2. Capacity-rank: top-K per month by p_hat (K = realistic agent throughput).
Both evaluated under the slot simulator, vs no-ML control and raw-Platt EV>0.

Usage: python selection_final.py [--pred ../data/predictions.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_sim import report, run_slots  # noqa: E402
from universe import REPO_ROOT  # noqa: E402

PRED = REPO_ROOT / "research" / "data" / "predictions.parquet"


def platt(p_tr: np.ndarray, y_tr: np.ndarray, p_te: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=1000)
    m.fit(p_tr.reshape(-1, 1), y_tr)
    return m.predict_proba(p_te.reshape(-1, 1))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(PRED))
    ap.add_argument("--cost-bps", type=float, default=25.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.pred)
    df = df[df["model"] == "lgbm"].copy()
    cost_r = args.cost_bps / 1e4 / df["d_final"]
    df["rm"] = df["r_mkt_eval"] - cost_r
    df["day"] = pd.to_datetime(df["as_of"], unit="ms", utc=True).dt.date
    months = sorted(df["month"].unique())

    # OOS-causal recalibration (needs >=3 prior months)
    df["p_cal"] = np.nan
    for i, m in enumerate(months):
        if i < 3:
            continue
        prior = df[df["month"].isin(months[:i]) & df["outcome"].isin(["tp", "sl"])]
        cur = df["month"] == m
        df.loc[cur, "p_cal"] = platt(
            prior["p_hat"].to_numpy(), (prior["outcome"] == "tp").astype(int).to_numpy(),
            df.loc[cur, "p_hat"].to_numpy())
    dfe = df[df["p_cal"].notna()].copy()
    n_months = dfe["month"].nunique()
    print(f"evaluation window: {dfe['month'].min()}..{dfe['month'].max()} ({n_months} months)")

    long_ = dfe["side"] == "long"
    p1 = dfe["pattern"] == "P1"
    bull = dfe["btc_above_ema50"] == 1
    dfe["ev_cal"] = dfe["p_cal"] * dfe["tp_rr"] - (1 - dfe["p_cal"]) \
        - args.cost_bps / 1e4 / dfe["d_final"]

    def sd_dedup(x: pd.DataFrame) -> pd.DataFrame:
        return (x.sort_values("as_of").groupby(["symbol", "day"], as_index=False)
                .head(1).sort_values("as_of"))

    rows = []
    # rule 1: causal-calibrated EV>0 within long-P1-bull
    for name, mask in [("lpb", long_ & p1 & bull), ("long-any", long_), ("lp1", long_ & p1)]:
        stream = sd_dedup(dfe[mask & (dfe["ev_cal"] > 0)])
        rows.append(report(run_slots(stream, 3), f"calEV>0 {name} slots3"))
    # rule 2: capacity-rank top-K/month by p_hat (within long-P1-bull, sym-day dedup first)
    base = sd_dedup(dfe[long_ & p1 & bull])
    for k in (30, 45, 60):
        sel = (base.sort_values("p_hat", ascending=False).groupby("month").head(k)
               .sort_values("as_of"))
        rows.append(report(run_slots(sel, 3), f"rank top{k}/mo lpb slots3"))
    # references
    rows.append(report(run_slots(base, 3), "no-ML lpb slots3 (control)"))
    raw_ev = sd_dedup(dfe[long_ & p1 & bull & (
        dfe["p_hat"] * dfe["tp_rr"] - (1 - dfe["p_hat"])
        - args.cost_bps / 1e4 / dfe["d_final"] > 0)])
    rows.append(report(run_slots(raw_ev, 3), "rawEV>0 lpb slots3 (ref)"))
    prodflow = sd_dedup(dfe[dfe["is_hard"] | dfe["is_marginal"]])
    rows.append(report(run_slots(prodflow, 3), "prod hard+marg slots3 (ref)"))

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(out.to_string(index=False))

    # calibration quality of p_cal
    res = dfe[dfe["outcome"].isin(["tp", "sl"])].copy()
    res["dec"] = pd.qcut(res["p_cal"], 5, duplicates="drop", labels=False)
    print("\np_cal quintiles (resolved): p_cal_mean vs realized tp")
    print(res.groupby("dec").agg(n=("p_cal", "size"), p=("p_cal", "mean"),
                                 tp=("outcome", lambda s: (s == "tp").mean())).round(3).to_string())


if __name__ == "__main__":
    main()
