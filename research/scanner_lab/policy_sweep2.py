"""Round 2: honest units (one event per symbol-day) + causal day-heat gate.

Hypotheses from round 1:
H1: uncapped EV>0 profit is inflated by same-symbol same-move repeats.
H2: within-day, extreme p_hat/extension = chase = bad; earliest/least-extended better.
H3: the real signal is day-level "heat" (breadth of simultaneous breakouts);
    causal proxy = count of EV>0 long-P1-bull events in trailing 24h.

Usage: python policy_sweep2.py [--model lgbm] [--cost-bps 25]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402

PRED = REPO_ROOT / "research" / "data" / "predictions.parquet"


def stats(sub: pd.DataFrame, months: int, name: str) -> dict:
    if len(sub) == 0:
        return {"policy": name, "n": 0}
    m = sub.groupby("month")["rm"].sum()
    res = sub[sub["outcome"].isin(["tp", "sl"])]
    return {
        "policy": name, "n": len(sub), "per_mo": round(len(sub) / months, 1),
        "tp": round((res["outcome"] == "tp").mean(), 3) if len(res) else np.nan,
        "R": round(sub["rm"].sum(), 1), "avg": round(sub["rm"].mean(), 3),
        "R_ret": round(sub["rr_"].sum(), 1),
        "mo_pos": f"{(m > 0).sum()}/{len(m)}", "worst": round(m.min(), 1),
        "shp": round(m.mean() / m.std(), 2) if len(m) > 2 and m.std() > 0 else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--cost-bps", type=float, default=25.0)
    args = ap.parse_args()

    df = pd.read_parquet(PRED)
    df = df[df["model"] == args.model].copy()
    cost_r = args.cost_bps / 1e4 / df["d_final"]
    df["rm"] = df["r_mkt_eval"] - cost_r
    df["rr_"] = df["r_ret_eval"] - cost_r * df["retest_filled"]
    df["ev"] = df["p_hat"] * df["tp_rr"] - (1 - df["p_hat"]) - cost_r
    df["day"] = pd.to_datetime(df["as_of"], unit="ms", utc=True).dt.date
    months = df["month"].nunique()

    base_mask = (df["side"] == "long") & (df["pattern"] == "P1") & (df["btc_above_ema50"] == 1)
    evpos = df[base_mask & (df["ev"] > 0)].sort_values("as_of")

    # causal trailing-24h heat: count of EV>0 long-P1-bull events (incl. same bar)
    t = evpos["as_of"].to_numpy()
    lo = np.searchsorted(t, t - 24 * 3_600_000, side="left")
    hi = np.searchsorted(t, t, side="right")          # includes same-timestamp batch
    evpos["heat24"] = hi - lo

    # H1: one per symbol-day, FIRST trigger chronologically
    first_sd = evpos.groupby(["symbol", "day"], as_index=False).head(1)
    # symbol-day dedup on wider sets for reference
    pool_lpb = df[base_mask].sort_values("as_of")
    first_sd_nomodel = pool_lpb.groupby(["symbol", "day"], as_index=False).head(1)

    rows = [
        stats(evpos, months, "EV>0 raw (ref, inflated)"),
        stats(first_sd, months, "EV>0 first/sym-day"),
        stats(first_sd_nomodel, months, "no-ML long-P1-bull first/sym-day"),
    ]
    # H2: within symbol-day pick LEAST extended instead of first
    least_ext = evpos.sort_values("breakout_dist_atr").groupby(["symbol", "day"], as_index=False).head(1)
    rows.append(stats(least_ext, months, "EV>0 least-ext/sym-day"))
    # H3: heat gate on first/sym-day
    for k in (2, 3, 5, 8):
        sub = first_sd[first_sd["heat24"] >= k]
        rows.append(stats(sub, months, f"EV>0 first/sym-day heat>={k}"))
    # heat gate + day cap by earliest
    for k, cap in ((3, 3), (5, 3), (5, 5)):
        sub = (first_sd[first_sd["heat24"] >= k].sort_values("as_of")
               .groupby("day").head(cap))
        rows.append(stats(sub, months, f"heat>={k} first/sym-day cap{cap}"))
    # anti-H2 control: most extended per symbol-day
    most_ext = evpos.sort_values("breakout_dist_atr", ascending=False).groupby(
        ["symbol", "day"], as_index=False).head(1)
    rows.append(stats(most_ext, months, "EV>0 most-ext/sym-day (control)"))

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(out.to_string(index=False))

    # monthly for the most promising: heat>=3 first/sym-day cap3
    pick = (first_sd[first_sd["heat24"] >= 3].sort_values("as_of").groupby("day").head(3))
    print("\n=== heat>=3 first/sym-day cap3: monthly ===")
    g = pick.groupby("month").agg(n=("rm", "size"), R=("rm", "sum"), R_ret=("rr_", "sum"),
                                  tp=("outcome", lambda s: round((s == "tp").mean(), 2)))
    print(g.round(1).to_string())
    vc = pick["symbol"].value_counts()
    print(f"top symbols: {dict(vc.head(6))}, uniq syms: {len(vc)}")
    print(f"R without top-10 events: {pick['rm'].sum() - pick.nlargest(10, 'rm')['rm'].sum():+.1f}")


if __name__ == "__main__":
    main()
