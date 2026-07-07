"""Model-free evaluation: production gate stack on the wide universe with
exit upgrades. No ML anywhere — pure label statistics on events.parquet.

1. Slice table: avg r by side/pattern/regime on the RAW pool (model-free).
2. Prod hard+marg stream (sym-day dedup) under slot sim with exit grid
   (stop mult x rr x hold recomputed on 5m paths).

Usage: python rules_eval.py [--cost-bps 25]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exit_sweep import outcomes_for_cell, slot_r  # noqa: E402
from universe import REPO_ROOT  # noqa: E402

EVENTS = REPO_ROOT / "research" / "data" / "events.parquet"
KL5 = REPO_ROOT / "research" / "data" / "klines" / "5m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--start", default="2025-04")   # match walk-forward window
    args = ap.parse_args()
    cost = args.cost_bps / 1e4

    ev = pd.read_parquet(EVENTS)
    ev = ev[ev["stop_feasible"] & ev["outcome"].isin(["tp", "sl", "none", "ambiguous"])].copy()
    t0 = int(pd.Timestamp(args.start + "-01", tz="UTC").timestamp() * 1000)
    ev = ev[ev["as_of"] >= t0]
    ev["rm"] = np.where(ev["outcome"] == "ambiguous", -1.0, ev["r_market"]) - cost / ev["d_final"]
    ev["month"] = pd.to_datetime(ev["as_of"], unit="ms", utc=True).dt.strftime("%Y-%m")
    ev["day"] = pd.to_datetime(ev["as_of"], unit="ms", utc=True).dt.date

    print(f"window {args.start}.. n={len(ev)} (model-free)\n")
    print("=== raw pool avg rm by side x pattern (per event, 25bps) ===")
    g = ev.groupby(["side", "pattern"]).agg(n=("rm", "size"), avg=("rm", "mean"))
    print(g.round(3).to_string())
    print("\n=== long pool by regime ===")
    lp = ev[ev["side"] == "long"]
    g2 = lp.groupby(lp["btc_above_ema50"] == 1).agg(n=("rm", "size"), avg=("rm", "mean"))
    print(g2.round(3).to_string())
    print("\n=== prod quality x pattern (long) ===")
    q = np.select([lp["is_hard"], lp["is_marginal"]], ["hard", "marginal"], "pool")
    g3 = lp.groupby([q, "pattern"]).agg(n=("rm", "size"), avg=("rm", "mean"),
                                        tp=("outcome", lambda s: (s == "tp").mean()))
    print(g3.round(3).to_string())

    # prod stream: hard+marginal (long only), sym-day dedup, chronological
    stream = (ev[(ev["is_hard"] | ev["is_marginal"]) & (ev["side"] == "long")]
              .sort_values("as_of").groupby(["symbol", "day"], as_index=False).head(1)
              .sort_values("as_of").reset_index(drop=True))
    print(f"\nprod long stream: {len(stream)} events, "
          f"{len(stream) / stream['month'].nunique():.0f}/mo")

    results: dict = {}
    for sym, sub in stream.groupby("symbol"):
        d5 = pd.read_parquet(KL5 / f"{sym}.parquet",
                             columns=["open_time", "high", "low", "close"])
        t5 = d5["open_time"].to_numpy(np.int64)
        h5, l5, c5 = d5["high"].to_numpy(), d5["low"].to_numpy(), d5["close"].to_numpy()
        for row in sub.itertuples():
            j0 = np.searchsorted(t5, row.as_of, "left")
            j2 = np.searchsorted(t5, row.as_of + 49 * 3_600_000, "left")
            pt, ph, pl, pc = t5[j0:j2], h5[j0:j2], l5[j0:j2], c5[j0:j2]
            for sm in (1.0, 1.25, 1.5):
                for rr in (2.0, 2.5, 3.0, 4.0):
                    r, te = outcomes_for_cell(pt, ph, pl, pc, row.as_of,
                                              row.entry, row.d_final * sm, rr, 24)
                    if np.isfinite(r):
                        r -= cost / (row.d_final * sm)
                    results.setdefault((sm, rr), []).append((row.Index, r, te))

    as_of_a = stream["as_of"].to_numpy(np.int64)
    syms_a = stream["symbol"].to_numpy()
    mon_a = stream["month"].to_numpy()
    rows = []
    for (sm, rr), vals in sorted(results.items()):
        idx, rs, tes = zip(*vals)
        rs_a = np.full(len(stream), np.nan)
        te_a = np.full(len(stream), np.nan)
        rs_a[list(idx)] = rs
        te_a[list(idx)] = tes
        tot, ntk = slot_r(as_of_a, syms_a, rs_a, te_a, slots=3)
        # monthly stability at no-capacity for CI feel
        dfm = pd.DataFrame({"m": mon_a, "r": rs_a}).dropna()
        ms = dfm.groupby("m")["r"].sum()
        rows.append({"stop_mult": sm, "rr": rr,
                     "avg_r": round(np.nanmean(rs_a), 3),
                     "win": round(np.nanmean(rs_a > 0), 3),
                     "slot3_R": round(tot, 1), "slot3_n": ntk,
                     "mo_pos": f"{(ms > 0).sum()}/{len(ms)}",
                     "worst_mo": round(ms.min(), 1)})
    out = pd.DataFrame(rows).sort_values("slot3_R", ascending=False)
    print("\n=== prod long stream exit grid (slot3) ===")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
