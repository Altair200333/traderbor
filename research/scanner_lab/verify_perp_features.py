"""Verification harness for perp_features.py (scanner-v3 A2/A3).

Runs the three mandatory checks and prints a COMPACT summary:
  1. smoke-run stats (rows, date range, NaN fraction/feature, 3-row sample)
  2. truncation test (leakage protocol #1): hard-cut inputs at as_of, recompute,
     assert exact equality per feature at the latest computable point.
  3. variable-cadence check: SOL 2h/4h funding episode -> funding_cum_3d must sum
     MORE than 8h-cadence would and match a manual window sum.

Run: F:/projects/traderbor/.venv/Scripts/python.exe research/scanner_lab/verify_perp_features.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perp_features import (  # noqa: E402
    ALL_FEATURES, FUNDING_COLS, METRIC_COLS, MS_D, MS_H, perp_feature_frame,
)

REPO = HERE.parents[1]
PERP = REPO / "research" / "data" / "perp"
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _eq(a: float, b: float) -> bool:
    if (a is None or (isinstance(a, float) and np.isnan(a))) and \
       (b is None or (isinstance(b, float) and np.isnan(b))):
        return True
    if np.isnan(a) or np.isnan(b):
        return False
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))


def smoke() -> dict[str, pd.DataFrame]:
    print("=== 1. SMOKE ===")
    frames = {}
    for p in PAIRS:
        fr = perp_feature_frame(p, PERP)
        frames[p] = fr
        d0 = pd.Timestamp(int(fr["as_of"].min()), unit="ms").date()
        d1 = pd.Timestamp(int(fr["as_of"].max()), unit="ms").date()
        print(f"\n[{p}] rows={len(fr)}  as_of {d0}..{d1}")
        nan = fr[ALL_FEATURES].isna().mean().round(4)
        print("  nan_frac: " + "  ".join(f"{c}={nan[c]:.3f}" for c in ALL_FEATURES))
    # 3-row sample from SOL inside the metrics window (all features populated)
    sol = frames["SOLUSDT"]
    live = sol.dropna(subset=METRIC_COLS)
    samp = (live if len(live) else sol).tail(3).copy()
    samp["as_of_utc"] = pd.to_datetime(samp["as_of"], unit="ms", utc=True)
    cols = ["as_of_utc"] + ALL_FEATURES
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("\n[SOLUSDT 3-row sample, metrics-live]\n"
              + samp[cols].round(5).to_string(index=False))
    return frames


def truncation(frames: dict[str, pd.DataFrame]) -> None:
    print("\n=== 2. TRUNCATION (leakage #1) ===")
    rng = np.random.default_rng(7)
    sol_fund = pd.read_parquet(PERP / "funding" / "SOLUSDT.parquet")
    sol_met = pd.read_parquet(PERP / "metrics_5m" / "SOLUSDT.parquet")
    met_lo, met_hi = int(sol_met["ts_ms"].min()), int(sol_met["ts_ms"].max())

    # 2 cut points inside the metrics window (tests metric features too) +
    # 3 across the full funding range (funding-only, incl. fast-cadence region).
    ft = np.sort(sol_fund["fundingTime"].to_numpy("int64"))
    in_win = ft[(ft >= met_lo) & (ft <= met_hi)]
    full = ft[(ft > ft[400]) & (ft < ft[-5])]
    cuts = list(rng.choice(in_win, size=min(2, len(in_win)), replace=False)) + \
        list(rng.choice(full, size=3, replace=False))

    per_feat_pass = {c: 0 for c in ALL_FEATURES}
    per_feat_n = {c: 0 for c in ALL_FEATURES}
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        (tdir / "funding").mkdir(parents=True)
        (tdir / "metrics_5m").mkdir(parents=True)
        for k, cut in enumerate(cuts):
            cut = int(cut)
            sol_fund[sol_fund["fundingTime"] <= cut].to_parquet(
                tdir / "funding" / "SOLUSDT.parquet", index=False)
            sol_met[sol_met["ts_ms"] <= cut].to_parquet(
                tdir / "metrics_5m" / "SOLUSDT.parquet", index=False)
            ftr = perp_feature_frame("SOLUSDT", tdir)
            aprime = int(ftr["as_of"].max())          # latest computable point <= cut
            rt = ftr[ftr["as_of"] == aprime].iloc[0]
            rf = frames["SOLUSDT"]
            rf = rf[rf["as_of"] == aprime]
            if rf.empty:
                print(f"  cut#{k} @ {pd.Timestamp(cut, unit='ms')}: aprime not in full frame -> skip")
                continue
            rf = rf.iloc[0]
            fails = []
            for c in ALL_FEATURES:
                ok = _eq(float(rf[c]), float(rt[c]))
                per_feat_n[c] += 1
                per_feat_pass[c] += int(ok)
                if not ok:
                    fails.append(c)
            tag = "in-metrics-win" if met_lo <= cut <= met_hi else "funding-only"
            print(f"  cut#{k} {tag} cmp@{pd.Timestamp(aprime, unit='ms')}: "
                  f"{'PASS' if not fails else 'FAIL ' + ','.join(fails)}")
    print("  per-feature pass/total:")
    print("   " + "  ".join(f"{c}={per_feat_pass[c]}/{per_feat_n[c]}" for c in ALL_FEATURES))


def cadence() -> None:
    print("\n=== 3. VARIABLE CADENCE (SOL 2h/4h) ===")
    fund = pd.read_parquet(PERP / "funding" / "SOLUSDT.parquet")
    ft = np.sort(fund["fundingTime"].to_numpy("int64"))
    fr_map = dict(zip(fund["fundingTime"], fund["fundingRate"]))
    fr = np.array([fr_map[t] for t in ft], dtype="float64")
    gaps_h = np.diff(ft) / MS_H
    fast = np.where(gaps_h < 6.0)[0]  # sub-8h -> 2h/4h episode
    print(f"  settlements={len(ft)}  median_gap_h={np.median(gaps_h):.2f}  "
          f"min_gap_h={gaps_h.min():.2f}  sub-6h_gaps={len(fast)}")
    if len(fast) == 0:
        print("  no fast-cadence episode found -> cannot verify"); return
    # pick a point ~1.5d after the densest fast gap so the 3d window straddles it
    i = fast[np.argmin(gaps_h[fast])]
    t = int(ft[i + 1] + int(1.5 * MS_D))
    frame = perp_feature_frame("SOLUSDT", PERP)
    row = frame[frame["as_of"] <= t].iloc[-1]
    a = int(row["as_of"])
    # replicate module anchoring: last settlement s <= a, window (s-3d, s]
    s_idx = np.searchsorted(ft, a, side="right") - 1
    s = int(ft[s_idx])
    win = (ft > s - 3 * MS_D) & (ft <= s)
    n_win = int(win.sum())
    manual_cum = float(fr[win].sum())
    val = float(row["funding_cum_3d"])
    print(f"  fast-gap idx={i} gap={gaps_h[i]:.2f}h  as_of={pd.Timestamp(a, unit='ms')}")
    print(f"  settlements in trailing-3d window={n_win} (pure-8h would be 9)  "
          f"-> {'MORE (variable OK)' if n_win > 9 else 'NOT more'}")
    print(f"  funding_cum_3d frame={val:.6g}  manual_window_sum={manual_cum:.6g}  "
          f"match={_eq(val, manual_cum)}")


if __name__ == "__main__":
    frames = smoke()
    truncation(frames)
    cadence()
    print("\nDONE")
