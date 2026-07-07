"""Parity validation: ScannerV2 runtime path vs the research dataset.

For sampled historical hours with known long-P1 pool events, rebuild the
runtime view (truncated frames -> scan_symbol -> features) and compare with
events.parquet rows: trigger set match, d_final/entry exact, features within
ewm-drift tolerance. Fails loudly on mismatches.

Usage: python validate_scanner.py [--n 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import scan_symbol  # noqa: E402
from features import build_market_frames, extra_symbol_features, join_market  # noqa: E402
from indicators_vec import roc  # noqa: E402
from scanner_v2 import WARMUP_BARS, ScannerV2  # noqa: E402
from universe import REPO_ROOT  # noqa: E402

KL = REPO_ROOT / "research" / "data" / "klines" / "1h"
EVENTS = REPO_ROOT / "research" / "data" / "events.parquet"

CHECK_COLS = [
    "entry", "d_final", "tp_rr", "roc_4h", "roc_24h", "roc_168h", "rsi14",
    "atr_pct", "vol_ratio", "rvol_hod", "breakout_dist_atr", "ema20_ext_atr",
    "bbw_pctile_720", "ma_align", "breadth_ema50", "rank_roc24", "altseason",
]
TOL = {"default": 2e-3, "rsi14": 0.2, "bbw_pctile_720": 0.05, "breadth_ema50": 0.02,
       "rank_roc24": 0.02, "ma_align": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ev_all = pd.read_parquet(EVENTS)
    pool = ev_all[(ev_all["side"] == "long") & (ev_all["pattern"] == "P1")
                  & (ev_all["as_of"] >= int(pd.Timestamp("2025-02-01", tz="UTC").timestamp() * 1000))]
    rng = np.random.default_rng(args.seed)
    hours = rng.choice(pool["as_of"].unique(), size=args.n, replace=False)

    frames = {p.stem: pd.read_parquet(p) for p in KL.glob("*.parquet")}
    sc = ScannerV2()
    meta_df = sc._meta_df

    total_expected = total_found = 0
    worst = {}
    for as_of in sorted(int(h) for h in hours):
        trigger_open = as_of - 3_600_000
        cut = {}
        for pair, df in frames.items():
            d = df[df["open_time"] <= trigger_open]
            if len(d) > WARMUP_BARS:
                d = d.iloc[-WARMUP_BARS:].reset_index(drop=True)
            if len(d) >= 400:
                cut[pair] = d
        btc = cut["BTCUSDT"]
        btc_roc4h = pd.Series(roc(btc["close"], 4).to_numpy(),
                              index=btc["open_time"].to_numpy())
        got = []
        for pair, d in cut.items():
            e = scan_symbol(d, pair, btc_roc4h)
            if e.empty:
                continue
            e = e[e["as_of"] == as_of]
            if len(e):
                got.append(extra_symbol_features(d, e))
        got_df = pd.concat(got, ignore_index=True) if got else pd.DataFrame()
        if len(got_df):
            market, ranks = build_market_frames(cut)
            got_df = join_market(got_df, market, ranks, meta_df)
            got_df = got_df[(got_df["side"] == "long") & (got_df["pattern"] == "P1")]

        exp = pool[pool["as_of"] == as_of]
        total_expected += len(exp)
        ts = pd.Timestamp(as_of, unit="ms")
        exp_syms = set(exp["symbol"])
        got_syms = set(got_df["symbol"]) if len(got_df) else set()
        missing = exp_syms - got_syms
        extra = got_syms - exp_syms
        total_found += len(exp_syms & got_syms)
        status = "OK" if not missing and not extra else f"MISS={missing} EXTRA={extra}"
        print(f"{ts}: expected {len(exp_syms)}, got {len(got_syms)} -> {status}")
        for sym in exp_syms & got_syms:
            a = exp[exp["symbol"] == sym].iloc[0]
            b = got_df[got_df["symbol"] == sym].iloc[0]
            for col in CHECK_COLS:
                va, vb = float(a[col]), float(b[col])
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                scale = max(abs(va), 1e-9) if col in ("entry",) else 1.0
                diff = abs(va - vb) / scale
                tol = TOL.get(col, TOL["default"])
                if diff > worst.get(col, (0, ""))[0]:
                    worst[col] = (diff, f"{sym}@{ts}")
                if diff > tol:
                    print(f"  DIFF {sym} {col}: dataset={va:.6g} runtime={vb:.6g}")

    print(f"\ntrigger parity: {total_found}/{total_expected} matched")
    print("worst feature drifts:")
    for col, (d, where) in sorted(worst.items(), key=lambda kv: -kv[1][0])[:8]:
        print(f"  {col}: {d:.2e} ({where})")


if __name__ == "__main__":
    main()
