"""Build the full event dataset: scan -> label -> features -> parquet.

Usage: python dataset.py [--out events.parquet]
Output: research/data/events.parquet + printed summary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import scan_symbol  # noqa: E402
from features import build_market_frames, extra_symbol_features, join_market  # noqa: E402
from indicators_vec import roc  # noqa: E402
from labeling import label_symbol  # noqa: E402
from universe import REPO_ROOT, load_universe  # noqa: E402

KL = REPO_ROOT / "research" / "data" / "klines"
OUT_DEFAULT = REPO_ROOT / "research" / "data" / "events.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--pairs", default="", help="comma list override for debugging")
    args = ap.parse_args()

    coins = load_universe()
    if args.pairs:
        keep = set(args.pairs.split(","))
        coins = [c for c in coins if c.pair in keep]

    t0 = time.time()
    frames_1h: dict[str, pd.DataFrame] = {}
    for c in coins:
        p = KL / "1h" / f"{c.pair}.parquet"
        if p.exists():
            frames_1h[c.pair] = pd.read_parquet(p)
    print(f"loaded {len(frames_1h)} 1h frames in {time.time()-t0:.0f}s", flush=True)

    btc = frames_1h["BTCUSDT"]
    btc_roc4h = pd.Series(roc(btc["close"], 4).to_numpy(), index=btc["open_time"].to_numpy())

    all_events = []
    for i, c in enumerate(coins, 1):
        if c.pair not in frames_1h:
            continue
        df1h = frames_1h[c.pair]
        if len(df1h) < 400:
            continue
        ev = scan_symbol(df1h, c.pair, btc_roc4h)
        if ev.empty:
            continue
        p5 = KL / "5m" / f"{c.pair}.parquet"
        if not p5.exists():
            continue
        df5m = pd.read_parquet(p5)
        ev = label_symbol(ev, df5m)
        ev = extra_symbol_features(df1h, ev)
        all_events.append(ev)
        if i % 25 == 0:
            print(f"[{i}/{len(coins)}] events so far: {sum(len(e) for e in all_events)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ev = pd.concat(all_events, ignore_index=True)
    print(f"scan+label done: {len(ev)} events in {time.time()-t0:.0f}s", flush=True)

    market, ranks = build_market_frames(frames_1h)
    meta = pd.DataFrame([{
        "symbol": c.pair,
        "tier": {"T1": 1, "T2": 2, "T3": 3}[c.tier],
        "uni_adr_pct": c.adr_pct,
        "uni_log_spot30": np.log10(max(c.spot30_musd, 0.05)),
    } for c in coins])
    ev = join_market(ev, market, ranks, meta)

    out = Path(args.out)
    ev.to_parquet(out, index=False)
    print(f"wrote {out} ({len(ev)} rows, {len(ev.columns)} cols) in {time.time()-t0:.0f}s")

    # summary
    res = ev[ev["outcome"].isin(["tp", "sl"])]
    print("\n=== outcome counts ===")
    print(ev["outcome"].value_counts().to_string())
    print(f"\nresolved tp rate: {(res['outcome'] == 'tp').mean():.3f} (n={len(res)})")
    print(f"ambiguous share of barrier-touch: "
          f"{(ev['outcome'] == 'ambiguous').sum() / max((ev['outcome'].isin(['tp','sl','ambiguous'])).sum(), 1):.3f}")
    print("\n=== by quality (production parity) ===")
    for name, mask in [("hard", ev["is_hard"]), ("marginal", ev["is_marginal"]),
                       ("pool_rest", ~ev["is_hard"] & ~ev["is_marginal"])]:
        sub = ev[mask & ev["outcome"].isin(["tp", "sl"])]
        n_all = int(mask.sum())
        if len(sub):
            print(f"{name}: n={n_all} resolved={len(sub)} tp_rate={(sub['outcome'] == 'tp').mean():.3f} "
                  f"avg_r_mkt={ev[mask]['r_market'].mean():+.3f} avg_r_retest={ev[mask]['r_retest'].mean():+.3f}")
        else:
            print(f"{name}: n={n_all} resolved=0")
    print("\n=== top symbols by event count ===")
    print(ev["symbol"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
