"""Resample 1h parquet -> 4h/1d candles and upsert into a LocalMarketCache DB.
Needed because agent digest/analysis tools query 4h/1d levels.

Usage: python resample_export.py --out <cache.sqlite3> [--intervals 4h,1d]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "agents-v2"))
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache  # noqa: E402

KL1H = REPO_ROOT / "research" / "data" / "klines" / "1h"
SPEC = {"4h": 4 * 3_600_000, "1d": 24 * 3_600_000}


def resample(df: pd.DataFrame, step: int) -> pd.DataFrame:
    g = df["open_time"] // step * step
    out = df.groupby(g).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
        taker_buy_base=("taker_buy_base", "sum"), n=("open", "size"),
    ).reset_index().rename(columns={"open_time": "bucket"})
    # drop incomplete buckets (head/tail of history)
    return out[out["n"] == step // 3_600_000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--intervals", default="4h,1d")
    args = ap.parse_args()
    cache = LocalMarketCache(path=Path(args.out))
    t0 = time.time()
    total = 0
    for interval in args.intervals.split(","):
        step = SPEC[interval]
        for c in load_universe():
            src = KL1H / f"{c.pair}.parquet"
            if not src.exists():
                continue
            df = pd.read_parquet(src)
            r = resample(df, step)
            candles = [
                Candle(symbol=c.pair, interval=interval,
                       open_time=int(row.bucket), close_time=int(row.bucket) + step - 1,
                       open=float(row.open), high=float(row.high), low=float(row.low),
                       close=float(row.close), volume=float(row.volume),
                       quote_volume=float(row.quote_volume), trade_count=int(row.trades),
                       taker_buy_base_volume=float(row.taker_buy_base))
                for row in r.itertuples()
            ]
            total += cache.upsert_candles(candles)
    print(f"DONE {total:,} rows in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
