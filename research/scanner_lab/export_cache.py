"""Export research parquet klines into a LocalMarketCache SQLite DB.

Lets the simulator/replay (and legacy screener) run on the wide 149-coin,
2-year dataset without re-fetching from Binance.

Usage:
  python export_cache.py --out ../../agents-v2/data/wide-2y/market_cache.sqlite3 \
      --intervals 1h,5m [--pairs BTCUSDT,ETHUSDT] [--start 2026-01-01] [--end 2026-07-06]
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

KL = REPO_ROOT / "research" / "data" / "klines"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


def export(out_path: Path, intervals: list[str], pairs: list[str],
           start_ms: int | None, end_ms: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = LocalMarketCache(path=out_path)
    t0 = time.time()
    total = 0
    for interval in intervals:
        step = INTERVAL_MS[interval]
        for i, pair in enumerate(pairs, 1):
            src = KL / interval / f"{pair}.parquet"
            if not src.exists():
                print(f"skip {pair} {interval}: no parquet")
                continue
            df = pd.read_parquet(src)
            if start_ms is not None:
                df = df[df["open_time"] >= start_ms]
            if end_ms is not None:
                df = df[df["open_time"] < end_ms]
            candles = [
                Candle(
                    symbol=pair, interval=interval,
                    open_time=int(r.open_time), close_time=int(r.open_time) + step - 1,
                    open=float(r.open), high=float(r.high), low=float(r.low),
                    close=float(r.close), volume=float(r.volume),
                    quote_volume=float(r.quote_volume),
                    trade_count=int(r.trades),
                    taker_buy_base_volume=float(r.taker_buy_base),
                )
                for r in df.itertuples()
            ]
            total += cache.upsert_candles(candles)
            if i % 25 == 0:
                print(f"[{interval} {i}/{len(pairs)}] rows so far {total:,} ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"DONE: {total:,} rows -> {out_path} in {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--intervals", default="1h")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    args = ap.parse_args()
    pairs = args.pairs.split(",") if args.pairs else [c.pair for c in load_universe()]
    to_ms = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000) if s else None  # noqa: E731
    export(Path(args.out), args.intervals.split(","), pairs, to_ms(args.start), to_ms(args.end))


if __name__ == "__main__":
    main()
