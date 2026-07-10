"""Append recent Binance spot klines to the existing parquet cache via REST.

download_klines.py rebuilds each pair file from zip dumps and the daily zips
lag ~1-2 days; this script extends existing parquet from the last stored bar
to now with GET /api/v3/klines (1000 bars/call). The in-progress candle is
dropped so the cache only ever contains closed bars. Pairs without an existing
parquet are skipped (zip downloader owns initial history).

Usage:
  python refresh_klines_rest.py --intervals 1h,5m [--pairs BTCUSDT,ETHUSDT]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_klines import KEEP, OUT_DIR, _get  # noqa: E402
from universe import load_universe  # noqa: E402

API = "https://api.binance.com/api/v3/klines"
STEP = {"1h": 3_600_000, "5m": 300_000, "15m": 900_000, "1m": 60_000}
# REST kline array indices for the KEEP columns
IDX = {"open_time": 0, "open": 1, "high": 2, "low": 3, "close": 4,
       "volume": 5, "quote_volume": 7, "trades": 8, "taker_buy_base": 9}


def fetch_tail(pair: str, interval: str, start_ms: int, now_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cur = start_ms
    while cur < now_ms:
        blob = _get(f"{API}?symbol={pair}&interval={interval}"
                    f"&startTime={cur}&limit=1000")
        if blob is None:  # 404 = symbol unknown to REST (delisted)
            break
        batch = json.loads(blob)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cur = batch[-1][0] + STEP[interval]
    if not rows:
        return pd.DataFrame(columns=KEEP)
    df = pd.DataFrame({c: [r[i] for r in rows] for c, i in IDX.items()})[KEEP]
    df = df[df["open_time"] + STEP[interval] <= now_ms]  # closed bars only
    return df


def refresh_pair(pair: str, interval: str, now_ms: int) -> dict:
    path = OUT_DIR / interval / f"{pair}.parquet"
    if not path.exists():
        return {"pair": pair, "interval": interval, "status": "no_parquet", "added": 0}
    df = pd.read_parquet(path)
    last = int(df["open_time"].max())
    tail = fetch_tail(pair, interval, last + STEP[interval], now_ms)
    if tail.empty:
        return {"pair": pair, "interval": interval, "status": "up_to_date", "added": 0,
                "last": pd.Timestamp(last, unit="ms").isoformat()}
    tail["open_time"] = tail["open_time"].astype("int64")
    for c in KEEP[1:]:
        tail[c] = tail[c].astype("float64")
    out = (pd.concat([df, tail], ignore_index=True)
           .drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True))
    out.to_parquet(path, index=False)
    return {"pair": pair, "interval": interval, "status": "ok", "added": int(len(tail)),
            "last": pd.Timestamp(int(out["open_time"].max()), unit="ms").isoformat()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", default="1h,5m")
    ap.add_argument("--pairs", default="", help="comma list override (default: universe)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pairs = (args.pairs.split(",") if args.pairs
             else [c.pair for c in load_universe()])
    intervals = args.intervals.split(",")
    now_ms = int(time.time() * 1000)
    tasks = [(p, iv) for iv in intervals for p in pairs]
    print(f"refresh {len(pairs)} pairs x {intervals} via REST", flush=True)

    results, errors = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(refresh_pair, p, iv, now_ms): (p, iv) for p, iv in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            p, iv = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                if i % 50 == 0 or res["status"] not in ("ok", "up_to_date"):
                    print(f"[{i}/{len(tasks)}] {p} {iv}: {res['status']} "
                          f"+{res['added']} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append({"pair": p, "interval": iv, "error": repr(e)})
                print(f"[{i}/{len(tasks)}] {p} {iv}: ERROR {e!r}", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r["pair"] for r in results if r["status"] == "no_parquet"]
    print(f"\nDONE in {time.time() - t0:.0f}s: extended={len(ok)} "
          f"rows_added={sum(r['added'] for r in ok):,} no_parquet={skipped} "
          f"errors={len(errors)}")
    if ok:
        print("max last bar:", max(r["last"] for r in ok))
    if errors:
        print("ERRORS:", errors[:10])


if __name__ == "__main__":
    main()
