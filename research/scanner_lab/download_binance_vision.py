"""Bulk download Binance UM-futures 1m klines from data.binance.vision.

Purpose: high-resolution price paths for the execution-aware simulator
(stop/TTL/fill paths at 1m resolution instead of 1h). These are FUTURES
prices (the instrument we actually trade), unlike research/data/*/klines
which are spot.

Output: research/data/binance_um/klines_1m/{PAIR}.parquet
  columns: open_time (ms), open, high, low, close, volume, quote_volume,
           count, taker_buy_volume
Coverage: monthly zips 2020-01..last-full-month + daily zips for the
current-month tail. 404 (pair not listed yet / no futures) skipped.
Resumable: existing parquet -> only months after its last timestamp are
fetched. Timestamps normalized to ms (some vision datasets switched to
microseconds in 2025).

Usage: python download_binance_vision.py [--workers 8] [--pairs A,B,...]
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402

BASE = "https://data.binance.vision/data/futures/um"
OUT_DIR = REPO_ROOT / "research" / "data" / "binance_um" / "klines_1m"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
        "ignore"]
KEEP = ["open_time", "open", "high", "low", "close", "volume", "quote_volume",
        "count", "taker_buy_volume"]


def month_list(start: str = "2020-01") -> list[str]:
    """All complete months from start to the month before the current one."""
    now = datetime.now(timezone.utc)
    y, m = int(start[:4]), int(start[5:7])
    out = []
    while (y, m) < (now.year, now.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def day_list() -> list[str]:
    """Days of the current month up to yesterday (daily zips lag ~1 day)."""
    now = datetime.now(timezone.utc)
    out, d = [], now.replace(day=1)
    while d.date() < now.date():
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def fetch_csv(session: requests.Session, url: str) -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            raw = zf.read(zf.namelist()[0])
            header = 0 if raw[:9] == b"open_time" else None
            df = pd.read_csv(io.BytesIO(raw), header=header, names=COLS)
            return df
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def build_pair(pair: str, months: list[str], days: list[str]) -> str:
    out_path = OUT_DIR / f"{pair}.parquet"
    last_ts = -1
    old = None
    if out_path.exists():
        old = pd.read_parquet(out_path)
        if len(old):
            last_ts = int(old["open_time"].max())

    session = requests.Session()
    parts = []
    for ym in months:
        y, m = int(ym[:4]), int(ym[5:7])
        nxt = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
        if nxt.timestamp() * 1000 <= last_ts:
            continue  # month fully covered already
        df = fetch_csv(session, f"{BASE}/monthly/klines/{pair}/1m/{pair}-1m-{ym}.zip")
        if df is not None:
            parts.append(df)
    for ymd in days:
        d0 = datetime.strptime(ymd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if (d0 + timedelta(days=1)).timestamp() * 1000 <= last_ts:
            continue
        df = fetch_csv(session, f"{BASE}/daily/klines/{pair}/1m/{pair}-1m-{ymd}.zip")
        if df is not None:
            parts.append(df)

    if not parts:
        return f"{pair}: no new data (existing rows {0 if old is None else len(old)})"
    new = pd.concat(parts, ignore_index=True)[KEEP]
    # normalize microsecond timestamps (vision format change 2025) to ms
    big = new["open_time"] > 1e14
    if big.any():
        new.loc[big, "open_time"] = new.loc[big, "open_time"] // 1000
    if old is not None:
        new = pd.concat([old, new], ignore_index=True)
    new = (new.drop_duplicates("open_time").sort_values("open_time")
           .reset_index(drop=True))
    new.to_parquet(out_path, index=False)
    mb = out_path.stat().st_size / 1e6
    t0 = pd.to_datetime(int(new["open_time"].iloc[0]), unit="ms", utc=True)
    t1 = pd.to_datetime(int(new["open_time"].iloc[-1]), unit="ms", utc=True)
    return (f"{pair}: {len(new):,} rows {t0:%Y-%m-%d}..{t1:%Y-%m-%d} "
            f"({mb:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pairs", type=str, default="")
    args = ap.parse_args()

    pairs = (args.pairs.split(",") if args.pairs
             else [c.pair for c in load_universe()])
    months, days = month_list(), day_list()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"pairs={len(pairs)} months={months[0]}..{months[-1]} "
          f"days={days[0]}..{days[-1]} -> {OUT_DIR}", flush=True)

    done, errors = 0, 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_pair, p, months, days): p for p in pairs}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                msg = f"{futs[fut]}: ERROR {e}"
            print(f"[{done}/{len(pairs)}] {msg}", flush=True)

    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.parquet")) / 1e6
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min, errors={errors}, "
          f"total {total_mb / 1000:.1f} GB in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
