"""Download ~2 years of Binance spot klines for the trading universe.

Source: https://data.binance.vision (public S3 dumps, no rate limits).
- monthly zips first; tail months that 404 fall back to daily zips
- head-month 404s = pre-listing, silently skipped
- timestamps normalized to ms (2025+ spot files use microseconds)
Output: research/data/klines/{interval}/{PAIR}.parquet + manifest.json

Usage:
  python download_klines.py --intervals 1h,5m --start 2024-07 --end 2026-06 \
      --daily-from 2026-06 --daily-to 2026-07-06
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402

BASE = "https://data.binance.vision/data/spot"
OUT_DIR = REPO_ROOT / "research" / "data" / "klines"
COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KEEP = ["open_time", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base"]

_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "traderbor-research/1.0"
        _tls.s = s
    return _tls.s


def _get(url: str, retries: int = 4) -> bytes | None:
    """Fetch url; None on 404; retry with backoff on other errors."""
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


def _parse_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    first_line = raw[:200].split(b"\n", 1)[0]
    header = 0 if first_line.startswith(b"open_time") else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=COLS)
    # 2025+ spot dumps use microsecond timestamps; normalize to ms
    if len(df) and df["open_time"].iloc[0] > 1e14:
        df["open_time"] = df["open_time"] // 1000
    return df[KEEP]


def _months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _days(start: date, end: date) -> list[str]:
    out, d = [], start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def download_pair(pair: str, interval: str, months: list[str],
                  daily_from: str, daily_days: list[str]) -> dict:
    frames: list[pd.DataFrame] = []
    monthly_miss: list[str] = []
    seen_any = False
    for ym in months:
        url = f"{BASE}/monthly/klines/{pair}/{interval}/{pair}-{interval}-{ym}.zip"
        blob = _get(url)
        if blob is None:
            # tail months may not be published yet -> daily fallback; head 404 = pre-listing
            if seen_any and ym >= daily_from:
                monthly_miss.append(ym)
            continue
        seen_any = True
        frames.append(_parse_zip(blob))
    # daily fallback for unpublished tail months + explicit daily range
    daily_targets = [d for d in daily_days
                     if d[:7] in monthly_miss or d[:7] > months[-1]]
    for day in daily_targets:
        url = f"{BASE}/daily/klines/{pair}/{interval}/{pair}-{interval}-{day}.zip"
        blob = _get(url)
        if blob is not None:
            frames.append(_parse_zip(blob))
    if not frames:
        return {"pair": pair, "interval": interval, "rows": 0, "status": "no_data"}
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df["open_time"] = df["open_time"].astype("int64")
    for c in KEEP[1:]:
        df[c] = df[c].astype("float64")
    out = OUT_DIR / interval / f"{pair}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    step = {"1h": 3_600_000, "5m": 300_000, "15m": 900_000, "1m": 60_000, "1d": 86_400_000}[interval]
    expected = (df["open_time"].iloc[-1] - df["open_time"].iloc[0]) // step + 1
    return {
        "pair": pair, "interval": interval, "rows": int(len(df)),
        "first": pd.Timestamp(df["open_time"].iloc[0], unit="ms").isoformat(),
        "last": pd.Timestamp(df["open_time"].iloc[-1], unit="ms").isoformat(),
        "missing_bars": int(expected - len(df)),
        "daily_fallback_months": monthly_miss,
        "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", default="1h")
    ap.add_argument("--start", default="2024-07")
    ap.add_argument("--end", default="2026-06")
    ap.add_argument("--daily-from", default="2026-05",
                    help="months >= this may use daily fallback if monthly missing")
    ap.add_argument("--daily-to", default="2026-07-06")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pairs", default="", help="comma list override (default: universe)")
    args = ap.parse_args()

    pairs = (args.pairs.split(",") if args.pairs
             else [c.pair for c in load_universe()])
    months = _months(args.start, args.end)
    last_month_end = date.fromisoformat(args.daily_to)
    daily_days = _days(date.fromisoformat(args.daily_from + "-01"), last_month_end)
    intervals = args.intervals.split(",")

    tasks = [(p, iv) for iv in intervals for p in pairs]
    print(f"{len(pairs)} pairs x {intervals} = {len(tasks)} tasks, months {months[0]}..{months[-1]}")
    results, errors = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_pair, p, iv, months, args.daily_from, daily_days): (p, iv)
                for p, iv in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            p, iv = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                if i % 25 == 0 or res["status"] != "ok":
                    print(f"[{i}/{len(tasks)}] {p} {iv}: {res['status']} rows={res.get('rows', 0)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append({"pair": p, "interval": iv, "error": repr(e)})
                print(f"[{i}/{len(tasks)}] {p} {iv}: ERROR {e!r}", flush=True)

    manifest_path = OUT_DIR / "manifest.json"
    existing = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
    for r in results:
        existing[f"{r['pair']}:{r['interval']}"] = r
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(existing, indent=1))

    ok = [r for r in results if r["status"] == "ok"]
    nodata = [r for r in results if r["status"] == "no_data"]
    print(f"\nDONE in {time.time() - t0:.0f}s: ok={len(ok)} no_data={[r['pair'] for r in nodata]} "
          f"errors={len(errors)}")
    print(f"total rows: {sum(r['rows'] for r in ok):,}")
    worst_gaps = sorted(ok, key=lambda r: -r["missing_bars"])[:8]
    print("worst gaps:", [(r["pair"], r["interval"], r["missing_bars"]) for r in worst_gaps])
    if errors:
        print("ERRORS:", errors[:10])


if __name__ == "__main__":
    main()
