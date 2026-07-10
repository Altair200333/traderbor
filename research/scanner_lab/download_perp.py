"""Download Binance USD-M perp data for the trading universe.

Sources:
- funding: fapi REST /fapi/v1/fundingRate, paginated forward by fundingTime+1
  (markPrice is empty string in historical rows -> NaN; unknown symbol -> empty
  list or HTTP 400 -> skipped silently, recorded in manifest)
- metrics: https://data.binance.vision daily zips, one CSV per day of 5m rows;
  missing day 404s are skipped; resumable (re-fetches from max stored date on)

1000-prefix alias: Binance lists some perps under a "1000X" contract
(1000BONKUSDT, 1000FLOKIUSDT, 1000PEPEUSDT, 1000SHIBUSDT, ...). funding rate is
dimensionless so it's usable as-is; when a plain symbol returns no funding rows
(or no metrics zips) we retry as "1000" + pair and store the result under the
original pair name, recording "alias": "1000<PAIR>" in the manifest entry.

Output: research/data/perp/funding/{PAIR}.parquet,
        research/data/perp/metrics_5m/{PAIR}.parquet + manifest.json

Usage:
  python download_perp.py --sources funding,metrics --start 2022-07-01 --end 2026-07-06
  python download_perp.py --pairs BONKUSDT,PEPEUSDT --sources funding --no-manifest
  python download_perp.py --pairs BTCUSDT,ETHUSDT,SOLUSDT --sources metrics --force-refresh
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402

FAPI_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
VISION_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
OUT_DIR = REPO_ROOT / "research" / "data" / "perp"
FUNDING_PAGE_SLEEP = 0.1  # per-worker pause between fapi pages
METRICS_COLS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]
METRICS_KEEP = ["ts_ms"] + METRICS_COLS[2:]

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


def _fapi_get(params: dict, retries: int = 5) -> list | None:
    """Fetch one fundingRate page; None on 400/404 (unknown symbol); 429 -> 30s backoff."""
    for attempt in range(retries):
        try:
            r = _session().get(FAPI_URL, params=params, timeout=30)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code in (400, 404):
            return None
        if r.status_code in (418, 429):
            time.sleep(30)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"fapi gave up after {retries} tries: {params}")


def _days(start: date, end: date) -> list[str]:
    out, d = [], start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _day_ms(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_funding_rows(symbol: str, start_ms: int, end_ms: int) -> list[dict] | None:
    """Page fundingRate for symbol. None = unknown symbol (400/404); [] = known, no rows."""
    rows: list[dict] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = _fapi_get({"symbol": symbol, "startTime": cursor,
                           "endTime": end_ms, "limit": 1000})
        if batch is None:  # 400/404 = not a futures symbol
            return None
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = int(batch[-1]["fundingTime"]) + 1
        time.sleep(FUNDING_PAGE_SLEEP)
    return rows


def download_funding(pair: str, start_ms: int, end_ms: int) -> dict:
    rows = _fetch_funding_rows(pair, start_ms, end_ms)
    unknown = rows is None
    alias: str | None = None
    if not rows:  # None (unknown symbol) or [] (empty range) -> try the 1000-prefixed contract
        alias_symbol = "1000" + pair
        alias_rows = _fetch_funding_rows(alias_symbol, start_ms, end_ms)
        if alias_rows:
            rows, alias = alias_rows, alias_symbol
        else:
            unknown = unknown and alias_rows is None
    if not rows:
        return {"pair": pair, "source": "funding", "rows": 0,
                "status": "no_symbol" if unknown else "no_data"}
    df = pd.DataFrame(rows)
    df["fundingTime"] = df["fundingTime"].astype("int64")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce").astype("float64")
    df["markPrice"] = pd.to_numeric(df.get("markPrice"), errors="coerce").astype("float64")
    df = (df[["fundingTime", "fundingRate", "markPrice"]]
          .drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True))
    out = OUT_DIR / "funding" / f"{pair}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    gaps_h = df["fundingTime"].diff().dropna() / 3_600_000
    result = {
        "pair": pair, "source": "funding", "rows": int(len(df)),
        "first": pd.Timestamp(df["fundingTime"].iloc[0], unit="ms").isoformat(),
        "last": pd.Timestamp(df["fundingTime"].iloc[-1], unit="ms").isoformat(),
        "median_gap_hours": round(float(gaps_h.median()), 3) if len(gaps_h) else None,
        "status": "ok",
    }
    if alias:
        result["alias"] = alias
    return result


def _parse_metrics_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0])
    header = 0 if raw[:200].split(b"\n", 1)[0].startswith(b"create_time") else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=METRICS_COLS)
    # force ns resolution: newer pandas parses second-precision strings as datetime64[s]
    ts = pd.to_datetime(df["create_time"]).astype("datetime64[ns]")
    df["ts_ms"] = ts.astype("int64") // 1_000_000
    return df[METRICS_KEEP]


def _fetch_metrics_frames(symbol: str, days: list[str]) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for day in days:
        blob = _get(f"{VISION_BASE}/{symbol}/{symbol}-metrics-{day}.zip")
        if blob is None:  # unpublished / pre-listing day / wrong symbol
            missing.append(day)
            continue
        frames.append(_parse_metrics_zip(blob))
    return frames, missing


def download_metrics(pair: str, days: list[str], force_refresh: bool = False) -> dict:
    out = OUT_DIR / "metrics_5m" / f"{pair}.parquet"
    frames: list[pd.DataFrame] = []
    had_existing = out.exists() and not force_refresh
    if had_existing:
        old = pd.read_parquet(out)
        frames.append(old)
        # daily zip for day D ends at D+1 00:00, so resume from the max date itself
        resume_day = pd.Timestamp(old["ts_ms"].max(), unit="ms").date().isoformat()
        days = [d for d in days if d >= resume_day]
    new_frames, missing = _fetch_metrics_frames(pair, days)
    alias: str | None = None
    if not had_existing and not new_frames and days:
        # every requested day 404'd under the plain symbol -> try the 1000-prefixed contract
        alias_symbol = "1000" + pair
        alias_frames, alias_missing = _fetch_metrics_frames(alias_symbol, days)
        if alias_frames:
            new_frames, missing, alias = alias_frames, alias_missing, alias_symbol
    frames.extend(new_frames)
    if not frames:
        return {"pair": pair, "source": "metrics", "rows": 0, "status": "no_data"}
    df = pd.concat(frames, ignore_index=True)
    df["ts_ms"] = df["ts_ms"].astype("int64")
    for c in METRICS_KEEP[1:]:
        df[c] = df[c].astype("float64")
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    result = {
        "pair": pair, "source": "metrics", "rows": int(len(df)),
        "first": pd.Timestamp(df["ts_ms"].iloc[0], unit="ms").isoformat(),
        "last": pd.Timestamp(df["ts_ms"].iloc[-1], unit="ms").isoformat(),
        "days_fetched": len(days) - len(missing), "days_missing": len(missing),
        "status": "ok",
    }
    if force_refresh:
        result["force_refresh"] = True
    if alias:
        result["alias"] = alias
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="funding,metrics")
    ap.add_argument("--start", default="2022-07-01")
    ap.add_argument("--end", default="2026-07-06")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pairs", default="", help="comma list override (default: universe)")
    ap.add_argument("--no-manifest", action="store_true",
                     help="skip reading/writing manifest.json (avoid races with a concurrent run)")
    ap.add_argument("--force-refresh", action="store_true",
                    help="metrics only: ignore existing parquet and refetch the requested full window")
    args = ap.parse_args()

    pairs = (args.pairs.split(",") if args.pairs
             else [c.pair for c in load_universe()])
    sources = args.sources.split(",")
    start_ms = _day_ms(args.start)
    end_ms = _day_ms(args.end) + 86_400_000 - 1  # end date inclusive
    days = _days(date.fromisoformat(args.start), date.fromisoformat(args.end))

    tasks = [(p, src) for src in sources for p in pairs]
    print(f"{len(pairs)} pairs x {sources} = {len(tasks)} tasks, {args.start}..{args.end}")
    results, errors = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for p, src in tasks:
            if src == "funding":
                futs[ex.submit(download_funding, p, start_ms, end_ms)] = (p, src)
            elif src == "metrics":
                futs[ex.submit(download_metrics, p, days, args.force_refresh)] = (p, src)
            else:
                raise SystemExit(f"unknown source: {src}")
        for i, fut in enumerate(as_completed(futs), 1):
            p, src = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                if i % 10 == 0 or res["status"] != "ok":
                    print(f"[{i}/{len(tasks)}] {p} {src}: {res['status']} rows={res.get('rows', 0)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append({"pair": p, "source": src, "error": repr(e)})
                print(f"[{i}/{len(tasks)}] {p} {src}: ERROR {e!r}", flush=True)

    if args.no_manifest:
        print("--no-manifest: skipped manifest.json read/write")
    else:
        manifest_path = OUT_DIR / "manifest.json"
        existing = {}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
        for r in results:
            existing[f"{r['pair']}:{r['source']}"] = r
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(existing, indent=1))

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]
    print(f"\nDONE in {time.time() - t0:.0f}s: ok={len(ok)} "
          f"skipped={[(r['pair'], r['source'], r['status']) for r in skipped]} errors={len(errors)}")
    print(f"total rows: {sum(r['rows'] for r in ok):,}")
    if errors:
        print("ERRORS:", errors[:10])


if __name__ == "__main__":
    main()
