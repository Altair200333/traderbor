"""Targeted order-book DEPTH download around liquidation-cascade events.

FEATURE AUDIT (pre-registered question #14): does resting order-book depth at the
liqrev trigger add conditioning signal beyond the BTC-context overlay we already
trade? This script pulls the raw depth snapshots; bookdepth_audit.py does the test.

SOURCE (verified 2026-07-08 via the S3 bucket): Binance USD-M futures bookDepth
daily dumps, published 2023-01-01 .. present, ~0.5 MB/day/symbol:
  https://data.binance.vision/data/futures/um/daily/bookDepth/{CONTRACT}/
      {CONTRACT}-bookDepth-{YYYY-MM-DD}.zip   (ZIP of one CSV, header row present)

ACTUAL CSV SCHEMA (inspected BTCUSDT 2023-01-02 / 2024-06-03, ETHUSDT 2025-03-15):
  columns: timestamp,percentage,depth,notional
    timestamp  str  "YYYY-MM-DD HH:MM:SS"  UTC, NAIVE (no tz marker). Same string
               format in 2023 and 2025 -- no epoch/microsecond variant observed;
               a numeric-epoch branch is kept defensively (>1e14 => microseconds/1000).
    percentage int  band, one of {-5,-4,-3,-2,-1,1,2,3,4,5}. NOT a price level:
               it is the % distance from mid. NEGATIVE = BID side (below price),
               POSITIVE = ASK side (above price). depth/notional at |k| are the
               CUMULATIVE resting size within k% of mid (Binance bookDepth
               semantics; monotone-increasing in |k| in every file inspected).
    depth      float  cumulative base-asset quantity within the band.
    notional   float  cumulative quote (USD) value within the band. <-- we use this.
  Cadence: ~2860-2880 snapshots/day (one every ~25-30 s), 10 rows/snapshot.

CONTRACT NAMING: ml_dataset symbols are the plain klines symbol (e.g. PEPEUSDT).
Several are listed ONLY 1000-prefixed on futures (verified: 1000PEPE/1000BONK/
1000SHIB exist, plain 404s). We try the plain contract first, fall back to
1000<PAIR>, and cache the winning prefix per symbol. All audit features are
notional RATIOS on the SAME contract (imbalance / slope / wall-pull) OR notional
normalised by spot quote-volume -- the 1000x price multiplier does not bias USD
notional, so the prefix choice is label-neutral.

EVENT SET: every research/data/liqrev/ml_dataset.parquet event with
ts >= 2023-01-02 (1209 events, 143 symbols). For each event we fetch the
trigger-date file, plus the PREVIOUS calendar day's file WHEN trigger hour < 12
-- that is exactly (and only) the case where the [t-12h, t+1h] keep-window reaches
into the prior UTC day (covers the -6h wall-pull baseline for early triggers).
Fetching prev-day for hour>=12 triggers would add zero in-window rows, so it is
skipped (DRY; disclosed).

KEEP WINDOW (pre-registered): from each downloaded day keep only rows with
snapshot ts in [t-12h, t+1h] for each event the file serves. Stored LONG per
symbol so the audit can pick the last snapshot <= trigger close, the snapshot
nearest t-6h (wall-pull), and the ~t-12h earlier baseline.

OUTPUT: research/data/binance_um/bookdepth_events/{PAIR}.parquet   (PAIR = plain
ml_dataset symbol) with columns
  event_ts (int64 ms, trigger-bar open), ts (int64 ms, snapshot), percentage (int8),
  depth (float64), notional (float64)      dedup on (event_ts, ts, percentage).

Usage:
  python download_bookdepth_events.py                 # full run, 8 workers
  python download_bookdepth_events.py --workers 6
  python download_bookdepth_events.py --smoke-symbols BTCUSDT,SOLUSDT
  python download_bookdepth_events.py --limit 40      # first N (symbol,date) tasks
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
ML_DATASET = REPO_ROOT / "research" / "data" / "liqrev" / "ml_dataset.parquet"
OUT_DIR = REPO_ROOT / "research" / "data" / "binance_um" / "bookdepth_events"
CSV_COLS = ["timestamp", "percentage", "depth", "notional"]
HOUR_MS = 3_600_000
PRE_MS = 12 * HOUR_MS   # keep t-12h .. t+1h
POST_MS = 1 * HOUR_MS
START_DATE = "2023-01-02"

_tls = threading.local()
_prefix_cache: dict[str, str] = {}     # symbol -> "plain" | "1000"
_prefix_lock = threading.Lock()


# --------------------------------------------------------------------------- io
def _session():
    if not hasattr(_tls, "s"):
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = "traderbor-research/1.0"
        _tls.s = s
    return _tls.s


def _get(url: str, retries: int = 3) -> bytes | None:
    """Return bytes; None on 404; retry w/ backoff on other errors, then raise."""
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


def _url(contract: str, date: str) -> str:
    return f"{VISION_BASE}/{contract}/{contract}-bookDepth-{date}.zip"


def _fetch_day(symbol: str, date: str) -> bytes | None:
    """Fetch one day's zip, resolving plain vs 1000-prefixed contract (cached)."""
    with _prefix_lock:
        known = _prefix_cache.get(symbol)
    cands = ([symbol] if known == "plain"
             else ["1000" + symbol] if known == "1000"
             else [symbol, "1000" + symbol])
    for c in cands:
        blob = _get(_url(c, date))
        if blob is not None:
            with _prefix_lock:
                _prefix_cache[symbol] = "1000" if c.startswith("1000") else "plain"
            return blob
    return None


# ---------------------------------------------------------------------- parsing
def _parse_day(blob: bytes) -> pd.DataFrame:
    """Parse a day CSV -> DataFrame(ts int64 ms, percentage int8, depth, notional)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0])
    header = 0 if raw[:9] == b"timestamp" else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=CSV_COLS)
    tcol = df["timestamp"]
    if pd.api.types.is_numeric_dtype(tcol):                 # defensive epoch branch
        et = tcol.astype("int64")
        if et.max() > 1e14:                                 # microseconds -> ms
            et = et // 1000
        ts_ms = et.to_numpy("int64")
    else:                                                    # observed: string UTC
        # pandas 3.0 parses to datetime64[us]; go via [ms] so the int64 is ms.
        dt = pd.to_datetime(tcol, utc=True).dt.tz_localize(None)
        ts_ms = dt.to_numpy(dtype="datetime64[ms]").astype("int64")
    return pd.DataFrame({
        "ts": ts_ms,
        "percentage": df["percentage"].astype("int8").to_numpy(),
        "depth": df["depth"].astype("float64").to_numpy(),
        "notional": df["notional"].astype("float64").to_numpy(),
    })


def _extract_window(day: pd.DataFrame, event_ts: int) -> pd.DataFrame:
    """Keep snapshot rows with ts in [event_ts-12h, event_ts+1h]; tag event_ts."""
    lo, hi = event_ts - PRE_MS, event_ts + POST_MS
    w = day[(day["ts"] >= lo) & (day["ts"] <= hi)]
    if w.empty:
        return w.iloc[0:0]
    return pd.DataFrame({
        "event_ts": np.int64(event_ts),
        "ts": w["ts"].to_numpy("int64"),
        "percentage": w["percentage"].to_numpy("int8"),
        "depth": w["depth"].to_numpy("float64"),
        "notional": w["notional"].to_numpy("float64"),
    })


def process_file(symbol: str, date: str, events: list[int]) -> dict:
    """Download+parse (symbol,date), extract every listed event window."""
    try:
        blob = _fetch_day(symbol, date)
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "date": date, "status": "error",
                "err": repr(e), "df": None, "n_rows": 0}
    if blob is None:
        return {"symbol": symbol, "date": date, "status": "404", "df": None, "n_rows": 0}
    day = _parse_day(blob)
    parts = [_extract_window(day, ev) for ev in events]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True) if parts else None
    return {"symbol": symbol, "date": date, "status": "ok",
            "df": df, "n_rows": int(len(df)) if df is not None else 0}


def _write_symbol(symbol: str, frames: list[pd.DataFrame]) -> int:
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    out = OUT_DIR / f"{symbol}.parquet"
    if out.exists():
        df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
    df = (df.drop_duplicates(["event_ts", "ts", "percentage"])
            .sort_values(["event_ts", "ts", "percentage"], kind="stable")
            .reset_index(drop=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)


# ------------------------------------------------------------------- event list
def build_events(start: str, symbols: set[str] | None) -> list[dict]:
    ml = pd.read_parquet(ML_DATASET, columns=["symbol", "ts"])
    start_ts = pd.Timestamp(start, tz="UTC")
    ml = ml[ml["ts"] >= start_ts]
    if symbols is not None:
        ml = ml[ml["symbol"].isin(symbols)]
    events = []
    for _, r in ml.iterrows():
        t = r["ts"]
        events.append({"symbol": r["symbol"], "t": t,
                       "event_ts": int(t.value // 1_000_000)})
    return events


def build_tasks(events: list[dict]) -> dict[tuple[str, str], list[int]]:
    """(symbol,date)->[event_ts]; adds PREV day iff hour(t)<12 (window reaches prior day)."""
    fe: dict[tuple[str, str], list[int]] = defaultdict(list)
    for e in events:
        t, sym, ets = e["t"], e["symbol"], e["event_ts"]
        fe[(sym, t.date().isoformat())].append(ets)
        if t.hour < 12:
            prev = (t - pd.Timedelta(days=1)).date().isoformat()
            fe[(sym, prev)].append(ets)
    return fe


# --------------------------------------------------------------------- coverage
def coverage_report(events: list[dict]) -> dict:
    """Per-event: is there >=1 snapshot inside the trigger bar [t, t+1h]?"""
    stored: dict[str, pd.DataFrame] = {}
    matched = 0
    for e in events:
        sym, ets = e["symbol"], e["event_ts"]
        if sym not in stored:
            p = OUT_DIR / f"{sym}.parquet"
            stored[sym] = pd.read_parquet(p, columns=["event_ts", "ts"]) if p.exists() \
                else pd.DataFrame(columns=["event_ts", "ts"])
        d = stored[sym]
        sub = d[d["event_ts"] == ets]
        if len(sub) and ((sub["ts"] >= ets) & (sub["ts"] <= ets + HOUR_MS)).any():
            matched += 1
    return {"events_total": len(events), "events_with_trigger_snap": matched,
            "events_missing": len(events) - matched}


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--smoke-symbols", default="")
    ap.add_argument("--limit", type=int, default=0, help="cap number of (symbol,date) tasks (test)")
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    syms = set(args.smoke_symbols.split(",")) if args.smoke_symbols else None
    events = build_events(args.start, syms)
    tasks = build_tasks(events)
    task_list = sorted(tasks.keys())
    if args.limit:
        task_list = task_list[:args.limit]
    n_sym = len({s for s, _ in task_list})
    print(f"events>={args.start}: {len(events)} | symbols: {n_sym} | "
          f"dedup (symbol,date) files: {len(task_list)}", flush=True)

    pending = Counter(sym for sym, _ in task_list)
    sym_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    n_ok = n_404 = n_err = n_rows = 0
    errors: list[str] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_file, s, d, tasks[(s, d)]): (s, d)
                for (s, d) in task_list}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, date = futs[fut]
            res = fut.result()
            if res["status"] == "ok":
                n_ok += 1
                n_rows += res["n_rows"]
                if res["df"] is not None and len(res["df"]):
                    sym_frames[sym].append(res["df"])
            elif res["status"] == "404":
                n_404 += 1
            else:
                n_err += 1
                if len(errors) < 20:
                    errors.append(f"{sym} {date}: {res.get('err')}")
            pending[sym] -= 1
            if pending[sym] == 0:
                _write_symbol(sym, sym_frames.pop(sym, []))
            if i % 50 == 0:
                print(f"[{i}/{len(task_list)}] ok={n_ok} 404={n_404} err={n_err} "
                      f"rows={n_rows:,} ({time.time()-t0:.0f}s)", flush=True)

    for sym, frames in list(sym_frames.items()):   # defensive flush
        _write_symbol(sym, frames)

    gb = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet")) / 1e9
    cov = coverage_report(events)
    print(f"\nDOWNLOAD DONE in {time.time()-t0:.0f}s: files ok={n_ok} 404={n_404} "
          f"err={n_err} | stored rows={n_rows:,} | parquet {gb:.3f} GB")
    print(f"coverage: {cov['events_with_trigger_snap']}/{cov['events_total']} events "
          f"have a trigger-bar snapshot ({cov['events_missing']} missing)")
    if errors:
        print("errors (first 20):", errors)


if __name__ == "__main__":
    main()
