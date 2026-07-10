"""Targeted bookTicker download around liquidation-cascade events (spread validation).

Goal: for each liqrev_v2 trigger event, pull REAL best-bid/ask quotes from Binance
USD-M daily bookTicker dumps around the event, so we can measure storm-time spreads
and judge whether the live candidate's assumed 10bps RT MAKER cost is realistic.

Source: https://data.binance.vision/data/futures/um/daily/bookTicker/{CONTRACT}/
        {CONTRACT}-bookTicker-{YYYY-MM-DD}.zip  (one CSV per day, every best-bid/ask
        change). CSV columns (header row present):
          update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,
          transaction_time,event_time    (event_time ms; >1e14 => microseconds, /1000)

COVERAGE CLIFF (verified 2026-07-08 via the S3 bucket listing): daily bookTicker is
published ONLY for 2023-05-16 .. 2024-03-30 for every symbol checked (BTC/ETH/SOL/
TRX/DOGE/1000PEPE = 320 files each). It was DISCONTINUED after 2024-03-30, so any
event after that 404s and cannot be measured from this source.

1000-prefix alias: liqrev events use the plain klines symbol (e.g. PEPEUSDT); the
listed contract may be 1000-prefixed (1000PEPEUSDT). We try the plain contract first,
fall back to 1000<PAIR>, and cache the winning prefix per symbol. Relative spread
((ask-bid)/mid) is scale-invariant, so 1000x contract prices do not bias the summary.

Pipeline: dedupe (symbol,date) downloads -> in memory extract each event's [t, t+3h]
window -> downsample to price-change rows only (drop pure qty updates) -> append
per-symbol parquet -> compute per-event spread summary + aggregate table by year.

Output:
  research/data/binance_um/bookticker_events/{PAIR}.parquet
      (event_ts, ts, bid, bid_qty, ask, ask_qty)  dedup on (event_ts, ts)
  research/data/binance_um/bookticker_events/spread_summary.parquet

Usage:
  python download_bookticker_events.py
  python download_bookticker_events.py --workers 6 --start 2023-05-17
  python download_bookticker_events.py --smoke-symbols BTCUSDT,SOLUSDT   # test subset
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liqrev_v2 import detect_events  # noqa: E402
from universe import REPO_ROOT, load_universe  # noqa: E402

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/bookTicker"
OUT_DIR = REPO_ROOT / "research" / "data" / "binance_um" / "bookticker_events"
CSV_COLS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
            "best_ask_qty", "transaction_time", "event_time"]
WINDOW_MS = 3 * 3_600_000          # [t, t+3h] kept per event
HOUR_MS = 3_600_000
BOOKTICKER_ERA_END = "2024-03-30"  # last published daily dump (informational)

_tls = threading.local()
_prefix_cache: dict[str, str] = {}   # symbol -> "plain" | "1000"
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
    """Return bytes; None on 404; retry with backoff on other errors, then raise."""
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
    return f"{VISION_BASE}/{contract}/{contract}-bookTicker-{date}.zip"


def _fetch_day(symbol: str, date: str) -> bytes | None:
    """Fetch one day's zip, resolving the plain vs 1000-prefixed contract (cached)."""
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
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0])
    header = 0 if raw[:20].startswith(b"update_id") else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=CSV_COLS,
                     usecols=["best_bid_price", "best_bid_qty", "best_ask_price",
                              "best_ask_qty", "event_time"])
    et = df["event_time"].astype("int64")
    if et.max() > 1e14:              # microseconds (defensive; no such files exist today)
        et = et // 1000
    df["event_time"] = et
    return df


def _extract_window(day: pd.DataFrame, event_ts: int) -> pd.DataFrame:
    """Keep [event_ts, event_ts+3h]; downsample to bid/ask price-change rows only."""
    lo, hi = event_ts, event_ts + WINDOW_MS
    w = day[(day["event_time"] >= lo) & (day["event_time"] <= hi)]
    if w.empty:
        return w.iloc[0:0]
    w = w.sort_values("event_time", kind="stable")
    bid, ask = w["best_bid_price"], w["best_ask_price"]
    keep = (bid != bid.shift()) | (ask != ask.shift())   # first row: shift->NaN->True
    w = w[keep]
    return pd.DataFrame({
        "event_ts": event_ts,
        "ts": w["event_time"].to_numpy("int64"),
        "bid": w["best_bid_price"].to_numpy("float64"),
        "bid_qty": w["best_bid_qty"].to_numpy("float64"),
        "ask": w["best_ask_price"].to_numpy("float64"),
        "ask_qty": w["best_ask_qty"].to_numpy("float64"),
    })


def process_file(symbol: str, date: str, events: list[int]) -> dict:
    """Download+parse (symbol,date), extract every listed event window. events=[event_ts_ms]."""
    try:
        blob = _fetch_day(symbol, date)
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "date": date, "status": "error",
                "err": repr(e), "df": None, "n_quotes": 0}
    if blob is None:
        return {"symbol": symbol, "date": date, "status": "404", "df": None, "n_quotes": 0}
    day = _parse_day(blob)
    parts = [_extract_window(day, ev) for ev in events]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True) if parts else None
    return {"symbol": symbol, "date": date, "status": "ok",
            "df": df, "n_quotes": int(len(df)) if df is not None else 0}


def _write_symbol(symbol: str, frames: list[pd.DataFrame]) -> int:
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    out = OUT_DIR / f"{symbol}.parquet"
    if out.exists():
        df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
    df = (df.drop_duplicates(["event_ts", "ts"])
            .sort_values(["event_ts", "ts"], kind="stable").reset_index(drop=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)


# ------------------------------------------------------------------- event list
def build_events(pairs: list[str], start: str) -> list[dict]:
    start_ts = pd.Timestamp(start, tz="UTC")
    events: list[dict] = []
    for i, pair in enumerate(pairs, 1):
        ev = detect_events(pair)
        for _, r in ev.iterrows():
            t = r["ts"]
            if t < start_ts:
                continue
            events.append({"symbol": pair, "t": t, "event_ts": int(t.value // 1_000_000)})
        if i % 40 == 0:
            print(f"  events scan [{i}/{len(pairs)}] -> {len(events)}", flush=True)
    return events


def build_tasks(events: list[dict]) -> dict[tuple[str, str], list[int]]:
    """(symbol,date) -> [event_ts_ms]; adds next date when hour(t) >= 22 (window crosses midnight)."""
    fe: dict[tuple[str, str], list[int]] = defaultdict(list)
    for e in events:
        t, sym, ets = e["t"], e["symbol"], e["event_ts"]
        fe[(sym, t.date().isoformat())].append(ets)
        if t.hour >= 22:
            fe[(sym, (t + pd.Timedelta(days=1)).date().isoformat())].append(ets)
    return fe


# ----------------------------------------------------------------------- spread
def spread_summary() -> pd.DataFrame:
    rows = []
    for pq in sorted(OUT_DIR.glob("*.parquet")):
        if pq.name == "spread_summary.parquet":
            continue
        sym = pq.stem
        df = pd.read_parquet(pq)
        mid = (df["bid"] + df["ask"]) / 2.0
        df = df.assign(spread_bps=(df["ask"] - df["bid"]) / mid * 1e4)
        df = df[(mid > 0) & df["spread_bps"].notna()]
        for ets, g in df.groupby("event_ts"):
            rel = g["ts"] - ets
            entry = g[(rel >= HOUR_MS) & (rel <= 2 * HOUR_MS)]["spread_bps"]
            casc = g[(rel >= 0) & (rel < HOUR_MS)]["spread_bps"]
            rows.append({
                "event_ts": int(ets), "symbol": sym,
                "med_spread_entry_bps": float(entry.median()) if len(entry) else float("nan"),
                "p90_spread_entry_bps": float(entry.quantile(0.90)) if len(entry) else float("nan"),
                "med_spread_cascade_bps": float(casc.median()) if len(casc) else float("nan"),
                "p90_spread_cascade_bps": float(casc.quantile(0.90)) if len(casc) else float("nan"),
                "n_quotes": int(len(g)),
            })
    return pd.DataFrame(rows)


def print_aggregate(summ: pd.DataFrame) -> None:
    if summ.empty:
        print("no spread summary rows (no covered events).")
        return
    s = summ.dropna(subset=["med_spread_entry_bps"]).copy()
    s["year"] = pd.to_datetime(s["event_ts"], unit="ms", utc=True).dt.year
    print("\n=== ENTRY-WINDOW MEDIAN SPREAD (bps) across events, by year ===")
    print(f"{'year':>6} {'n':>5} {'median':>8} {'p75':>8} {'p90':>8} "
          f"{'>10bps':>7} {'>25bps':>7}")
    for grp in (list(s.groupby("year")) + [("ALL", s)]):
        yr, g = grp
        m = g["med_spread_entry_bps"]
        print(f"{str(yr):>6} {len(g):>5} {m.median():>8.2f} {m.quantile(.75):>8.2f} "
              f"{m.quantile(.90):>8.2f} {(m > 10).mean():>7.1%} {(m > 25).mean():>7.1%}")
    m = s["med_spread_entry_bps"]
    print(f"\nShare of events with median ENTRY spread >10bps: {(m > 10).mean():.1%}  "
          f"({int((m > 10).sum())}/{len(m)})")
    print(f"Share of events with median ENTRY spread >25bps: {(m > 25).mean():.1%}  "
          f"({int((m > 25).sum())}/{len(m)})")


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--start", default="2023-05-17", help="keep events with ts >= this (bookTicker era)")
    ap.add_argument("--smoke-symbols", default="", help="restrict to these pairs (test)")
    args = ap.parse_args()

    pairs = [c.pair for c in load_universe()]
    if args.smoke_symbols:
        keep = set(args.smoke_symbols.split(","))
        pairs = [p for p in pairs if p in keep]
    print(f"universe: {len(pairs)} pairs | keeping events ts >= {args.start} "
          f"(bookTicker published {'2023-05-16'}..{BOOKTICKER_ERA_END})")

    events = build_events(pairs, args.start)
    tasks = build_tasks(events)
    task_list = sorted(tasks.keys())
    print(f"events qualifying: {len(events)} | dedup (symbol,date) files to fetch: {len(task_list)}")

    pending = Counter(sym for sym, _ in task_list)
    sym_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    n_ok = n_404 = n_err = n_quotes = 0
    errors: list[str] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_file, sym, date, tasks[(sym, date)]): (sym, date)
                for (sym, date) in task_list}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, date = futs[fut]
            res = fut.result()
            if res["status"] == "ok":
                n_ok += 1
                n_quotes += res["n_quotes"]
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
            if i % 25 == 0:
                print(f"[{i}/{len(task_list)}] ok={n_ok} 404={n_404} err={n_err} "
                      f"quotes={n_quotes:,} ({time.time()-t0:.0f}s)", flush=True)

    # flush any symbol whose pending never hit 0 (shouldn't happen, defensive)
    for sym, frames in list(sym_frames.items()):
        _write_symbol(sym, frames)

    gb = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet")) / 1e9
    print(f"\nDOWNLOAD DONE in {time.time()-t0:.0f}s: files ok={n_ok} 404={n_404} "
          f"err={n_err} | stored quotes={n_quotes:,} | parquet {gb:.3f} GB")
    if errors:
        print("errors (first 20):", errors)

    summ = spread_summary()
    if not summ.empty:
        out = OUT_DIR / "spread_summary.parquet"
        summ.to_parquet(out, index=False)
        print(f"spread summary -> {out}  ({len(summ)} events)")
    print_aggregate(summ)


if __name__ == "__main__":
    main()
