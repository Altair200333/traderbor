"""Q23 — liqrev v2 POINT-IN-TIME universe validation (pre-registered, one-shot).

QUESTION: does liqrev v2 survive a point-in-time universe that includes coins
later delisted (and coins never selected into the 2026 survivor list of 149)?
Survivorship is the biggest standing caveat on the deployable strategy: for a
LONG strategy on crashing coins, a survivors-only universe is flattered because
cascades on coins that died were never observed. VALIDATION ONLY: every
strategy parameter is FROZEN; the only change is the universe.

================================ FROZEN SPEC ==================================
(written before the first analysis run; nothing below changes afterwards)

STUDY WINDOW: triggers with 2022-07-01 00:00 UTC <= ts < 2026-07-06 00:00 UTC.
DEV = ts < 2025-01-01, LIVE = ts >= 2025-01-01 (repo protocol).

UNIVERSE ENUMERATION (task 1):
  - Source: S3 listing of data.binance.vision, prefix
    data/futures/um/monthly/klines/ (delimiter=/, paginated by marker).
  - Include: symbols ending in USDT.  Exclude (declared before run):
      * delivery-dated names (suffix _YYMMDD)
      * non-USDT quote (BUSD/USDC/... folders simply don't end in USDT)
      * "...USDTSETTLED" folders are ALIASES for settled contracts: mapped to
        the plain name; used as a data source iff the plain folder is absent
        or missing months.
      * stable/fiat bases: USDC BUSD TUSD USDP DAI FDUSD AEUR EUR EURI USDE
        USD1 BFUSD XUSD USDF
      * index products (not coins): BTCDOM DEFI BLUEBIRD FOOTBALL
      * BTCUSD1 style non-standard folders (base must be [A-Z0-9]+, and the
        folder must contain at least one monthly 1h zip)
  - Survivor mapping: a symbol is "survivor-covered" iff it equals a survivor
    pair (universe.py, 149) or equals one after stripping a leading
    1000000/1000/1M multiplier prefix (e.g. 1000PEPEUSDT -> PEPEUSDT).
    Survivor-covered symbols are NOT added (their spot-data detection is the
    parity baseline).  Remaining = PIT-extra candidates.  Candidates whose
    data ends before 2022-07-01 are "pre-window" (dead before study, excluded,
    counted).  Within PIT-extra candidates, if two folders map to the same
    stripped base, keep the one with more monthly files (declared dedupe).

DATA (task 2):
  - 1h UM-futures klines, monthly zips 2022-05..2026-06 + daily zips
    2026-07-01..2026-07-05, resumable, ~0.3s per-worker sleep, into
    research/data/binance_um/pit_extra/klines_1h/{SYM}.parquet.
    (2022-05 start = 30d volume-gate warmup before window start.)
  - futures metrics daily zips (5m rows, sum_open_interest) — same source and
    parser as research/data/perp/metrics_5m (download_perp.py conventions) —
    into research/data/binance_um/pit_extra/metrics/{SYM}.parquet.
    COST CONTROL (declared): metrics are fetched only for days where the
    symbol's $1M volume gate passes (expanded +-2 days, clipped to
    2022-06-28..2026-07-05).  Detection requires liq_ok, so days outside gate
    windows can never produce an event; this changes nothing.
  - A symbol without OI data cannot enter detection (exactly like research:
    missing metrics parquet -> no events).  Coverage reported honestly.

PIT UNIVERSE (task 3): at each hour a symbol is in-universe iff
  trailing 30d median daily quote volume > $1M (rolling(30).median on daily
  sums, ffilled to hours — EXACT liqrev_study code; the 30-day rolling window
  itself enforces "traded >= 30 days").  Computed from the symbol's own data
  only.  Honesty note: survivors' gate uses SPOT quote volume (research
  convention, unchanged for parity); PIT-extra symbols use FUTURES quote
  volume (only data that exists for them) — a declared, unavoidable deviation.

DETECTOR (task 4, frozen from liqrev_study/liqrev_v2):
  1h grid; ret_6h <= -8% AND OI_6h <= -10% (5m OI resampled 1h last, ffilled);
  liq gate as above; 24h per-symbol cooldown.  Survivors run through
  liqrev_v2.detect_events VERBATIM (spot 1h + perp/metrics_5m) — parity target
  1308 canonical events (tolerance +-2% for data-refresh drift; exact diff vs
  ml_dataset.parquet reported).  PIT-extra symbols: identical math on futures
  1h klines + pit metrics.  TAIL RULE: frozen detector drops events within
  25 bars of data end; for PIT-extra symbols whose data ends before
  2026-07-03 (delisted), such events are RETAINED and force-closed at the last
  available close (delisting realism, task 5); for alive PIT-extra symbols and
  all survivors the frozen drop rule stands.

TRADE CONVENTION (task 5, = deployed config, liqrev_v2.simulate maker mode +
  the deploy-spec disaster stop):
  maker limit at trigger-bar close, valid next 1h bar only (TTL 1h), filled
  iff next-bar low < limit; unfilled = skipped.  Disaster stop -20% from
  entry, gap-aware (exit at min(open, stop)), checked from entry bar over the
  hold; else exit at close of bar i+24 (~+24h).  Cost cells 25bps and 10bps
  RT.  DELISTING REALISM: if data ends while the position is open, close at
  the last available close, flag forced_close; sensitivity = extra haircut of
  0 / 5 / 10 pct (absolute return subtraction) on forced closes only.
  1h-sim caveat (declared): research's 1m-path validation retained 93.8% of
  the 1h edge (results_1m.json / audit 4.3a); we cite it, we do not re-run 1m.

PORTFOLIO (task 6): filled events in ts order; 15 slots x 1/15 equity; slot
  busy until ts+24h; taken iff free slot; eq *= 1 + w*net_ret/15
  (liqrev_ml_model machinery).  Overlay ON: w = min(2, 2*P),
  P = searchsorted(frozen scores_sorted, -btc_ret_6h, side='right')/839 from
  bot/artifacts/liqrev_overlay.json; btc_ret_6h = BTCUSDT spot 1h close
  pct_change(6) at trigger ts (missing -> w=1, counted).  10bps cell.
  Runs: survivors-only baseline vs PIT (survivors + new events merged), each
  on full span and LIVE (>= 2025-01-01) windows; headline haircut 0%,
  sensitivity rerun at 10% forced-close haircut.

PRE-REGISTERED VERDICT RULE (frozen before first run):
  PASS iff ALL of
   (a) NEW-event pooled net mean (filled, 25bps, 0% haircut) > -1.0%/event
   (b) PIT full-span portfolio CAGR >= 0.60 * survivors-baseline full-span
       CAGR (same machinery, overlay ON, 10bps, 0% haircut)
   (c) LIVE (>=2025-01-01) PIT portfolio total return > 0
  else FAIL, clause by clause.
  DATA-BLOCKED (not a pass) iff delisted-symbol data proves unavailable:
   (d1) < 80% of in-window PIT-extra candidates yield kline data, or
   (d2) < 70% of gate-qualified PIT-extra symbols yield any OI metrics rows.
  Survivor parity: |n_survivor_events - 1308| <= 26 required; otherwise the
  study is flagged PARITY-FAIL and the verdict is reported as unreliable.

HONESTY LEDGER (reported in output): OI coverage per gate-qualified symbol;
  futures-vs-spot volume gate deviation for new symbols; day-clustering of
  NEW events (distinct event days; share of days overlapping survivor
  events); forced-close counts; any BUSD-only-era coins missed by the USDT
  quote filter.

Artifacts: research/data/binance_um/pit_extra/{klines_1h,metrics}/,
  pit_symbols.json, klines_manifest.json, metrics_manifest.json (same dir);
  research/data/liqrev/results_pit.json (all tables).

Usage:
  python liqrev_pit_validation.py enumerate
  python liqrev_pit_validation.py download-klines  [--workers 10]
  python liqrev_pit_validation.py download-metrics [--workers 10]
  python liqrev_pit_validation.py analyze
================================================================================
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402
from liqrev_v2 import detect_events, HOLD_BARS, KL_DIR, OI_DIR  # noqa: E402
from download_perp import _parse_metrics_zip, METRICS_KEEP  # noqa: E402
from download_binance_vision import COLS as KL_COLS, KEEP as KL_KEEP  # noqa: E402

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION = "https://data.binance.vision"
PIT_DIR = REPO_ROOT / "research" / "data" / "binance_um" / "pit_extra"
PIT_KL = PIT_DIR / "klines_1h"
PIT_MET = PIT_DIR / "metrics"
ART_DIR = REPO_ROOT / "research" / "data" / "liqrev"
OVERLAY_PATH = REPO_ROOT / "bot" / "artifacts" / "liqrev_overlay.json"

SPAN_START = pd.Timestamp("2022-07-01", tz="UTC")
SPAN_END = pd.Timestamp("2026-07-06", tz="UTC")
LIVE_START = pd.Timestamp("2025-01-01", tz="UTC")
DELISTED_CUT = pd.Timestamp("2026-07-03", tz="UTC")  # data ends before -> delisted
KL_MONTH_LO, KL_MONTH_HI = "2022-05", "2026-06"
JULY_DAYS = [f"2026-07-{d:02d}" for d in range(1, 6)]
MET_DAY_LO, MET_DAY_HI = "2022-06-28", "2026-07-05"
STOP_PCT = 0.20
SLOTS = 15
STABLE_BASES = {"USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD", "AEUR", "EUR",
                "EURI", "USDE", "USD1", "BFUSD", "XUSD", "USDF"}
INDEX_BASES = {"BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL"}
MULT_RE = re.compile(r"^(1000000|1000|1M)")
SLEEP = 0.3  # per-worker politeness sleep between HTTP requests

_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "traderbor-research/1.0"
        _tls.s = s
    return _tls.s


def _get(url: str, retries: int = 4) -> bytes | None:
    time.sleep(SLEEP)
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
            time.sleep(2 * (attempt + 1))
    return None


def s3_list(prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    """Paginated S3 list: returns (keys, common_prefixes)."""
    keys: list[str] = []
    prefs: list[str] = []
    marker = ""
    while True:
        url = f"{S3}?prefix={quote(prefix)}"
        if delimiter:
            url += f"&delimiter={delimiter}"
        if marker:
            url += f"&marker={quote(marker)}"
        xml = _get(url)
        if xml is None:
            break
        x = xml.decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        prefs += re.findall(r"<Prefix>([^<]+)</Prefix>", x)[1:]  # [0] = query prefix
        if "<IsTruncated>true</IsTruncated>" not in x:
            break
        nm = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)
        marker = nm[0] if nm else (keys[-1] if keys else (prefs[-1] if prefs else ""))
        if not marker:
            break
    return keys, prefs


# ------------------------------------------------------------- enumeration ---
def cmd_enumerate() -> dict:
    PIT_DIR.mkdir(parents=True, exist_ok=True)
    _, prefs = s3_list("data/futures/um/monthly/klines/")
    folders = [p.split("/")[-2] for p in prefs]
    survivors = {c.pair for c in load_universe()}

    def strip_mult(s: str) -> str:
        return MULT_RE.sub("", s)

    settled = {s[:-len("SETTLED")]: s for s in folders if s.endswith("USDTSETTLED")}
    usdt = [s for s in folders if s.endswith("USDT")
            and not re.search(r"_\d{6}$", s) and re.fullmatch(r"[A-Z0-9]+", s)]
    excluded_stable = [s for s in usdt if s[:-4] in STABLE_BASES]
    excluded_index = [s for s in usdt if s[:-4] in INDEX_BASES]
    clean = [s for s in usdt
             if s[:-4] not in STABLE_BASES and s[:-4] not in INDEX_BASES]
    survivor_covered = [s for s in clean
                        if s in survivors or strip_mult(s) in survivors]
    candidates = [s for s in clean if s not in survivor_covered]
    # settled-only folders (plain absent) also become candidates
    settled_only = [plain for plain, full in settled.items()
                    if plain not in folders and plain.endswith("USDT")
                    and plain not in survivors and strip_mult(plain) not in survivors
                    and plain[:-4] not in STABLE_BASES and plain[:-4] not in INDEX_BASES]
    candidates = sorted(set(candidates) | set(settled_only))
    # declared dedupe: same stripped base among candidates -> resolved at
    # download time by file count; here we just record the collision groups
    by_base: dict[str, list[str]] = {}
    for s in candidates:
        by_base.setdefault(strip_mult(s), []).append(s)
    collisions = {b: ss for b, ss in by_base.items() if len(ss) > 1}

    # delisted-dump spot-check (task 2 verification)
    spot = {}
    for s in ["FTTUSDT", "SRMUSDT", "TORNUSDT"]:
        k, _ = s3_list(f"data/futures/um/monthly/klines/{s}/1h/", delimiter=None)
        zips = [x for x in k if x.endswith(".zip")]
        spot[s] = {"n_monthly_zips": len(zips),
                   "first": zips[0].split("-1h-")[-1][:7] if zips else None,
                   "last": zips[-1].split("-1h-")[-1][:7] if zips else None}

    out = {"run_utc": datetime.now(timezone.utc).isoformat(),
           "n_folders_total": len(folders),
           "n_usdt_nondated": len(usdt),
           "n_excluded_stable": len(excluded_stable),
           "excluded_stable": excluded_stable,
           "n_excluded_index": len(excluded_index),
           "n_clean": len(clean),
           "n_survivor_covered": len(survivor_covered),
           "survivor_covered": sorted(survivor_covered),
           "n_survivors_without_um_folder":
               len(survivors) - len({s for s in survivor_covered}),
           "n_candidates": len(candidates),
           "candidates": candidates,
           "settled_aliases": settled,
           "collision_groups": collisions,
           "delisted_spot_check": spot}
    (PIT_DIR / "pit_symbols.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if not isinstance(v, list) or len(v) < 25}, indent=1))
    print(f"candidates: {len(candidates)}  -> {PIT_DIR / 'pit_symbols.json'}")
    return out


# ---------------------------------------------------------------- klines -----
def _parse_kline_zip(blob: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(blob))
    raw = zf.read(zf.namelist()[0])
    header = 0 if raw[:9] == b"open_time" else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=KL_COLS)[KL_KEEP]
    big = df["open_time"] > 1e14
    if big.any():
        df.loc[big, "open_time"] = df.loc[big, "open_time"] // 1000
    return df


def _dl_klines_one(sym: str, settled: dict[str, str]) -> dict:
    folders = [sym]
    if sym in settled:
        folders.append(settled[sym])
    keys: dict[str, str] = {}  # month/day token -> zip key (plain preferred)
    first_month = None
    for f in folders:
        ks, _ = s3_list(f"data/futures/um/monthly/klines/{f}/1h/", delimiter=None)
        for k in ks:
            if not k.endswith(".zip"):
                continue
            tok = k.split("-1h-")[-1][:7]
            if first_month is None or tok < first_month:
                first_month = tok
            if KL_MONTH_LO <= tok <= KL_MONTH_HI and tok not in keys:
                keys[tok] = k
    if not keys:
        return {"symbol": sym, "status": "pre_window" if first_month else "no_data",
                "first_month": first_month}
    out_path = PIT_KL / f"{sym}.parquet"
    last_ts = -1
    old = None
    if out_path.exists():
        old = pd.read_parquet(out_path)
        if len(old):
            last_ts = int(old["open_time"].max())
    parts = []
    for tok in sorted(keys):
        y, m = int(tok[:4]), int(tok[5:7])
        nxt = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
        if nxt.timestamp() * 1000 <= last_ts:
            continue
        blob = _get(f"{VISION}/{keys[tok]}")
        if blob is not None:
            parts.append(_parse_kline_zip(blob))
    if max(keys) == KL_MONTH_HI:  # alive through last full month -> July tail
        for ymd in JULY_DAYS:
            d0 = pd.Timestamp(ymd, tz="UTC")
            if (d0 + pd.Timedelta("1D")).timestamp() * 1000 <= last_ts:
                continue
            for f in folders:
                blob = _get(f"{VISION}/data/futures/um/daily/klines/{f}/1h/"
                            f"{f}-1h-{ymd}.zip")
                if blob is not None:
                    parts.append(_parse_kline_zip(blob))
                    break
    if not parts and old is None:
        return {"symbol": sym, "status": "no_data", "first_month": first_month}
    new = pd.concat([old] + parts if old is not None else parts, ignore_index=True)
    new = (new.drop_duplicates("open_time").sort_values("open_time")
           .reset_index(drop=True))
    new.to_parquet(out_path, index=False)
    t0 = pd.to_datetime(int(new["open_time"].iloc[0]), unit="ms", utc=True)
    t1 = pd.to_datetime(int(new["open_time"].iloc[-1]), unit="ms", utc=True)
    return {"symbol": sym, "status": "ok", "rows": int(len(new)),
            "first_month": first_month,
            "first": f"{t0:%Y-%m-%d}", "last": f"{t1:%Y-%m-%d}"}


def cmd_download_klines(workers: int) -> None:
    meta = json.loads((PIT_DIR / "pit_symbols.json").read_text(encoding="utf-8"))
    candidates, settled = meta["candidates"], meta["settled_aliases"]
    PIT_KL.mkdir(parents=True, exist_ok=True)
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_dl_klines_one, s, settled): s for s in candidates}
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = {"symbol": futs[fut], "status": f"ERROR {e}"}
            results.append(r)
            if done % 25 == 0 or r["status"].startswith("ERROR"):
                print(f"[{done}/{len(candidates)}] {r['symbol']}: {r['status']}",
                      flush=True)
    (PIT_DIR / "klines_manifest.json").write_text(
        json.dumps({"run_utc": datetime.now(timezone.utc).isoformat(),
                    "results": results}, indent=1), encoding="utf-8")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_pre = sum(1 for r in results if r["status"] == "pre_window")
    print(f"klines done: ok={n_ok} pre_window={n_pre} "
          f"other={len(results) - n_ok - n_pre}", flush=True)


# ---------------------------------------------------------------- metrics ----
def load_pit_klines(sym: str) -> pd.DataFrame | None:
    p = PIT_KL / f"{sym}.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p)
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    return k.set_index("ts").sort_index()


def gate_days(k: pd.DataFrame) -> list[str]:
    """Days whose 30d-median daily quote volume passes $1M (research math)."""
    dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
    days = dvol30[dvol30 > 1e6].index
    expanded: set[str] = set()
    for d in days:
        for off in range(-2, 3):
            expanded.add((d + pd.Timedelta(days=off)).strftime("%Y-%m-%d"))
    return sorted(d for d in expanded if MET_DAY_LO <= d <= MET_DAY_HI)


def _dl_metrics_one(sym: str, settled: dict[str, str],
                    state: dict) -> dict:
    k = load_pit_klines(sym)
    if k is None:
        return {"symbol": sym, "status": "no_klines"}
    days = gate_days(k)
    if not days:
        return {"symbol": sym, "status": "never_gated"}
    seen = set(state.get(sym, {}).get("done", []))
    out_path = PIT_MET / f"{sym}.parquet"
    frames = [pd.read_parquet(out_path)] if out_path.exists() else []
    if frames and not seen:
        # restart without manifest state: seed done-days from the parquet
        # (a day zip is atomic; 404-days are simply retried once, cheap)
        ts = pd.to_datetime(frames[0]["ts_ms"], unit="ms", utc=True)
        seen = set(ts.strftime("%Y-%m-%d").unique())
    todo = [d for d in days if d not in seen]
    folders = [sym] + ([settled[sym]] if sym in settled else [])
    n404 = 0
    for day in todo:
        blob = None
        for f in folders:
            blob = _get(f"{VISION}/data/futures/um/daily/metrics/{f}/"
                        f"{f}-metrics-{day}.zip")
            if blob is not None:
                break
        if blob is None:
            n404 += 1
        else:
            frames.append(_parse_metrics_zip(blob))
        seen.add(day)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df["ts_ms"] = df["ts_ms"].astype("int64")
        df = (df.drop_duplicates("ts_ms").sort_values("ts_ms")
              .reset_index(drop=True))
        df.to_parquet(out_path, index=False)
        rows = int(len(df))
    else:
        rows = 0
    return {"symbol": sym, "status": "ok" if rows else "no_metrics",
            "gate_days": len(days), "fetched_new": len(todo) - n404,
            "missing_404": n404, "rows": rows, "done": sorted(seen)}


def cmd_download_metrics(workers: int) -> None:
    meta = json.loads((PIT_DIR / "pit_symbols.json").read_text(encoding="utf-8"))
    settled = meta["settled_aliases"]
    PIT_MET.mkdir(parents=True, exist_ok=True)
    state_path = PIT_DIR / "metrics_manifest.json"
    state = (json.loads(state_path.read_text(encoding="utf-8"))
             if state_path.exists() else {})
    syms = sorted(p.stem for p in PIT_KL.glob("*.parquet"))
    results, done, lock = {}, 0, threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_dl_metrics_one, s, settled, state): s for s in syms}
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = {"symbol": futs[fut], "status": f"ERROR {e}"}
            with lock:
                results[r["symbol"]] = {k: v for k, v in r.items() if k != "symbol"}
                if done % 20 == 0:
                    state_path.write_text(json.dumps(results, indent=0),
                                          encoding="utf-8")
                    print(f"[{done}/{len(syms)}] {r['symbol']}: {r['status']} "
                          f"(new {r.get('fetched_new', 0)})", flush=True)
    # merge with prior state (resume safety)
    for s, v in state.items():
        if s not in results:
            results[s] = v
    state_path.write_text(json.dumps(results, indent=0), encoding="utf-8")
    n_ok = sum(1 for v in results.values() if v.get("status") == "ok")
    print(f"metrics done: ok={n_ok} / {len(results)}", flush=True)


# --------------------------------------------------------------- detection ---
def detect_pit(sym: str) -> pd.DataFrame:
    """Frozen detector math on PIT-extra futures 1h klines + pit metrics.

    Identical to liqrev_v2.detect_events except: (1) kline/OI paths, (2) the
    declared tail rule — events within 25 bars of data end are RETAINED (with
    tail_truncated=True) iff the symbol's data ends before DELISTED_CUT.
    """
    kp, op = PIT_KL / f"{sym}.parquet", PIT_MET / f"{sym}.parquet"
    if not kp.exists() or not op.exists():
        return pd.DataFrame()
    k = load_pit_klines(sym)
    oi = pd.read_parquet(op, columns=["ts_ms", "sum_open_interest"])
    oi_s = pd.Series(oi["sum_open_interest"].to_numpy(),
                     index=pd.to_datetime(oi["ts_ms"], unit="ms", utc=True))
    oi_s = oi_s[~oi_s.index.duplicated(keep="last")]
    oi_h = oi_s.resample("1h").last().reindex(k.index).ffill()
    ret6 = k["close"].pct_change(6)
    doi6 = oi_h.pct_change(6)
    dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
    liq_ok = dvol30.reindex(k.index, method="ffill") > 1e6
    mask = ((ret6 <= -0.08) & (doi6 <= -0.10) & liq_ok).fillna(False)

    delisted = k.index[-1] < DELISTED_CUT
    rows, last_t = [], None
    for t in k.index[mask]:
        if not (SPAN_START <= t < SPAN_END):
            continue
        if last_t is not None and (t - last_t) < pd.Timedelta("24h"):
            continue
        i = k.index.get_loc(t)
        tail = i + HOLD_BARS + 1 >= len(k)
        if tail and not delisted:
            continue  # frozen rule for alive symbols
        if i + 1 >= len(k):
            continue  # no entry bar exists at all
        last_t = t
        rows.append({"symbol": sym, "ts": t, "i": i, "ret6": float(ret6.loc[t]),
                     "trig_low": float(k["low"].iloc[i]),
                     "trig_close": float(k["close"].iloc[i]),
                     "tail_truncated": bool(tail)})
    return pd.DataFrame(rows)


def simulate_deploy(ev: pd.DataFrame, kcache: dict[str, pd.DataFrame],
                    rt_cost: float) -> pd.DataFrame:
    """Frozen deploy convention: maker@trig_close TTL 1h, stop -20% from
    entry (gap-aware), exit close of bar i+24; forced close at last bar if
    data ends mid-hold (only possible for tail_truncated events)."""
    out = []
    for _, r in ev.iterrows():
        k = kcache[r["symbol"]]
        i = int(r["i"])
        bar1_low = float(k["low"].iloc[i + 1])
        limit = float(r["trig_close"])
        if not bar1_low < limit:
            out.append({"symbol": r["symbol"], "ts": r["ts"], "filled": False,
                        "is_new": r.get("is_new", False)})
            continue
        entry = limit
        stop = entry * (1 - STOP_PCT)
        last_bar = min(i + 1 + HOLD_BARS - 1, len(k) - 1)
        exit_px, stopped, forced = None, False, False
        for j in range(i + 1, last_bar + 1):
            o, lo = float(k["open"].iloc[j]), float(k["low"].iloc[j])
            if o <= stop or lo <= stop:
                exit_px, stopped = min(o, stop), True
                break
        if exit_px is None:
            exit_px = float(k["close"].iloc[last_bar])
            forced = last_bar < i + 1 + HOLD_BARS - 1
        ret = exit_px / entry - 1.0 - rt_cost
        out.append({"symbol": r["symbol"], "ts": r["ts"], "filled": True,
                    "ret": float(ret), "stopped": stopped,
                    "forced_close": forced,
                    "is_new": r.get("is_new", False)})
    return pd.DataFrame(out)


# --------------------------------------------------------------- portfolio ---
def load_overlay() -> tuple[np.ndarray, int]:
    d = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
    return np.asarray(d["scores_sorted"], dtype=float), int(d["n"])


def btc_ret6_series() -> pd.Series:
    k = pd.read_parquet(KL_DIR / "BTCUSDT.parquet", columns=["open_time", "close"])
    idx = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    c = pd.Series(k["close"].to_numpy(), index=idx).sort_index()
    return c.pct_change(6)


def overlay_weight(ts: pd.Timestamp, btc6: pd.Series,
                   scores: np.ndarray, n: int) -> float:
    v = btc6.get(ts, np.nan)
    if not np.isfinite(v):
        return 1.0  # declared fallback (counted by caller)
    p = float(np.searchsorted(scores, -v, side="right")) / n
    return min(2.0, 2.0 * p)


def run_portfolio(tr: pd.DataFrame, weights: pd.Series,
                  start: pd.Timestamp | None = None) -> dict:
    t = tr[tr["filled"]].sort_values("ts")
    if start is not None:
        t = t[t["ts"] >= start]
    if t.empty:
        return {"n_taken": 0}
    eq, busy, curve, n_taken = 1.0, [], [], 0
    for _, r in t.iterrows():
        busy = [b for b in busy if b > r["ts"]]
        if len(busy) < SLOTS:
            w = float(weights.loc[r.name])
            eq *= (1 + w * r["ret"] / SLOTS)
            busy.append(r["ts"] + pd.Timedelta("24h"))
            n_taken += 1
        curve.append((r["ts"], eq))
    c = pd.Series(dict(curve))
    years = max((c.index[-1] - c.index[0]).days / 365.25, 1e-9)
    m = c.resample("MS").last().ffill().pct_change().dropna()
    yearly = c.groupby(c.index.year).last() / c.groupby(c.index.year).first() - 1
    return {"total": round(float(eq - 1), 4),
            "cagr": round(float(eq ** (1 / years) - 1), 4),
            "maxDD": round(float((c / c.cummax() - 1).min()), 4),
            "worst_month": round(float(m.min()), 4) if len(m) else None,
            "n_taken": n_taken,
            "by_year": {str(y): round(float(v), 3) for y, v in yearly.items()}}


# ----------------------------------------------------------------- analyze ---
def _ev_stats(f: pd.DataFrame) -> dict:
    if f.empty:
        return {"n": 0}
    return {"n": int(len(f)),
            "net_mean": round(float(f["ret"].mean()), 4),
            "net_median": round(float(f["ret"].median()), 4),
            "win": round(float((f["ret"] > 0).mean()), 3),
            "stop_rate": round(float(f["stopped"].mean()), 3),
            "n_forced": int(f["forced_close"].sum())}


def cmd_analyze() -> None:
    meta = json.loads((PIT_DIR / "pit_symbols.json").read_text(encoding="utf-8"))
    klman = json.loads((PIT_DIR / "klines_manifest.json").read_text(encoding="utf-8"))
    metman = json.loads((PIT_DIR / "metrics_manifest.json").read_text(encoding="utf-8"))

    # ---- data-blocked clauses -------------------------------------------
    kl_res = klman["results"]
    in_window = [r for r in kl_res if r["status"] in ("ok", "no_data")]
    n_kl_ok = sum(1 for r in kl_res if r["status"] == "ok")
    d1_frac = n_kl_ok / max(len(in_window), 1)
    gated = {s: v for s, v in metman.items()
             if v.get("status") in ("ok", "no_metrics")}
    n_met_ok = sum(1 for v in gated.values() if v.get("status") == "ok")
    d2_frac = n_met_ok / max(len(gated), 1)
    data_blocked = d1_frac < 0.80 or d2_frac < 0.70

    # ---- PIT universe size per year (task 3) ----------------------------
    pit_syms = sorted(p.stem for p in PIT_KL.glob("*.parquet"))
    year_hours: dict[int, float] = {}
    year_ever: dict[int, set] = {}
    hours_per_year: dict[int, int] = {}

    def gate_hours(k: pd.DataFrame, sym: str, spot: bool) -> None:
        vol = k["quote_volume"]
        dvol30 = vol.resample("1D").sum().rolling(30).median()
        liq = dvol30[dvol30 > 1e6]
        for d in liq.index:
            if not (SPAN_START <= d < SPAN_END):
                continue
            y = d.year
            year_hours[y] = year_hours.get(y, 0) + 24
            year_ever.setdefault(y, set()).add(sym)

    survivors = [c.pair for c in load_universe()]
    for pair in survivors:
        p = KL_DIR / f"{pair}.parquet"
        if not p.exists():
            continue
        k = pd.read_parquet(p, columns=["open_time", "quote_volume"])
        k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
        gate_hours(k.set_index("ts").sort_index(), pair, True)
    surv_year_ever = {y: len(s) for y, s in year_ever.items()}
    surv_year_hours = dict(year_hours)
    year_hours, year_ever = {}, {}
    for sym in pit_syms:
        gate_hours(load_pit_klines(sym), sym, False)
    for y in range(2022, 2027):
        hours_per_year[y] = int((min(pd.Timestamp(f"{y+1}-01-01", tz="UTC"), SPAN_END)
                                 - max(pd.Timestamp(f"{y}-01-01", tz="UTC"),
                                       SPAN_START)).total_seconds() // 3600)
    universe_table = {}
    for y in sorted(hours_per_year):
        if hours_per_year[y] <= 0:
            continue
        universe_table[str(y)] = {
            "survivors_ever_gated": surv_year_ever.get(y, 0),
            "survivors_avg_in_universe":
                round(surv_year_hours.get(y, 0) / hours_per_year[y], 1),
            "new_ever_gated": len(year_ever.get(y, set())),
            "new_avg_in_universe":
                round(year_hours.get(y, 0) / hours_per_year[y], 1)}

    # ---- OI coverage honesty --------------------------------------------
    cov = {"n_gate_qualified_new": len(gated),
           "n_with_metrics": n_met_ok,
           "no_metrics_symbols": [s for s, v in gated.items()
                                  if v.get("status") == "no_metrics"],
           "frac_metric_days_404":
               round(sum(v.get("missing_404", 0) for v in gated.values())
                     / max(sum(v.get("gate_days", 0) for v in gated.values()), 1), 3)}

    # ---- detection (task 4) ---------------------------------------------
    print("detecting survivors (parity)...", flush=True)
    surv_parts, kcache = [], {}
    for idx, pair in enumerate(survivors, 1):
        e = detect_events(pair)
        if len(e):
            e = e[(e["ts"] >= SPAN_START) & (e["ts"] < SPAN_END)]
        if len(e):
            e = e.copy()
            e["tail_truncated"] = False
            e["is_new"] = False
            surv_parts.append(e)
            k = pd.read_parquet(KL_DIR / f"{pair}.parquet",
                                columns=["open_time", "open", "high", "low", "close"])
            k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
            kcache[pair] = k.set_index("ts").sort_index()
        if idx % 40 == 0:
            print(f"  [{idx}/{len(survivors)}]", flush=True)
    surv_ev = pd.concat(surv_parts, ignore_index=True)

    canon = pd.read_parquet(ART_DIR / "ml_dataset.parquet", columns=["symbol", "ts"])
    canon_keys = set(zip(canon["symbol"], canon["ts"]))
    run_keys = set(zip(surv_ev["symbol"], surv_ev["ts"]))
    parity = {"n_canonical": len(canon_keys), "n_this_run": len(run_keys),
              "n_match": len(canon_keys & run_keys),
              "only_this_run": sorted(str(x) for x in (run_keys - canon_keys))[:20],
              "only_canonical": sorted(str(x) for x in (canon_keys - run_keys))[:20]}
    parity_ok = abs(len(run_keys) - 1308) <= 26

    print("detecting PIT-extra...", flush=True)
    new_parts = []
    for idx, sym in enumerate(pit_syms, 1):
        e = detect_pit(sym)
        if len(e):
            e["is_new"] = True
            new_parts.append(e)
            kk = load_pit_klines(sym)
            kcache[sym] = kk[["open", "high", "low", "close"]]
        if idx % 60 == 0:
            print(f"  [{idx}/{len(pit_syms)}]", flush=True)
    new_ev = (pd.concat(new_parts, ignore_index=True)
              if new_parts else pd.DataFrame(columns=surv_ev.columns))
    by_year_new = ({str(y): int(n) for y, n in
                    new_ev.groupby(new_ev["ts"].dt.year).size().items()}
                   if len(new_ev) else {})
    by_sym_new = ({s: int(n) for s, n in new_ev.groupby("symbol").size()
                   .sort_values(ascending=False).head(30).items()}
                  if len(new_ev) else {})

    # ---- per-event economics (task 5) -----------------------------------
    cells = {}
    trades_by_cost = {}
    for cost, tag in [(0.0025, "25bps"), (0.0010, "10bps")]:
        tr_s = simulate_deploy(surv_ev, kcache, cost)
        tr_n = (simulate_deploy(new_ev, kcache, cost)
                if len(new_ev) else pd.DataFrame(
                    columns=["symbol", "ts", "filled", "ret", "stopped",
                             "forced_close", "is_new"]))
        trades_by_cost[tag] = (tr_s, tr_n)
        cell = {}
        for name, tr in [("survivor", tr_s), ("new", tr_n)]:
            f = tr[tr["filled"]] if len(tr) else tr
            n_ev = int(len(tr))
            sub = {"n_events": n_ev,
                   "fill_rate": round(len(f) / n_ev, 3) if n_ev else None,
                   "pooled": _ev_stats(f)}
            for w, lab in [(f[f["ts"] < LIVE_START] if len(f) else f, "DEV"),
                           (f[f["ts"] >= LIVE_START] if len(f) else f, "LIVE")]:
                sub[lab] = _ev_stats(w)
            if name == "new" and len(f):
                for h in (0.05, 0.10):
                    fh = f.copy()
                    fh.loc[fh["forced_close"], "ret"] -= h
                    sub[f"haircut_{int(h*100)}pct"] = _ev_stats(fh)
            cell[name] = sub
        cells[tag] = cell

    # day clustering honesty
    if len(new_ev):
        nd = new_ev["ts"].dt.floor("D")
        sd = set(surv_ev["ts"].dt.floor("D"))
        clustering = {"n_new_events": int(len(new_ev)),
                      "n_distinct_days": int(nd.nunique()),
                      "share_days_overlapping_survivor_events":
                          round(float(nd.isin(sd).mean()), 3)}
    else:
        clustering = {"n_new_events": 0}

    # ---- portfolio (task 6): overlay ON, 10bps --------------------------
    scores, n_sc = load_overlay()
    btc6 = btc_ret6_series()
    tr_s, tr_n = trades_by_cost["10bps"]
    all_tr = pd.concat([tr_s, tr_n], ignore_index=True).sort_values("ts")
    all_tr = all_tr.reset_index(drop=True)
    w_all = pd.Series([overlay_weight(t, btc6, scores, n_sc)
                       for t in all_tr["ts"]], index=all_tr.index)
    n_w_fallback = int(sum(1 for t in all_tr["ts"]
                           if not np.isfinite(btc6.get(t, np.nan))))
    surv_mask = ~all_tr["is_new"].astype(bool)
    port = {}
    for hlab, h in [("haircut0", 0.0), ("haircut10", 0.10)]:
        tr_h = all_tr.copy()
        if h:
            m = tr_h["filled"].fillna(False) & tr_h["forced_close"].fillna(False)
            tr_h.loc[m, "ret"] -= h
        port[hlab] = {
            "survivors_full": run_portfolio(tr_h[surv_mask], w_all),
            "pit_full": run_portfolio(tr_h, w_all),
            "survivors_live": run_portfolio(tr_h[surv_mask], w_all, LIVE_START),
            "pit_live": run_portfolio(tr_h, w_all, LIVE_START)}

    # ---- verdict ---------------------------------------------------------
    new_f25 = trades_by_cost["25bps"][1]
    new_f25 = new_f25[new_f25["filled"]] if len(new_f25) else new_f25
    a_val = float(new_f25["ret"].mean()) if len(new_f25) else None
    a_pass = a_val is not None and a_val > -0.01
    base_cagr = port["haircut0"]["survivors_full"].get("cagr", 0.0)
    pit_cagr = port["haircut0"]["pit_full"].get("cagr", 0.0)
    b_pass = pit_cagr >= 0.60 * base_cagr
    c_val = port["haircut0"]["pit_live"].get("total", None)
    c_pass = c_val is not None and c_val > 0
    verdict = ("DATA-BLOCKED" if data_blocked
               else ("PASS" if (a_pass and b_pass and c_pass) else "FAIL"))

    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "frozen in liqrev_pit_validation.py docstring (Q23)",
        "universe_accounting": {
            "s3_folders_total": meta["n_folders_total"],
            "usdt_nondated": meta["n_usdt_nondated"],
            "clean_after_exclusions": meta["n_clean"],
            "survivor_covered": meta["n_survivor_covered"],
            "pit_candidates": meta["n_candidates"],
            "klines_ok": n_kl_ok,
            "pre_window_dead": sum(1 for r in kl_res
                                   if r["status"] == "pre_window"),
            "kline_no_data": sum(1 for r in kl_res if r["status"] == "no_data"),
            "d1_frac_klines": round(d1_frac, 3),
            "d2_frac_metrics": round(d2_frac, 3),
            "delisted_spot_check": meta["delisted_spot_check"]},
        "pit_universe_by_year": universe_table,
        "oi_coverage": cov,
        "detector_parity": {**parity, "parity_ok": bool(parity_ok)},
        "new_events": {"total": int(len(new_ev)),
                       "tail_truncated": int(new_ev["tail_truncated"].sum())
                       if len(new_ev) else 0,
                       "by_year": by_year_new, "by_symbol_top30": by_sym_new},
        "per_event_cells": cells,
        "day_clustering": clustering,
        "portfolio_overlay_10bps": port,
        "overlay_weight_fallbacks": n_w_fallback,
        "verdict": {
            "result": verdict,
            "a_new_net_mean_25bps": None if a_val is None else round(a_val, 4),
            "a_threshold": -0.01, "a_pass": bool(a_pass),
            "b_pit_cagr": pit_cagr, "b_base_cagr": base_cagr,
            "b_ratio": round(pit_cagr / base_cagr, 3) if base_cagr else None,
            "b_pass": bool(b_pass),
            "c_pit_live_total": c_val, "c_pass": bool(c_pass),
            "parity_ok": bool(parity_ok),
            "data_blocked": bool(data_blocked)}}
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "results_pit.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    print(f"\nartifacts -> {ART_DIR / 'results_pit.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["enumerate", "download-klines",
                                    "download-metrics", "analyze"])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.cmd == "enumerate":
        cmd_enumerate()
    elif args.cmd == "download-klines":
        cmd_download_klines(args.workers)
    elif args.cmd == "download-metrics":
        cmd_download_metrics(args.workers)
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
