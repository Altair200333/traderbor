"""Q25 pre-registered study: is DATED-futures basis (cash-and-carry on Binance
quarterly delivery futures) alive as a harvestable rent in 2025-26, or did it
die with funding carry?

FROZEN SPEC (written before first analysis run, 2026-07-10). Data inventory
probes (S3 symbol enumeration, existing parquet spans) ran before this freeze;
no basis series or P&L was computed before the freeze.

CONTEXT: funding-carry study (docs/notes/2026-07-08/carry-and-liqrev-studies.md,
study 1) found perp funding rent existed only ~2024 (+10.4% climate) and is
NEGATIVE in 2025-26. BIS reports historical dated basis 6-8% ann. with >40%
spikes. Hypothesis: dated basis is compressed by the same institutional
capital, but has different clientele (no funding-path risk, locked
convergence) -- verify, don't assume.

DATA (all free, data.binance.vision, resumable into research/data/basis/):
- UM dated: BTCUSDT_YYMMDD / ETHUSDT_YYMMDD 1h klines, all 24 contracts each
  (210326..261225 per S3 listing 2026-07-10).
- CM dated: BTCUSD_YYMMDD / ETHUSD_YYMMDD 1h klines, 26 contracts each
  (200925..261225); _PERP symbols excluded.
- Spot S: fresh download of BTCUSDT/ETHUSDT SPOT 1h klines 2020-01..present
  (repo's research/data/klines/1h only covers 2024-07+, insufficient).
- Perp reference: research/data/binance_um/klines_1m/{BTCUSDT,ETHUSDT}.parquet
  resampled to 1h closes.
- Funding (calendar variant leg): research/data/perp/funding/{PAIR}.parquet
  (fapi realized funding, coverage starts 2022-07-01 -- calendar-variant
  entries are restricted to ts >= that date; declared here, not tuned).

DESIGN:
1. BASIS SERIES. Expiry = YYMMDD 08:00 UTC (Binance delivery settlement).
   Bar close time = open_time + 1h; hours_to_expiry h2e = (expiry - close
   time)/1h; rows with h2e <= 0 dropped. For every contract-hour:
   ann_basis_spot = (F/S_spot - 1) * (8760/h2e);
   ann_basis_perp = (F/S_perp - 1) * (8760/h2e)  [robustness S].
   CM contracts use the same USDT spot as S (USD~USDT caveat).
   Front-quarter splice: at each hour, front = contract with the nearest
   expiry among those with h2e > 168 (roll at 7d to expiry) and a kline at
   that hour. Output hourly panel parquet + daily (last-obs-of-UTC-day) front
   series.
2. CLIMATE TABLE (core answer). Per instrument (UM BTC, UM ETH, CM BTC,
   CM ETH), by YEAR: mean/median of daily front-quarter ann_basis_spot; share
   of days > 5% and > 10% ann. Descriptive callouts: monthly means for
   2023-12..2024-02 (spot ETF) and 2024-10..2024-12 (ETF options/election).
   Robustness: same yearly means on ann_basis_perp (UM only).
3. HARVEST SIM -- UM BTC and UM ETH only. CM is scoped OUT of the sim upfront:
   inverse payoff makes P&L coin-denominated; CM appears in the climate table
   only.
   Variant A (cash-and-carry): long spot + short front quarterly. Enter at the
   close of the first hour where front ann_basis_spot >= T, T in {5%, 8%, 12%}
   (declared grid). Hold the ENTERED contract; exit at the close of its first
   hour with h2e <= 168 (7d-before-expiry; no roll compounding inside a
   trade). One position at a time per cell; after exit, re-entry allowed
   whenever the (possibly new) front qualifies.
   Variant B (perp-vs-quarterly calendar): long perp + short front quarterly,
   signal = front ann_basis_perp >= T; funding leg = long perp accrues
   -sum(funding_rate) over settlements in (entry, exit].
   COSTS (declared): spot 10bps RT; futures/perp taker 5bps per side (10bps RT
   per leg); slippage 5bps per fill x 4 fills = 20bps. Total 40bps RT on
   notional for both variants.
   CAPITAL: full collateralization, no leverage: capital = 2 x notional (spot
   or perp leg fully paid + short-futures margin at 1x). Net returns are
   reported on this capital; per-notional APR also shown descriptively.
   Trade P&L on notional: A: (S_x/S_e - 1) - (F_x/F_e - 1) - 0.0040;
   B: same on perp + funding accrual - 0.0040.
4. REPORTING per cell (instrument x variant x T), DEV (< 2025-01-01) and LIVE
   (>= 2025-01-01) separately and by-year; trades assigned to periods by ENTRY
   time. Per cell: n trades, mean net APR while deployed (= total net P&L on
   capital / deployed years), time-deployed share (hourly in-trade mask over
   period hours), portfolio APR (= total net P&L on capital / period calendar
   years; idle cash earns 0), maxDD on hourly MTM equity (price-spread MTM;
   funding credited at exit -- declared simplification), worst per-trade
   spread excursion min_t[(S_t/S_e-1)-(F_t/F_e-1)], and count of excursions
   beyond -10% on notional (margin-stress flag; at 1x nothing liquidates).
5. PRE-REGISTERED VERDICT. DEV selection: the cell with max DEV portfolio APR
   among cells with >= 3 DEV trades (if none, >= 1; declared fallback). Dated
   basis is ALIVE iff that cell's LIVE portfolio APR >= +5% net AND LIVE
   time-deployed >= 15%. COMPRESSED if LIVE portfolio APR > 0 but either bar
   missed. DEAD if LIVE portfolio APR <= 0 or LIVE time-deployed < 2%.
   All 12 cells reported regardless.
6. OUTPUTS: research/data/basis/basis_panel_1h.parquet,
   research/data/basis/front_daily.parquet,
   research/data/basis/results_dated_basis.json (printed).

HONESTY / CAVEATS (frozen): convergence at expiry is guaranteed, the PATH is
not -- worst MTM spread excursion quantified (BIS margin-risk point).
Binance-only venue (Bybit/OKX delivery existence checked by API one-liner, no
download). Collateral yield assumed 0% -- understates carry vs T-bill-parked
benchmarks. CM inverse payoff scoped out of the sim. USDT spot proxies USD
for CM basis. Funding leg coverage starts 2022-07. Klines are last-trade
closes, not executable quotes; thin dated books mean real spreads can exceed
the declared 5bps slippage, especially pre-2022.

Usage:
  python dated_basis_study.py download   # enumerate + fetch dated/spot 1h
  python dated_basis_study.py build      # basis panel + front series
  python dated_basis_study.py sim        # climate + harvest sim + verdict
  python dated_basis_study.py all
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "research" / "data" / "basis"
DATED_DIR = OUT_DIR / "klines_1h"
SPOT_DIR = OUT_DIR / "spot_1h"
PERP_1M = REPO_ROOT / "research" / "data" / "binance_um" / "klines_1m"
FUNDING_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION = "https://data.binance.vision"

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
        "ignore"]
KEEP = ["open_time", "close"]

ASSETS = ["BTC", "ETH"]
BOOKS = {"um": "USDT", "cm": "USD"}  # book -> symbol quote piece
DEV_END_MS = int(pd.Timestamp("2025-01-01", tz="UTC").value // 10**6)
THRESHOLDS = [0.05, 0.08, 0.12]
ROLL_H = 168
COST_RT = 0.0040          # 40bps RT on notional, both variants
CAPITAL_MULT = 2.0        # capital = 2 x notional (full collateralization)
HOURS_YEAR = 8760
EXPIRY_HOUR_UTC = 8
ALIVE_APR = 0.05
ALIVE_DEPLOY = 0.15
DEAD_DEPLOY = 0.02


# ---------------------------------------------------------------- download --

def _get(session: requests.Session, url: str, retries: int = 4) -> bytes | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


def s3_list_keys(session: requests.Session, prefix: str) -> list[str]:
    """All keys under prefix (no delimiter -> full key listing, paginated)."""
    keys, marker = [], None
    while True:
        url = f"{S3}?prefix={prefix}"
        if marker:
            url += f"&marker={marker}"
        txt = session.get(url, timeout=30).text
        page = re.findall(r"<Key>([^<]+)</Key>", txt)
        keys += page
        if "<IsTruncated>true</IsTruncated>" in txt and page:
            marker = page[-1]
        else:
            return keys


def s3_list_prefixes(session: requests.Session, prefix: str) -> list[str]:
    out, marker = [], None
    while True:
        url = f"{S3}?delimiter=/&prefix={prefix}"
        if marker:
            url += f"&marker={marker}"
        txt = session.get(url, timeout=30).text
        page = re.findall(r"<Prefix>([^<]+)</Prefix>", txt)
        out += [p for p in page if p != prefix]
        if "<IsTruncated>true</IsTruncated>" in txt and page:
            nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", txt)
            marker = nm.group(1) if nm else page[-1]
        else:
            return out


def enumerate_dated(session: requests.Session) -> list[tuple[str, str]]:
    """[(book, symbol)] for all dated BTC/ETH contracts on both books."""
    out = []
    for book, quote in BOOKS.items():
        for asset in ASSETS:
            pfx = f"data/futures/{book}/monthly/klines/{asset}{quote}_"
            for p in s3_list_prefixes(session, pfx):
                sym = p.rstrip("/").split("/")[-1]
                if re.fullmatch(rf"{asset}{quote}_\d{{6}}", sym):
                    out.append((book, sym))
    return sorted(out)


def _read_kline_zip(raw: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(raw))
    inner = zf.read(zf.namelist()[0])
    header = 0 if inner[:9] == b"open_time" else None
    df = pd.read_csv(io.BytesIO(inner), header=header, names=COLS)
    df = df[KEEP].copy()
    ot = df["open_time"].astype("int64")
    df["open_time"] = np.where(ot > 10**14, ot // 1000, ot)  # us -> ms
    return df


def expiry_ts_ms(symbol: str) -> int:
    ymd = symbol.split("_")[1]
    dt = datetime(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]),
                  EXPIRY_HOUR_UTC, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def download_contract(book: str, sym: str) -> str:
    out_path = DATED_DIR / book / f"{sym}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    expired = expiry_ts_ms(sym) + 86_400_000 < now_ms
    last_ts = -1
    old = None
    if out_path.exists():
        old = pd.read_parquet(out_path)
        if len(old):
            last_ts = int(old["open_time"].max())
        if expired:
            return f"{book}/{sym}: complete ({len(old)})"

    session = requests.Session()
    session.headers["User-Agent"] = "traderbor-research/1.0"
    parts = []
    mkeys = s3_list_keys(session, f"data/futures/{book}/monthly/klines/{sym}/1h/")
    mkeys = [k for k in mkeys if k.endswith(".zip")]
    last_month_end = -1
    for k in sorted(mkeys):
        m = re.search(r"-1h-(\d{4})-(\d{2})\.zip$", k)
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        nxt = datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc)
        month_end = int(nxt.timestamp() * 1000)
        last_month_end = max(last_month_end, month_end)
        if month_end <= last_ts:
            continue
        raw = _get(session, f"{VISION}/{k}")
        if raw:
            parts.append(_read_kline_zip(raw))
    if not expired:
        dkeys = s3_list_keys(session, f"data/futures/{book}/daily/klines/{sym}/1h/")
        for k in sorted(k for k in dkeys if k.endswith(".zip")):
            m = re.search(r"-1h-(\d{4}-\d{2}-\d{2})\.zip$", k)
            if not m:
                continue
            d0 = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            day_end = int(d0.timestamp() * 1000) + 86_400_000
            if day_end <= max(last_ts, last_month_end):
                continue
            raw = _get(session, f"{VISION}/{k}")
            if raw:
                parts.append(_read_kline_zip(raw))
    if not parts:
        return f"{book}/{sym}: no new data"
    df = pd.concat([old, *parts]) if old is not None else pd.concat(parts)
    df = (df.drop_duplicates("open_time").sort_values("open_time")
            .reset_index(drop=True))
    df.to_parquet(out_path, index=False)
    return f"{book}/{sym}: {len(df)} rows"


def download_spot(pair: str) -> str:
    out_path = SPOT_DIR / f"{pair}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_ts = -1
    old = None
    if out_path.exists():
        old = pd.read_parquet(out_path)
        if len(old):
            last_ts = int(old["open_time"].max())
    session = requests.Session()
    session.headers["User-Agent"] = "traderbor-research/1.0"
    parts = []
    now = datetime.now(timezone.utc)
    y, m = 2020, 1
    while (y, m) < (now.year, now.month):
        nxt = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
        if int(nxt.timestamp() * 1000) > last_ts:
            url = f"{VISION}/data/spot/monthly/klines/{pair}/1h/{pair}-1h-{y:04d}-{m:02d}.zip"
            raw = _get(session, url)
            if raw:
                parts.append(_read_kline_zip(raw))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    d = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while d.date() < now.date():
        if int(d.timestamp() * 1000) + 86_400_000 > last_ts:
            url = (f"{VISION}/data/spot/daily/klines/{pair}/1h/"
                   f"{pair}-1h-{d.strftime('%Y-%m-%d')}.zip")
            raw = _get(session, url)
            if raw:
                parts.append(_read_kline_zip(raw))
        d += pd.Timedelta(days=1)
    if not parts:
        return f"spot/{pair}: up to date"
    df = pd.concat([old, *parts]) if old is not None else pd.concat(parts)
    df = (df.drop_duplicates("open_time").sort_values("open_time")
            .reset_index(drop=True))
    df.to_parquet(out_path, index=False)
    return f"spot/{pair}: {len(df)} rows"


def cmd_download(workers: int) -> None:
    session = requests.Session()
    contracts = enumerate_dated(session)
    print(f"enumerated {len(contracts)} dated contracts")
    jobs = [("contract", b, s) for b, s in contracts] + \
           [("spot", None, f"{a}USDT") for a in ASSETS]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for kind, book, sym in jobs:
            fn = download_contract if kind == "contract" else download_spot
            args = (book, sym) if kind == "contract" else (sym,)
            futs[ex.submit(fn, *args)] = sym
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print(f"{futs[f]}: FAILED {e}", flush=True)
    inv = [{"book": b, "symbol": s, "expiry_utc":
            pd.Timestamp(expiry_ts_ms(s), unit="ms", tz="UTC").isoformat()}
           for b, s in contracts]
    (OUT_DIR / "contract_inventory.json").write_text(json.dumps(inv, indent=1))


# ------------------------------------------------------------------- build --

def _load_1h_close(path: Path, name: str) -> pd.Series:
    df = pd.read_parquet(path)
    ot = df["open_time"].astype("int64")
    df["open_time"] = np.where(ot > 10**14, ot // 1000, ot)
    s = (df.drop_duplicates("open_time").set_index("open_time")["close"]
           .sort_index())
    s.name = name
    return s


def _perp_1h_close(pair: str) -> pd.Series:
    df = pd.read_parquet(PERP_1M / f"{pair}.parquet",
                         columns=["open_time", "close"])
    ot = df["open_time"].astype("int64")
    df["open_time"] = np.where(ot > 10**14, ot // 1000, ot)
    hour = (df["open_time"] // 3_600_000) * 3_600_000
    s = df.groupby(hour)["close"].last()
    s.name = "perp"
    return s


def cmd_build() -> None:
    inv = json.loads((OUT_DIR / "contract_inventory.json").read_text())
    spot = {a: _load_1h_close(SPOT_DIR / f"{a}USDT.parquet", "spot")
            for a in ASSETS}
    perp = {a: _perp_1h_close(f"{a}USDT") for a in ASSETS}
    rows = []
    for c in inv:
        book, sym = c["book"], c["symbol"]
        asset = "BTC" if sym.startswith("BTC") else "ETH"
        p = DATED_DIR / book / f"{sym}.parquet"
        if not p.exists():
            print(f"missing klines: {book}/{sym}")
            continue
        f = _load_1h_close(p, "F").to_frame()
        f["spot"] = spot[asset].reindex(f.index)
        f["perp"] = perp[asset].reindex(f.index)
        exp = expiry_ts_ms(sym)
        close_time = f.index.to_numpy(dtype="int64") + 3_600_000
        f["h2e"] = (exp - close_time) / 3_600_000.0
        f = f[f["h2e"] > 0].dropna(subset=["spot"])
        f["ann_basis_spot"] = (f["F"] / f["spot"] - 1) * (HOURS_YEAR / f["h2e"])
        f["ann_basis_perp"] = (f["F"] / f["perp"] - 1) * (HOURS_YEAR / f["h2e"])
        f = f.reset_index().rename(columns={"open_time": "ts"})
        f["book"], f["asset"], f["symbol"], f["expiry_ms"] = book, asset, sym, exp
        rows.append(f)
    panel = pd.concat(rows, ignore_index=True)
    # front flag: nearest expiry with h2e > ROLL_H, per (book, asset, ts)
    elig = panel[panel["h2e"] > ROLL_H]
    idx = elig.groupby(["book", "asset", "ts"])["expiry_ms"].idxmin()
    panel["is_front"] = False
    panel.loc[idx, "is_front"] = True
    panel.to_parquet(OUT_DIR / "basis_panel_1h.parquet", index=False)
    front = panel[panel["is_front"]].copy()
    front["date"] = pd.to_datetime(front["ts"], unit="ms").dt.date.astype(str)
    daily = (front.sort_values("ts")
                  .groupby(["book", "asset", "date"]).last().reset_index())
    daily.to_parquet(OUT_DIR / "front_daily.parquet", index=False)
    print(f"panel rows={len(panel)} front rows={len(front)} daily rows={len(daily)}")


# --------------------------------------------------------------------- sim --

def _load_funding(pair: str) -> pd.Series:
    df = pd.read_parquet(FUNDING_DIR / f"{pair}.parquet")
    ts = df["fundingTime"].astype("int64")
    s = pd.Series(df["fundingRate"].to_numpy(), index=ts.to_numpy()).sort_index()
    return s


def climate_tables() -> dict:
    daily = pd.read_parquet(OUT_DIR / "front_daily.parquet")
    daily["year"] = daily["date"].str[:4]
    daily["month"] = daily["date"].str[:7]
    out = {}
    for (book, asset), g in daily.groupby(["book", "asset"]):
        key = f"{book.upper()} {asset}"
        by_year = {}
        for y, gy in g.groupby("year"):
            b = gy["ann_basis_spot"]
            by_year[y] = {
                "n_days": int(len(b)),
                "mean": round(float(b.mean()), 4),
                "median": round(float(b.median()), 4),
                "share_gt_5pct": round(float((b > 0.05).mean()), 3),
                "share_gt_10pct": round(float((b > 0.10).mean()), 3),
            }
        callouts = {m: round(float(g.loc[g["month"] == m, "ann_basis_spot"].mean()), 4)
                    for m in ["2023-12", "2024-01", "2024-02",
                              "2024-10", "2024-11", "2024-12"]
                    if (g["month"] == m).any()}
        entry = {"by_year": by_year, "etf_callouts_monthly_mean": callouts}
        if book == "um":
            entry["robustness_perp_S_yearly_mean"] = {
                y: round(float(gy["ann_basis_perp"].mean()), 4)
                for y, gy in g.groupby("year")}
        out[key] = entry
    return out


def _period_of(ts_ms: int) -> str:
    return "DEV" if ts_ms < DEV_END_MS else "LIVE"


def simulate_cell(front: pd.DataFrame, paths: dict[str, pd.DataFrame],
                  variant: str, thr: float,
                  funding: pd.Series | None) -> list[dict]:
    scol = "spot" if variant == "carry" else "perp"
    bcol = "ann_basis_spot" if variant == "carry" else "ann_basis_perp"
    fund_start = int(funding.index.min()) if funding is not None else None
    ts_arr = front["ts"].to_numpy()
    trades = []
    i = 0
    n = len(front)
    while i < n:
        row = front.iloc[i]
        ok = row[bcol] >= thr and np.isfinite(row[scol])
        if variant == "calendar" and (fund_start is None or row["ts"] < fund_start):
            ok = False
        if not ok:
            i += 1
            continue
        sym, ts_e, F_e, S_e = row["symbol"], int(row["ts"]), row["F"], row[scol]
        path = paths[sym]
        held = path[path["ts"] > ts_e]
        if held.empty:
            i += 1
            continue
        past_roll = held[held["h2e"] <= ROLL_H]
        x = past_roll.iloc[0] if len(past_roll) else held.iloc[-1]
        held = held[held["ts"] <= int(x["ts"])]
        spread = (held[scol] / S_e - 1) - (held["F"] / F_e - 1)
        pnl_price = float(spread.iloc[-1])
        fund_acc = 0.0
        if variant == "calendar":
            mask = (funding.index > ts_e) & (funding.index <= int(x["ts"]))
            fund_acc = -float(funding[mask].sum())
        hours = (int(x["ts"]) - ts_e) / 3_600_000.0
        trades.append({
            "symbol": sym, "entry_ts": ts_e, "exit_ts": int(x["ts"]),
            "entry_basis": round(float(row[bcol]), 4), "hours": hours,
            "pnl_price": pnl_price, "funding": fund_acc,
            "pnl_net": pnl_price + fund_acc - COST_RT,
            "worst_excursion": float(min(spread.min(), 0.0)),
            "spread_path": spread.to_numpy(), "ts_path": held["ts"].to_numpy(),
        })
        i = int(np.searchsorted(ts_arr, int(x["ts"]), side="right"))
    return trades


def cell_metrics(trades: list[dict], front: pd.DataFrame) -> dict:
    ts = front["ts"].to_numpy()
    in_trade = np.zeros(len(ts), dtype=bool)
    equity = np.zeros(len(ts))
    realized = 0.0
    for t in trades:
        m = (ts > t["entry_ts"]) & (ts <= t["exit_ts"])
        in_trade |= m
        sp = pd.Series(t["spread_path"], index=t["ts_path"]).reindex(ts[m])
        sp = sp.ffill().fillna(0.0).to_numpy()
        equity[m] = realized + sp / CAPITAL_MULT
        realized += t["pnl_net"] / CAPITAL_MULT
        equity[ts > t["exit_ts"]] = realized
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    maxdd = float((equity - peak).min()) if len(equity) else 0.0

    out = {}
    years_all = pd.to_datetime(ts, unit="ms").year
    for period in ["DEV", "LIVE"] + sorted(set(str(y) for y in years_all)):
        if period == "DEV":
            pmask = ts < DEV_END_MS
        elif period == "LIVE":
            pmask = ts >= DEV_END_MS
        else:
            pmask = years_all == int(period)
        ph = int(pmask.sum())
        if ph == 0:
            continue
        if period in ("DEV", "LIVE"):
            tr = [t for t in trades if _period_of(t["entry_ts"]) == period]
        else:
            tr = [t for t in trades
                  if pd.Timestamp(t["entry_ts"], unit="ms").year == int(period)]
        pnl_cap = sum(t["pnl_net"] for t in tr) / CAPITAL_MULT
        dep_h = float(in_trade[pmask].sum())
        out[period] = {
            "n_trades": len(tr),
            "portfolio_apr": round(pnl_cap / (ph / HOURS_YEAR), 4),
            "apr_while_deployed": round(pnl_cap / (dep_h / HOURS_YEAR), 4)
            if dep_h else None,
            "apr_on_notional": round(pnl_cap * CAPITAL_MULT / (ph / HOURS_YEAR), 4),
            "time_deployed": round(dep_h / ph, 3),
            "worst_excursion": round(min((t["worst_excursion"] for t in tr),
                                         default=0.0), 4),
            "n_excursion_lt_-10pct": sum(t["worst_excursion"] < -0.10 for t in tr),
        }
    out["maxdd_on_capital"] = round(maxdd, 4)
    return out


def cmd_sim() -> None:
    panel = pd.read_parquet(OUT_DIR / "basis_panel_1h.parquet")
    results = {"spec": "Q25 dated basis study, frozen 2026-07-10",
               "climate": climate_tables(), "cells": {}}
    funding = {a: _load_funding(f"{a}USDT") for a in ASSETS}
    for asset in ASSETS:
        sub = panel[(panel["book"] == "um") & (panel["asset"] == asset)]
        front = sub[sub["is_front"]].sort_values("ts").reset_index(drop=True)
        paths = {s: g.sort_values("ts")[["ts", "F", "spot", "perp", "h2e"]]
                 for s, g in sub.groupby("symbol")}
        for variant in ["carry", "calendar"]:
            for thr in THRESHOLDS:
                trades = simulate_cell(front, paths, variant, thr,
                                       funding[asset])
                key = f"UM {asset} | {variant} | T={int(thr*100)}%"
                results["cells"][key] = cell_metrics(trades, front)
    # DEV selection + verdict (frozen clause)
    def dev_apr(c):
        return c.get("DEV", {}).get("portfolio_apr", -9)
    elig = {k: c for k, c in results["cells"].items()
            if c.get("DEV", {}).get("n_trades", 0) >= 3}
    if not elig:
        elig = {k: c for k, c in results["cells"].items()
                if c.get("DEV", {}).get("n_trades", 0) >= 1}
    sel_key = max(elig, key=lambda k: dev_apr(elig[k])) if elig else None
    verdict = "DEAD (no DEV trades anywhere)"
    if sel_key:
        live = results["cells"][sel_key].get("LIVE", {})
        apr, dep = live.get("portfolio_apr", 0), live.get("time_deployed", 0)
        if apr is None:
            apr = 0.0
        if apr >= ALIVE_APR and dep >= ALIVE_DEPLOY:
            verdict = "ALIVE"
        elif apr > 0 and dep >= DEAD_DEPLOY:
            verdict = "COMPRESSED"
        elif apr > 0:
            verdict = "DEAD (deployment ~never)"
        else:
            verdict = "DEAD"
    results["dev_selected_cell"] = sel_key
    results["verdict"] = verdict
    (OUT_DIR / "results_dated_basis.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "build", "sim", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.stage in ("download", "all"):
        cmd_download(a.workers)
    if a.stage in ("build", "all"):
        cmd_build()
    if a.stage in ("sim", "all"):
        cmd_sim()


if __name__ == "__main__":
    main()
