"""Q27 - SHORT-SQUEEZE CONTINUATION LONG: is buying the up-cascade tradable?

PRE-REGISTERED spec (FROZEN 2026-07-10 BEFORE the first run; lab protocol:
DEV < 2025-01-01 selects, LIVE >= 2025-01-01 verdicts; controls mandatory).

MOTIVATION. The squeeze-fade study (research/scanner_lab/squeeze_fade.py,
docs/notes/2026-07-08 study 1) established that up-cascade events (price
+8%/6h AND OI -10%/6h = shorts force-liquidated) CONTINUE upward on the full
span: raw fwd +2.4% @12h, +2.8% @48h, P(down)@48h only 0.37; fading them lost.
Q27 tests the mirror trade: LONG the continuation.

KNOWN LANDMINE (stated up front, tested explicitly): the fade study's HOLDOUT
showed fading pumps WORKED in 2025-26 (its +1.71%/trade holdout was explained
by the price-only control) -- which implies continuation-long may be DEAD
exactly in the LIVE window. A clean NO is an expected, valuable outcome.

EVENTS (detection REUSED byte-for-byte from squeeze_fade.detect_events,
use_oi=True, single severity T=+8%):
  ret_6h >= +0.08 AND OI_6h_change <= -0.10, 1h grid, full span 2022-07..2026-07,
  liquidity gate: 30d-median of daily quote_volume > $1M,
  24h per-symbol cooldown, guard i+1+48 < n (kept identical to squeeze_fade so
  the event set is byte-identical to the fade study's 390 events).
  Report n, by-year, and overlap with liqrev down-cascade events
  (ret_6h <= -8% AND OI_6h <= -10%, same gates; same symbol same UTC day both
  directions = "chaos day", counted).

DIRECTION = LONG. DECLARED CELLS (2 entries x 3 exits = 6 cells, NO other axis):
  ENTRY (a) "taker": fill at next 1h bar OPEN. Per-side cost 5.5bps fee
            + 5bps slippage = 10.5bps.
        (b) "maker": limit BUY at trigger-bar CLOSE, valid the next 1h bar only
            (TTL 1h), filled IFF next bar LOW < limit (trade-through below =
            fills only on pullback; misses runners). Per-side 2bps + 0 slippage.
            Unfilled = skipped; fill rate reported.
  EXIT  time exit at the CLOSE of bar entry_bar+H-1, H in {4, 12, 24} hours.
        Exit side is always TAKER (10.5bps), stop exits included.
  => rt_cost: taker cells 21.0bps, maker cells 12.5bps.
  STOP  disaster stop at entry*0.85 (-15% from entry; one declared value, no
        grid -- insurance, not tuning). Gap-aware, checked every held bar
        including the entry bar: if open<=stop or low<=stop -> exit at
        min(open, stop) that bar (long convention).
  FUNDING realized (not assumed), long-signed: funding = -sum(fundingRate) at
        settlements strictly inside (entry_ts, exit_ts]; entry_ts = entry bar
        open time; exit_ts = entry_ts + H hours for time exits, stop-bar open
        time for stop exits. Long PAYS positive funding. Fallback 0 when the
        symbol has no funding file; realized-data share reported.
  net = exit/entry - 1 - rt_cost + funding.

CONTROLS (mandatory, same trade mechanics, ALL 6 cells each):
  (i)  price-only: ret_6h >= +8% with NO OI condition (is OI-collapse specific?)
  (ii) random-bars matched by symbol-liquidity: for each event symbol, 5x that
       symbol's event count of random liquidity-passing bars (seed 20260710).

PORTFOLIO (liqrev_v2.portfolio convention, amended per Q27 spec): 10 slots x
equity/10, event-ordered, slot frees at actual exit, PLUS 24h per-symbol
cooldown at the portfolio level (skip if the symbol entered < 24h ago).

SELECTION: DEV-best cell = max DEV per-event net mean among cells with DEV
n_filled >= 30; tie-break DEV portfolio CAGR. LIVE performance of that single
cell is the verdict number.

STATS: day-clustered SE (clusters = UTC event date, CR1 correction
G/(G-1)) on the DEV-selected cell, both windows; 90% CI = mean +/- 1.645*SE.

PRE-REGISTERED VERDICT (all three required for TRADABLE, else NO clause by
clause):
  A. DEV-selected cell on LIVE: net mean >= +50bps/trade AND day-clustered
     90% CI excludes 0 (lower bound > 0).
  B. Beats the price-only control (same cell config, LIVE) by >= +30bps.
  C. Portfolio LIVE maxDD > -15%.

HONESTY DIAGNOSTICS (reported, not selected on):
  - per-trade p5/p1 net and max adverse excursion MAE = min(low over held
    bars)/entry - 1 (mean/p5/p1) -- buying strength has worst-case path risk;
  - POST-HOC LABELED: net split by event stretch above its own rolling 24h
    VWAP (vwap24 = 24-bar rolling sum(quote_volume)/sum(volume); stretch =
    trig_close/vwap24 - 1; split at the filled-trade median) -- late-squeeze
    entries are the known killer;
  - survivors-only 149-coin universe picked in 2026 flatters longs, pre-2024
    inflated -- stated caveat;
  - funding drag per trade and share of trades with realized funding data.

Usage: python squeeze_continuation.py   (from research/scanner_lab)
Artifacts: research/data/squeeze/results_continuation.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402

KL_DIR = REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h"
OI_DIR = REPO_ROOT / "research" / "data" / "perp" / "metrics_5m"
FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
ART_DIR = REPO_ROOT / "research" / "data" / "squeeze"

SLOTS = 10
DETECT_GUARD = 48          # squeeze_fade-identical guard -> identical event set
STOP_PCT = -0.15
TAKER_SIDE = 0.00055 + 0.0005     # 5.5bps fee + 5bps slippage
MAKER_SIDE = 0.0002               # 2bps + 0 slippage
RT_TAKER = 2 * TAKER_SIDE         # 21.0 bps
RT_MAKER = MAKER_SIDE + TAKER_SIDE  # 12.5 bps
HOLDS = [4, 12, 24]
DEV_CUT = pd.Timestamp("2025-01-01", tz="UTC")
SEED = 20260710
RANDOM_MULT = 5
MIN_DEV_N = 30
VERDICT_MIN_LIVE_BPS = 0.0050
VERDICT_MIN_MARGIN_BPS = 0.0030
VERDICT_MAX_DD = -0.15


# ------------------------------------------------------------------ data cache
def load_symbol(pair: str) -> dict | None:
    kp = KL_DIR / f"{pair}.parquet"
    if not kp.exists():
        return None
    k = pd.read_parquet(kp, columns=["open_time", "open", "high", "low",
                                     "close", "volume", "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
    liq_ok = (dvol30.reindex(k.index, method="ffill") > 1e6).to_numpy()
    vwap24 = (k["quote_volume"].rolling(24).sum()
              / k["volume"].rolling(24).sum().replace(0, np.nan))
    return {"index": k.index,
            "open": k["open"].to_numpy(dtype=float),
            "high": k["high"].to_numpy(dtype=float),
            "low": k["low"].to_numpy(dtype=float),
            "close": k["close"].to_numpy(dtype=float),
            "ret6": k["close"].pct_change(6).to_numpy(dtype=float),
            "liq_ok": liq_ok,
            "vwap24": vwap24.to_numpy(dtype=float)}


def load_doi6(pair: str, index: pd.DatetimeIndex) -> np.ndarray | None:
    op = OI_DIR / f"{pair}.parquet"
    if not op.exists():
        return None
    oi = pd.read_parquet(op, columns=["ts_ms", "sum_open_interest"])
    oi_s = pd.Series(oi["sum_open_interest"].to_numpy(),
                     index=pd.to_datetime(oi["ts_ms"], unit="ms", utc=True))
    oi_h = oi_s.resample("1h").last().reindex(index).ffill()
    return oi_h.pct_change(6).to_numpy(dtype=float)


# ------------------------------------------------------------------ detection
def cooldown_events(pair: str, d: dict, mask: np.ndarray,
                    guard: int) -> pd.DataFrame:
    """squeeze_fade-identical: 24h per-symbol cooldown + end-of-data guard."""
    rows, last_t = [], None
    n = len(d["index"])
    for i in np.nonzero(mask)[0]:
        t = d["index"][i]
        if last_t is not None and (t - last_t) < pd.Timedelta("24h"):
            continue
        if i + 1 + guard >= n:
            continue
        last_t = t
        vw = d["vwap24"][i]
        rows.append({"symbol": pair, "ts": t, "i": int(i),
                     "ret6": float(d["ret6"][i]),
                     "trig_close": float(d["close"][i]),
                     "stretch": float(d["close"][i] / vw - 1.0)
                     if np.isfinite(vw) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ funding
def load_funding_map(pairs: list[str]) -> dict:
    """squeeze_fade-identical prefix-cumsum store: O(log n) sums over (a,b]."""
    out = {}
    for p in pairs:
        fp = FUND_DIR / f"{p}.parquet"
        if not fp.exists():
            continue
        f = pd.read_parquet(fp, columns=["fundingTime", "fundingRate"])
        ts = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("1h")
        s = pd.Series(f["fundingRate"].to_numpy(), index=ts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        times = np.asarray(s.index.as_unit("ns").asi8)
        cum = np.concatenate([[0.0], np.cumsum(s.to_numpy(dtype=float))])
        out[p] = (times, cum)
    return out


def funding_long(fund, a: pd.Timestamp, b: pd.Timestamp) -> float:
    """LONG-signed realized funding over settlements a < t <= b (pays +rate)."""
    if fund is None:
        return 0.0
    times, cum = fund
    lo = int(np.searchsorted(times, a.value, side="right"))
    hi = int(np.searchsorted(times, b.value, side="right"))
    return -float(cum[hi] - cum[lo])


# ------------------------------------------------------------------ simulation
def simulate(ev: pd.DataFrame, cache: dict, fundmap: dict,
             entry_mode: str, hold: int, rt_cost: float) -> pd.DataFrame:
    out = []
    for r in ev.itertuples(index=False):
        d = cache[r.symbol]
        i = r.i
        eb = i + 1
        if eb + hold > len(d["index"]):
            continue
        if entry_mode == "taker":
            entry = d["open"][eb]
        else:  # maker: limit BUY at trigger close, next bar only, trade-through
            limit = r.trig_close
            if d["low"][eb] < limit:
                entry = limit
            else:
                out.append({"symbol": r.symbol, "ts": r.ts, "filled": False})
                continue
        stop = entry * (1.0 + STOP_PCT)
        entry_ts = d["index"][eb]
        exit_px, exit_bar, stopped = None, eb + hold - 1, False
        mae_low = np.inf
        for j in range(eb, eb + hold):
            o, lo = d["open"][j], d["low"][j]
            mae_low = min(mae_low, lo)
            if o <= stop or lo <= stop:
                exit_px, exit_bar, stopped = min(o, stop), j, True
                break
        if exit_px is None:
            exit_px = d["close"][eb + hold - 1]
        exit_ts_fund = (d["index"][exit_bar] if stopped
                        else entry_ts + pd.Timedelta(hours=hold))
        gross = exit_px / entry - 1.0                       # LONG pnl
        fund = funding_long(fundmap.get(r.symbol), entry_ts, exit_ts_fund)
        out.append({"symbol": r.symbol, "ts": r.ts, "filled": True,
                    "ret": gross - rt_cost + fund, "gross": gross,
                    "fund": fund, "stopped": stopped,
                    "mae": mae_low / entry - 1.0,
                    "has_fund": r.symbol in fundmap,
                    "stretch": getattr(r, "stretch", np.nan),
                    "exit_ts": d["index"][exit_bar] + pd.Timedelta(hours=1)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ portfolio
def portfolio(tr: pd.DataFrame) -> dict:
    """liqrev_v2-identical slot logic + Q27's 24h per-symbol cooldown."""
    t = tr[tr["filled"]].sort_values("ts")
    eq, busy, curve, n_taken = 1.0, [], [], 0
    last_sym: dict[str, pd.Timestamp] = {}
    for _, r in t.iterrows():
        busy = [b for b in busy if b > r["ts"]]
        cool = last_sym.get(r["symbol"])
        if len(busy) < SLOTS and (cool is None
                                  or r["ts"] - cool >= pd.Timedelta("24h")):
            eq *= (1 + r["ret"] / SLOTS)
            busy.append(r["exit_ts"])
            last_sym[r["symbol"]] = r["ts"]
            n_taken += 1
        curve.append((r["ts"], eq))
    c = pd.Series(dict(curve))
    if c.empty:
        return {}
    years = max((c.index[-1] - c.index[0]).days / 365.25, 1e-9)
    m = c.resample("MS").last().ffill().pct_change().dropna()
    yearly = c.groupby(c.index.year).last() / c.groupby(c.index.year).first() - 1
    return {"total": round(float(eq - 1), 4),
            "cagr": round(float(eq ** (1 / years) - 1), 4),
            "maxDD": round(float((c / c.cummax() - 1).min()), 4),
            "worst_month": round(float(m.min()), 4) if len(m) else None,
            "n_taken": n_taken,
            "by_year": {str(k): round(float(v), 3) for k, v in yearly.items()}}


# ------------------------------------------------------------------ stats
def cluster_se(f: pd.DataFrame) -> tuple[float | None, float | None]:
    """Day-clustered (UTC event date) SE of the mean, CR1 G/(G-1)."""
    if len(f) < 2:
        return None, None
    x = f["ret"].to_numpy(dtype=float)
    days = f["ts"].dt.floor("D")
    resid = x - x.mean()
    s = pd.Series(resid, index=days.to_numpy()).groupby(level=0).sum()
    g = len(s)
    if g < 2:
        return None, None
    se = float(np.sqrt(g / (g - 1) * (s.to_numpy() ** 2).sum()) / len(x))
    return se, g


def _metrics(f: pd.DataFrame) -> dict:
    if not len(f):
        return {"n": 0}
    return {"n": int(len(f)),
            "net_mean": round(float(f["ret"].mean()), 4),
            "net_median": round(float(f["ret"].median()), 4),
            "win": round(float((f["ret"] > 0).mean()), 3),
            "gross_mean": round(float(f["gross"].mean()), 4),
            "fund_mean": round(float(f["fund"].mean()), 5),
            "stop_rate": round(float(f["stopped"].mean()), 3),
            "p5": round(float(f["ret"].quantile(0.05)), 4),
            "p1": round(float(f["ret"].quantile(0.01)), 4),
            "mae_mean": round(float(f["mae"].mean()), 4),
            "mae_p5": round(float(f["mae"].quantile(0.05)), 4),
            "mae_p1": round(float(f["mae"].quantile(0.01)), 4)}


def report(tr: pd.DataFrame, tag: str) -> dict:
    f = tr[tr["filled"]] if len(tr) else tr
    dev = f[f["ts"] < DEV_CUT] if len(f) else f
    live = f[f["ts"] >= DEV_CUT] if len(f) else f
    return {"tag": tag,
            "n_events": int(len(tr)),
            "n_filled": int(len(f)),
            "fill_rate": round(float(len(f) / len(tr)), 3) if len(tr) else None,
            "ALL": _metrics(f),
            "DEV": {**_metrics(dev),
                    "portfolio": portfolio(tr[tr["ts"] < DEV_CUT]) if len(tr) else {}},
            "LIVE": {**_metrics(live),
                     "portfolio": portfolio(tr[tr["ts"] >= DEV_CUT]) if len(tr) else {}}}


def by_year(f: pd.DataFrame) -> dict:
    out = {}
    for y, g in f.groupby(f["ts"].dt.year):
        out[str(y)] = {"n": int(len(g)),
                       "net_mean": round(float(g["ret"].mean()), 4),
                       "win": round(float((g["ret"] > 0).mean()), 3)}
    return out


# ------------------------------------------------------------------ random control
def random_bars(ev: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Symbol-liquidity matched: per event-symbol, RANDOM_MULT x its event
    count of random liquidity-passing bars (same guard), seed fixed."""
    rng = np.random.default_rng(SEED)
    rows = []
    for sym, cnt in ev["symbol"].value_counts().items():
        d = cache[sym]
        n = len(d["index"])
        idx = np.nonzero(d["liq_ok"])[0]
        idx = idx[(idx > 6) & (idx + 1 + DETECT_GUARD < n)]
        if not len(idx):
            continue
        take = min(cnt * RANDOM_MULT, len(idx))
        for i in rng.choice(idx, size=take, replace=False):
            vw = d["vwap24"][i]
            rows.append({"symbol": sym, "ts": d["index"][i], "i": int(i),
                         "ret6": float(d["ret6"][i]),
                         "trig_close": float(d["close"][i]),
                         "stretch": float(d["close"][i] / vw - 1.0)
                         if np.isfinite(vw) else np.nan})
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------------ main
def main() -> None:
    pairs = [c.pair for c in load_universe()]
    print(f"universe: {len(pairs)} pairs", flush=True)
    fundmap = load_funding_map(pairs)
    print(f"funding loaded for {len(fundmap)} symbols", flush=True)

    cache, ev_sq, ev_px, ev_liq = {}, [], [], []
    for n_done, pair in enumerate(pairs, 1):
        d = load_symbol(pair)
        if d is None:
            continue
        cache[pair] = d
        doi6 = load_doi6(pair, d["index"])
        with np.errstate(invalid="ignore"):
            up = (d["ret6"] >= 0.08) & d["liq_ok"]
            up = np.nan_to_num(up, nan=False)
            if doi6 is not None:
                sq = up & np.nan_to_num(doi6 <= -0.10, nan=False)
                dn = (np.nan_to_num(d["ret6"] <= -0.08, nan=False)
                      & d["liq_ok"] & np.nan_to_num(doi6 <= -0.10, nan=False))
            else:
                sq = np.zeros(len(up), dtype=bool)
                dn = np.zeros(len(up), dtype=bool)
        e = cooldown_events(pair, d, sq, DETECT_GUARD)
        if len(e):
            ev_sq.append(e)
        e = cooldown_events(pair, d, up, DETECT_GUARD)
        if len(e):
            ev_px.append(e)
        e = cooldown_events(pair, d, dn, 24)   # liqrev's own guard
        if len(e):
            ev_liq.append(e)
        if n_done % 40 == 0:
            print(f"[{n_done}/{len(pairs)}] sq={sum(len(x) for x in ev_sq)} "
                  f"px={sum(len(x) for x in ev_px)} "
                  f"liqrev={sum(len(x) for x in ev_liq)}", flush=True)

    ev_sq = pd.concat(ev_sq, ignore_index=True).sort_values("ts").reset_index(drop=True)
    ev_px = pd.concat(ev_px, ignore_index=True).sort_values("ts").reset_index(drop=True)
    ev_liq = pd.concat(ev_liq, ignore_index=True).sort_values("ts").reset_index(drop=True)
    rnd = random_bars(ev_sq, cache)
    print(f"squeeze events: {len(ev_sq)}  price-only: {len(ev_px)}  "
          f"liqrev(down): {len(ev_liq)}  random: {len(rnd)}", flush=True)

    # ---- counts, by-year, chaos-day overlap with liqrev -------------------
    ev_years = {str(y): int(c) for y, c in
                ev_sq["ts"].dt.year.value_counts().sort_index().items()}
    sq_days = set(zip(ev_sq["symbol"], ev_sq["ts"].dt.floor("D")))
    lq_days = set(zip(ev_liq["symbol"], ev_liq["ts"].dt.floor("D")))
    chaos = sq_days & lq_days
    chaos_by_year = {}
    for _, day in chaos:
        chaos_by_year[str(day.year)] = chaos_by_year.get(str(day.year), 0) + 1
    n_sq_on_chaos = int(sum(1 for s, t in
                            zip(ev_sq["symbol"], ev_sq["ts"].dt.floor("D"))
                            if (s, t) in chaos))
    overlap = {"n_squeeze_events": int(len(ev_sq)),
               "events_by_year": ev_years,
               "n_liqrev_down_events": int(len(ev_liq)),
               "chaos_symbol_days": len(chaos),
               "squeeze_events_on_chaos_days": n_sq_on_chaos,
               "chaos_share_of_squeeze": round(n_sq_on_chaos / len(ev_sq), 3),
               "chaos_by_year": dict(sorted(chaos_by_year.items()))}
    print("overlap:", json.dumps(overlap), flush=True)

    # ---- 6 cells on events + both controls ---------------------------------
    cells = [(em, h) for em in ["taker", "maker"] for h in HOLDS]
    grid, grid_px, grid_rnd = {}, {}, {}
    for em, h in cells:
        cost = RT_TAKER if em == "taker" else RT_MAKER
        tag = f"{em}_h{h}"
        tr = simulate(ev_sq, cache, fundmap, em, h, cost)
        grid[tag] = {"config": {"entry": em, "hold": h, "rt_cost_bps": cost * 1e4},
                     "report": report(tr, tag), "trades": tr}
        grid_px[tag] = report(simulate(ev_px, cache, fundmap, em, h, cost),
                              f"px_{tag}")
        grid_rnd[tag] = report(simulate(rnd, cache, fundmap, em, h, cost),
                               f"rnd_{tag}")
        r = grid[tag]["report"]
        print(f"  {tag:<10} filled={r['n_filled']:>4} "
              f"DEV={r['DEV'].get('net_mean')} LIVE={r['LIVE'].get('net_mean')} "
              f"| px DEV={grid_px[tag]['DEV'].get('net_mean')} "
              f"LIVE={grid_px[tag]['LIVE'].get('net_mean')} "
              f"| rnd LIVE={grid_rnd[tag]['LIVE'].get('net_mean')}", flush=True)

    # ---- DEV selection ------------------------------------------------------
    def dev_key(item):
        rep = item[1]["report"]["DEV"]
        if rep.get("n", 0) < MIN_DEV_N or rep.get("net_mean") is None:
            return (-1e9, -1e9)
        cagr = (rep.get("portfolio") or {}).get("cagr")
        return (rep["net_mean"], cagr if cagr is not None else -9)
    best_tag = max(grid.items(), key=dev_key)[0]
    best = grid[best_tag]
    tr_best = best["trades"]
    f_best = tr_best[tr_best["filled"]]
    print(f"\nDEV-selected cell: {best_tag}  {best['config']}", flush=True)

    # ---- day-clustered CI on selected cell, both windows --------------------
    stats = {}
    for wname, w in [("DEV", f_best[f_best["ts"] < DEV_CUT]),
                     ("LIVE", f_best[f_best["ts"] >= DEV_CUT])]:
        se, g = cluster_se(w)
        mean = float(w["ret"].mean()) if len(w) else None
        ci = ([round(mean - 1.645 * se, 4), round(mean + 1.645 * se, 4)]
              if se is not None else None)
        stats[wname] = {"n": int(len(w)), "n_day_clusters": g,
                        "net_mean": round(mean, 4) if mean is not None else None,
                        "se_dayclustered": round(se, 4) if se else None,
                        "ci90": ci}

    # ---- verdict clause by clause -------------------------------------------
    live_mean = stats["LIVE"]["net_mean"]
    ci_lo = stats["LIVE"]["ci90"][0] if stats["LIVE"]["ci90"] else None
    px_live = grid_px[best_tag]["LIVE"].get("net_mean")
    margin_live = (round(live_mean - px_live, 4)
                   if live_mean is not None and px_live is not None else None)
    px_dev = grid_px[best_tag]["DEV"].get("net_mean")
    dev_mean = stats["DEV"]["net_mean"]
    margin_dev = (round(dev_mean - px_dev, 4)
                  if dev_mean is not None and px_dev is not None else None)
    live_dd = (best["report"]["LIVE"].get("portfolio") or {}).get("maxDD")
    clause_a = (live_mean is not None and live_mean >= VERDICT_MIN_LIVE_BPS
                and ci_lo is not None and ci_lo > 0)
    clause_b = margin_live is not None and margin_live >= VERDICT_MIN_MARGIN_BPS
    clause_c = live_dd is not None and live_dd > VERDICT_MAX_DD
    verdict = {
        "clause_A_live_mean_ge_50bps_ci90_ex0": {
            "pass": bool(clause_a), "live_net_mean": live_mean,
            "ci90_low": ci_lo, "required": ">= 0.005 and CI90 low > 0"},
        "clause_B_beats_price_only_by_30bps_live": {
            "pass": bool(clause_b), "margin_live": margin_live,
            "price_only_live": px_live, "margin_dev": margin_dev,
            "price_only_dev": px_dev, "required": ">= 0.003"},
        "clause_C_portfolio_live_maxdd": {
            "pass": bool(clause_c), "live_maxDD": live_dd,
            "required": "> -0.15"},
        "TRADABLE": bool(clause_a and clause_b and clause_c)}

    # ---- honesty diagnostics -------------------------------------------------
    med_stretch = float(f_best["stretch"].median())
    lo_s = f_best[f_best["stretch"] <= med_stretch]
    hi_s = f_best[f_best["stretch"] > med_stretch]
    diag = {
        "by_year_selected_cell": by_year(f_best),
        "stretch_split_POSTHOC": {
            "median_stretch_vs_vwap24": round(med_stretch, 4),
            "below_median": {"n": int(len(lo_s)),
                             "net_mean": round(float(lo_s["ret"].mean()), 4)},
            "above_median": {"n": int(len(hi_s)),
                             "net_mean": round(float(hi_s["ret"].mean()), 4)},
            "live_below": {"n": int((lo_s["ts"] >= DEV_CUT).sum()),
                           "net_mean": round(float(
                               lo_s[lo_s["ts"] >= DEV_CUT]["ret"].mean()), 4)
                           if (lo_s["ts"] >= DEV_CUT).any() else None},
            "live_above": {"n": int((hi_s["ts"] >= DEV_CUT).sum()),
                           "net_mean": round(float(
                               hi_s[hi_s["ts"] >= DEV_CUT]["ret"].mean()), 4)
                           if (hi_s["ts"] >= DEV_CUT).any() else None}},
        "funding": {
            "per_trade_mean": round(float(f_best["fund"].mean()), 5),
            "per_trade_mean_live": round(float(
                f_best[f_best["ts"] >= DEV_CUT]["fund"].mean()), 5)
            if (f_best["ts"] >= DEV_CUT).any() else None,
            "realized_data_share": round(float(f_best["has_fund"].mean()), 3)},
        "caveat": "149-coin universe picked 2026: survivorship flatters longs, "
                  "pre-2024 inflated"}

    # ---- artifact -------------------------------------------------------------
    for tag in grid:
        grid[tag].pop("trades")
    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "study": "Q27_squeeze_continuation_long",
        "spec": {
            "direction": "LONG",
            "event": "ret_6h>=+0.08 AND OI_6h<=-0.10, $1M liq, 24h cooldown, "
                     "1h grid, guard 48 (squeeze_fade-identical)",
            "cells": "entry {taker@next_open 21bps RT, maker@trig_close TTL1h "
                     "12.5bps RT} x exit {4,12,24}h close, disaster stop -15% "
                     "gap-aware, funding realized long-signed",
            "portfolio": {"slots": SLOTS, "symbol_cooldown_h": 24},
            "dev_cut": DEV_CUT.isoformat(),
            "selection": "max DEV net_mean, n>=30, tie-break DEV portfolio CAGR",
            "verdict_rule": "LIVE mean>=+50bps & day-clustered CI90 ex 0 & "
                            "beats price-only by >=30bps LIVE & LIVE maxDD>-15%",
            "landmine": "fade holdout won 2025-26 (price-only explained) => "
                        "continuation-long expected dead in LIVE; clean NO ok"},
        "counts_and_overlap": overlap,
        "grid_events": {t: {"config": g["config"], "report": g["report"]}
                        for t, g in grid.items()},
        "grid_control_price_only": grid_px,
        "grid_control_random": grid_rnd,
        "dev_selected": {"tag": best_tag, "config": best["config"],
                         "stats_dayclustered": stats},
        "verdict": verdict,
        "diagnostics": diag,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    fp = ART_DIR / "results_continuation.json"
    fp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nartifacts -> {fp}\n", flush=True)
    print(json.dumps({"selected": best_tag, "stats": stats, "verdict": verdict,
                      "by_year": diag["by_year_selected_cell"],
                      "overlap": overlap}, indent=1, default=str))


if __name__ == "__main__":
    main()
