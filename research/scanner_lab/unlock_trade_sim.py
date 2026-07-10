"""Token-unlock TRADE SIMULATION (one-shot, PRE-REGISTERED, question #18).

Prices the Q17 unlock event-study signal as ACTUAL TRADES (costs, funding,
entry timing, portfolio, DEV/LIVE split). Q17 (docs/notes/2026-07-09/
open-tickets-closeout.md) found, market-adjusted vs random controls:
  * LARGE unlocks (>=3% supply): -9.2% raw CAR in the 30d BEFORE the cliff
    (-6.1pp incremental; control alt itself -3%/30d), stable DEV/LIVE, 76% neg.
  * AFTER the cliff: +4.2pp/7d incremental POSITIVE (raw +3.4%/7d) -
    sell-the-anticipation / resolve-at-the-event, NOT the folk "dump after".
That was an event study only (no costs/funding/timing/portfolio). This script
prices it. Discipline: selection on DEV only, single verdict shot on LIVE,
ALL grid cells reported for both windows, NO post-hoc parameter moves.

================================ FROZEN SPEC ================================
DATA
  * Events: research/data/unlocks/unlock_events.parquet. AUTHORITATIVE date =
    event_date (string, parsed to midnight UTC). NOTE: event_ts in this file is
    in SECONDS (1512777600 == 2017-12-09), NOT ms as the ticket claimed - we use
    event_date and ignore event_ts. frac_supply = cliff/max_supply (fraction).
  * Instrument: research/data/binance_um/klines_1m/{PAIR}.parquet, resampled to
    1D (last close for entry/exit) and 1h (max high for the squeeze metric).
  * Funding: research/data/perp/funding/{PAIR}.parquet [fundingTime(ms),
    fundingRate]. Loader/summation pattern REUSED from idio_short_study.py
    (funding_sum: +sum of rates settled in (entry_ns, exit_ns]; short RECEIVES
    +funding, long PAYS +funding). Settlement cadence (1h recent / 8h legacy) is
    irrelevant because we sum every settlement inside the hold.

WINDOWS  DEV = event_date < 2025-01-01 ; LIVE = event_date >= 2025-01-01.

UNIVERSE / GATES
  * Universe = the 68 unlock-bearing pairs present in unlock_events that have
    UM klines+funding (the "unlock universe"; also the control/basket pool).
  * Liquidity gate: 30d-median daily quote_volume > $1M at entry_date.
  * Dedup: per threshold bucket, skip an event if the SAME symbol had another
    qualifying cliff (same bucket) within the previous 30 days.
  * A trade is FILLED only if entry_date and exit_date both have a daily close.

RETURN CONVENTIONS (per unit notional on allocated capital)
  * SHORT net = (1 - exit/entry) - COST_RT + funding_sum(entry,exit)
  * LONG  net = (exit/entry - 1) - COST_RT - funding_sum(entry,exit)
  * COST_RT = 0.0025 (25 bps round-trip) PER LEG. Hedge leg pays its own
    COST_RT and its own funding.
  * Funding fallback: if the hold starts before a pair's funding history, the
    uncovered portion uses that year's climate mean DAILY funding (computed from
    the whole funding panel); such trades are COUNTED and reported.

LEG S1 - pre-cliff SHORT (grid = 3 entries x 2 thresholds x 3 hedges = 18)
  * Entry: close of {event_date-30, -20, -10}.  Exit: close of event_date-1.
  * Threshold: frac_supply {>=0.03, >=0.01}.
  * Hedge arms (equal notional, same timestamps, own costs+funding):
      none | long BTCUSDT | long equal-weight basket of the unlock universe.
      Cell net = short_token_net + hedge_long_net (hedge_long_net=0 for none).
  * Selection: best DEV net/trade among cells with n_DEV>=30 (relax to >=20 if
    none reach 30, and say so).

LEG S2 - post-cliff LONG (2 cells, no hedge - declared)
  * Entry: close of event_date.  Exit: close of event_date+7.
  * Threshold {>=0.03, >=0.01}. LONG net convention above.

CONTROL (mandatory, S1 only) - same-dates matched RANDOM-token short
  * For each S1 token-trade, short a random DIFFERENT universe token on the
    SAME entry/exit dates (must have both closes). 20 seeds; report mean and
    the seed distribution. Because the hedge leg is identical for token and
    control on the same dates, cell_net - control_net == token_short_net -
    random_short_net (the pure token-selection effect). Pre-registered
    specificity: selected S1 cell must beat its control by >=150 bps/trade DEV.

PORTFOLIO (DEV-selected S1 cell; S2 cell): 10 slots, cap = equity/10 at entry,
  slot busy until exit, skip if no free slot (count skips). Continuous daily
  mark-to-market (token+hedge daily close path, entry cost as constant drag,
  funding accrued to date); realize exact net at exit. Daily equity/ret/n_open;
  CAGR and maxDD computed per window (DEV segment, LIVE segment).

PASS BARS (frozen)
  * S1: DEV-selected cell on LIVE keeps the same sign AND net >= +100 bps/trade
    after costs+funding AND beats its control by >= 100 bps/trade on LIVE; AND
    portfolio CAGR > 0 in BOTH windows.
  * S2: LIVE same sign AND net >= +50 bps/trade after costs+funding.
  * Any miss = NO for that leg. No reinterpretation.

STORM STRESS (reported regardless of verdict): monthly return table of the
  selected portfolios; zoom windows 2022-05, 2022-11, 2023-06-10, 2024-08-05,
  2025-02-03, 2026-01..06; worst-5 trades per leg + per-trade p5/p1; S1 squeeze
  tail = fraction of trades whose intra-hold 1h high moved >+15% against the
  short; max concurrent positions and slot-skip count. DIAGNOSTIC ONLY (no
  selection): selected-cell per-trade returns bucketed by BTC 30d return at
  entry.

CAVEATS (state honestly): survivorship universe (2026 survivors);
  DefiLlama self-reported supply denominators; anticipation is public knowledge;
  unlock windows cluster/overlap - dispersion read at event & month level.

Artifacts:
  research/data/unlocks/results_trade_sim.json
  research/data/unlocks/daily_returns_s1.parquet
  research/data/unlocks/daily_returns_s2.parquet
Usage: python unlock_trade_sim.py
============================================================================
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
KL_DIR = REPO_ROOT / "research" / "data" / "binance_um" / "klines_1m"
FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
UNLOCK_DIR = REPO_ROOT / "research" / "data" / "unlocks"
EVENTS_PATH = UNLOCK_DIR / "unlock_events.parquet"

COST_RT = 0.0025
DEV_CUT = pd.Timestamp("2025-01-01")
LIQ_MIN = 1e6
DEDUP_DAYS = 30
S1_ENTRIES = [30, 20, 10]
S1_HEDGES = ["none", "btc", "basket"]
THRESHOLDS = [0.03, 0.01]
S2_HOLD_DAYS = 7
N_SEEDS = 20
SLOTS = 10
DAY_NS = 86_400_000_000_000
STORM_MONTHS = ["2022-05", "2022-11", "2023-06", "2024-08", "2025-02",
                "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]

# --------------------------------------------------------------------------- #
# Loaders (cached)
# --------------------------------------------------------------------------- #
_daily: dict[str, tuple[pd.Series, pd.Series]] = {}
_hourly_high: dict[str, pd.Series] = {}
_fund: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
_fund_first: dict[str, int] = {}


def daily(pair: str) -> tuple[pd.Series, pd.Series] | None:
    """(close, 30d-median-daily-quote-volume) indexed by midnight-UTC date."""
    if pair in _daily:
        return _daily[pair]
    p = KL_DIR / f"{pair}.parquet"
    if not p.exists():
        _daily[pair] = None
        return None
    k = pd.read_parquet(p, columns=["open_time", "close", "quote_volume"])
    k.index = pd.to_datetime(k["open_time"], unit="ms")
    c = k["close"].resample("1D").last()
    v = k["quote_volume"].resample("1D").sum()
    liq = v.rolling(30).median()
    c = c[~c.index.duplicated(keep="last")].dropna()
    _daily[pair] = (c, liq)
    return _daily[pair]


def hourly_high(pair: str) -> pd.Series:
    if pair in _hourly_high:
        return _hourly_high[pair]
    k = pd.read_parquet(KL_DIR / f"{pair}.parquet", columns=["open_time", "high"])
    k.index = pd.to_datetime(k["open_time"], unit="ms")
    h = k["high"].resample("1h").max().dropna()
    _hourly_high[pair] = h
    return h


def load_funding(pair: str):
    """(settle_ns, rate). REUSED pattern from idio_short_study.load_funding."""
    if pair in _fund:
        return _fund[pair]
    p = FUND_DIR / f"{pair}.parquet"
    if not p.exists():
        _fund[pair] = None
        _fund_first[pair] = np.iinfo(np.int64).max
        return None
    f = pd.read_parquet(p, columns=["fundingTime", "fundingRate"])
    ts = pd.to_datetime(f["fundingTime"], unit="ms").dt.round("1h")
    s = pd.Series(f["fundingRate"].to_numpy(np.float64), index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    arr = (s.index.asi8.astype(np.int64), s.to_numpy(np.float64))
    _fund[pair] = arr
    _fund_first[pair] = int(arr[0][0]) if len(arr[0]) else np.iinfo(np.int64).max
    return arr


def funding_sum(pair: str, entry_ns: int, exit_ns: int) -> float:
    fc = load_funding(pair)
    if fc is None:
        return 0.0
    settle, rate = fc
    lo = int(np.searchsorted(settle, entry_ns, side="right"))
    hi = int(np.searchsorted(settle, exit_ns, side="right"))
    if hi <= lo:
        return 0.0
    return float(rate[lo:hi].sum())


# per-year climate DAILY funding mean (fallback), built in main()
CLIMATE_DAILY: dict[int, float] = {}


def funding_with_fallback(pair: str, entry_ns: int, exit_ns: int, year: int,
                          hold_days: float) -> tuple[float, bool]:
    """Return (funding_sum, used_fallback). Fallback only for the portion of the
    hold BEFORE the pair's funding history begins."""
    ffirst = _fund_first.get(pair)
    if ffirst is None:
        load_funding(pair)
        ffirst = _fund_first.get(pair, np.iinfo(np.int64).max)
    if entry_ns >= ffirst:
        return funding_sum(pair, entry_ns, exit_ns), False
    # hold starts before funding history -> climate for uncovered days
    covered = funding_sum(pair, max(entry_ns, ffirst), exit_ns)
    uncov_days = max(0.0, (min(ffirst, exit_ns) - entry_ns) / DAY_NS)
    return covered + CLIMATE_DAILY.get(year, 0.0) * uncov_days, True


def close_ns(date: pd.Timestamp) -> int:
    """Funding timestamp of a daily close == end of that UTC day."""
    return int((date + pd.Timedelta(days=1)).value)


# --------------------------------------------------------------------------- #
# Single-leg returns
# --------------------------------------------------------------------------- #
def _prices(pair: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp):
    d = daily(pair)
    if d is None:
        return None
    c = d[0]
    if entry_date not in c.index or exit_date not in c.index:
        return None
    return float(c.loc[entry_date]), float(c.loc[exit_date])


def short_leg(pair, entry_date, exit_date):
    pr = _prices(pair, entry_date, exit_date)
    if pr is None:
        return None
    e, x = pr
    if e <= 0 or x <= 0:
        return None
    hold = max(1.0, (exit_date - entry_date).days)
    fund, fb = funding_with_fallback(pair, close_ns(entry_date), close_ns(exit_date),
                                     entry_date.year, hold)
    gross = 1.0 - x / e
    return {"gross": gross, "fund": fund, "net": gross - COST_RT + fund, "fb": fb}


def long_leg(pair, entry_date, exit_date):
    pr = _prices(pair, entry_date, exit_date)
    if pr is None:
        return None
    e, x = pr
    if e <= 0 or x <= 0:
        return None
    hold = max(1.0, (exit_date - entry_date).days)
    fund, fb = funding_with_fallback(pair, close_ns(entry_date), close_ns(exit_date),
                                     entry_date.year, hold)
    gross = x / e - 1.0
    return {"gross": gross, "fund": fund, "net": gross - COST_RT - fund, "fb": fb}


def basket_long(universe, entry_date, exit_date, exclude=None):
    nets, fbs = [], 0
    for p in universe:
        if p == exclude:
            continue
        r = long_leg(p, entry_date, exit_date)
        if r is not None:
            nets.append(r["net"])
            fbs += int(r["fb"])
    if not nets:
        return None
    return {"net": float(np.mean(nets)), "n": len(nets), "fb": int(fbs > 0)}


# --------------------------------------------------------------------------- #
# Event preparation
# --------------------------------------------------------------------------- #
def dedup_events(ev: pd.DataFrame, thr: float) -> pd.DataFrame:
    sub = ev[ev["frac_supply"] >= thr].sort_values(["pair", "ed"]).copy()
    keep, last = [], {}
    for idx, r in sub.iterrows():
        p, d = r["pair"], r["ed"]
        if p in last and (d - last[p]).days < DEDUP_DAYS:
            continue
        keep.append(idx)
        last[p] = d
    return sub.loc[keep]


def liquidity_ok(pair, entry_date) -> bool:
    d = daily(pair)
    if d is None:
        return False
    liq = d[1]
    if entry_date not in liq.index:
        return False
    v = liq.loc[entry_date]
    return bool(pd.notna(v) and v > LIQ_MIN)


# --------------------------------------------------------------------------- #
# S1 token-trade set (per threshold x entry) + controls
# --------------------------------------------------------------------------- #
def build_s1_token_trades(ev, universe):
    """dict[(thr, entry_off)] -> list of trade dicts (short leg only, filled,
    dedup+liquidity gated), with control seed returns attached."""
    out = {}
    funnel = {}
    for thr in THRESHOLDS:
        ded = dedup_events(ev, thr)
        for off in S1_ENTRIES:
            trades = []
            n_liqfail = 0
            n_fillfail = 0
            for _, r in ded.iterrows():
                d = r["ed"]
                entry_date = d - pd.Timedelta(days=off)
                exit_date = d - pd.Timedelta(days=1)
                if not liquidity_ok(r["pair"], entry_date):
                    n_liqfail += 1
                    continue
                sh = short_leg(r["pair"], entry_date, exit_date)
                if sh is None:
                    n_fillfail += 1
                    continue
                win = "DEV" if d < DEV_CUT else "LIVE"
                trades.append({"pair": r["pair"], "entry": entry_date,
                               "exit": exit_date, "event": d, "win": win,
                               "short": sh})
            # controls: 20 seeds of random universe short, same dates
            for t in trades:
                elig = [p for p in universe if p != t["pair"]]
                ctrls = []
                for seed in range(N_SEEDS):
                    rng = np.random.default_rng(
                        seed * 1_000_003 + int(t["entry"].value // DAY_NS))
                    order = rng.permutation(len(elig))
                    val = None
                    for j in order:
                        rr = short_leg(elig[j], t["entry"], t["exit"])
                        if rr is not None:
                            val = rr["net"]
                            break
                    if val is not None:
                        ctrls.append(val)
                t["ctrl_seeds"] = ctrls
                t["ctrl_mean"] = float(np.mean(ctrls)) if ctrls else np.nan
            out[(thr, off)] = trades
            funnel[(thr, off)] = {"dedup": int(len(ded)), "liq_fail": n_liqfail,
                                  "fill_fail": n_fillfail, "filled": len(trades),
                                  "filled_DEV": sum(t["win"] == "DEV" for t in trades),
                                  "filled_LIVE": sum(t["win"] == "LIVE" for t in trades)}
    return out, funnel


def s1_cell(token_trades, universe, thr, off, hedge):
    """Return per-window stats + per-trade list for one grid cell."""
    trades = token_trades[(thr, off)]
    rows = []
    for t in trades:
        short_net = t["short"]["net"]
        if hedge == "none":
            hnet = 0.0
        elif hedge == "btc":
            hr = long_leg("BTCUSDT", t["entry"], t["exit"])
            hnet = hr["net"] if hr else 0.0
        else:
            hr = basket_long(universe, t["entry"], t["exit"], exclude=t["pair"])
            hnet = hr["net"] if hr else 0.0
        cell_net = short_net + hnet
        ctrl_net = t["ctrl_mean"] + hnet  # hedge identical -> cancels in diff
        rows.append({**t, "hnet": hnet, "cell_net": cell_net,
                     "ctrl_net": ctrl_net})
    res = {}
    for win in ["DEV", "LIVE"]:
        w = [r for r in rows if r["win"] == win]
        nets = np.array([r["cell_net"] for r in w])
        ctrls = np.array([r["ctrl_net"] for r in w if not np.isnan(r["ctrl_net"])])
        shorts = np.array([r["short"]["net"] for r in w])
        res[win] = {
            "n": len(w),
            "net_mean": float(nets.mean()) if len(nets) else np.nan,
            "net_med": float(np.median(nets)) if len(nets) else np.nan,
            "short_only_mean": float(shorts.mean()) if len(shorts) else np.nan,
            "ctrl_mean": float(ctrls.mean()) if len(ctrls) else np.nan,
            "beat_ctrl": float(nets.mean() - ctrls.mean()) if len(nets) and len(ctrls) else np.nan,
            "win_rate": float((nets > 0).mean()) if len(nets) else np.nan,
            "fund_mean": float(np.mean([r["short"]["fund"] for r in w])) if w else np.nan,
        }
    return res, rows


# --------------------------------------------------------------------------- #
# S2 long trades
# --------------------------------------------------------------------------- #
def build_s2(ev):
    out = {}
    for thr in THRESHOLDS:
        ded = dedup_events(ev, thr)
        rows = []
        n_liqfail = n_fillfail = 0
        for _, r in ded.iterrows():
            d = r["ed"]
            entry_date = d
            exit_date = d + pd.Timedelta(days=S2_HOLD_DAYS)
            if not liquidity_ok(r["pair"], entry_date):
                n_liqfail += 1
                continue
            lo = long_leg(r["pair"], entry_date, exit_date)
            if lo is None:
                n_fillfail += 1
                continue
            win = "DEV" if d < DEV_CUT else "LIVE"
            rows.append({"pair": r["pair"], "entry": entry_date, "exit": exit_date,
                         "event": d, "win": win, "net": lo["net"],
                         "gross": lo["gross"], "fund": lo["fund"], "fb": lo["fb"]})
        stats = {}
        for win in ["DEV", "LIVE"]:
            w = [x for x in rows if x["win"] == win]
            nets = np.array([x["net"] for x in w])
            stats[win] = {"n": len(w),
                          "net_mean": float(nets.mean()) if len(nets) else np.nan,
                          "net_med": float(np.median(nets)) if len(nets) else np.nan,
                          "win_rate": float((nets > 0).mean()) if len(nets) else np.nan}
        out[thr] = {"stats": stats, "rows": rows,
                    "funnel": {"dedup": len(ded), "liq_fail": n_liqfail,
                               "fill_fail": n_fillfail, "filled": len(rows)}}
    return out


# --------------------------------------------------------------------------- #
# Portfolio (continuous daily MTM, 10 slots)
# --------------------------------------------------------------------------- #
def _leg_mtm(rows, pair_close_path, universe):
    pass  # placeholder (unused) - MTM computed inline below


def portfolio(rows, side, universe, hedge="none"):
    """rows: filled trades sorted by entry. side='short'|'long'.
    Continuous daily MTM. Returns (daily_df, metrics)."""
    if not rows:
        return pd.DataFrame(columns=["date", "ret", "equity", "n_open"]), {}
    rows = sorted(rows, key=lambda r: r["entry"])
    start = min(r["entry"] for r in rows)
    end = max(r["exit"] for r in rows)
    days = pd.date_range(start, end, freq="1D")

    # precompute each trade's daily leg-return path (fraction of cap)
    def path_for(r):
        e_tok = _prices(r["pair"], r["entry"], r["entry"])
        cpath = daily(r["pair"])[0]
        seg = cpath.loc[r["entry"]:r["exit"]]
        e0 = float(seg.iloc[0])
        if side == "short":
            base = 1.0 - seg / e0
        else:
            base = seg / e0 - 1.0
        # hedge path
        if hedge == "btc":
            hseg = daily("BTCUSDT")[0].loc[r["entry"]:r["exit"]]
            h0 = float(hseg.iloc[0])
            hbase = (hseg / h0 - 1.0).reindex(seg.index, method="ffill").fillna(0.0)
        elif hedge == "basket":
            hb = []
            for p in universe:
                if p == r["pair"]:
                    continue
                cp = daily(p)
                if cp is None:
                    continue
                s2 = cp[0].loc[r["entry"]:r["exit"]]
                if len(s2) < 1:
                    continue
                s2 = s2 / float(s2.iloc[0]) - 1.0
                hb.append(s2.reindex(seg.index, method="ffill"))
            hbase = (pd.concat(hb, axis=1).mean(axis=1).fillna(0.0)
                     if hb else pd.Series(0.0, index=seg.index))
        else:
            hbase = pd.Series(0.0, index=seg.index)
        # funding accrued to each day (short receives +, long pays -)
        settle_fund = pd.Series(0.0, index=seg.index)
        for dt in seg.index:
            f = funding_sum(r["pair"], close_ns(r["entry"]), close_ns(dt))
            settle_fund.loc[dt] = f if side == "short" else -f
        cost_drag = COST_RT * (2 if hedge != "none" else 1)
        frac = base + hbase + settle_fund - cost_drag
        frac.loc[r["exit"]] = r.get("_final_net", frac.loc[r["exit"]])
        return frac

    for r in rows:
        r["_final_net"] = r["cell_net"] if "cell_net" in r else r["net"]
    paths = {id(r): path_for(r) for r in rows}

    equity = 1.0
    cash = 1.0
    open_pos = []
    by_entry = {}
    for r in rows:
        by_entry.setdefault(r["entry"], []).append(r)
    prev_eq = 1.0
    recs = []
    skips = 0
    max_open = 0
    for day in days:
        # open new positions at their entry day
        for r in by_entry.get(day, []):
            if len(open_pos) >= SLOTS:
                skips += 1
                continue
            eq_now = cash + sum(p["cap"] * (1 + paths[id(p)].get(day, paths[id(p)].iloc[-1]))
                                for p in open_pos)
            cap = eq_now / SLOTS
            cash -= cap
            r["cap"] = cap
            open_pos.append(r)
        # mark & close
        still = []
        for p in open_pos:
            fr = paths[id(p)]
            val = p["cap"] * (1 + (fr.loc[day] if day in fr.index else fr.iloc[-1]))
            if day >= p["exit"]:
                cash += val  # realize
            else:
                still.append(p)
        open_pos = still
        max_open = max(max_open, len(open_pos))
        eq = cash + sum(p["cap"] * (1 + (paths[id(p)].loc[day] if day in paths[id(p)].index
                                         else paths[id(p)].iloc[-1])) for p in open_pos)
        ret = eq / prev_eq - 1.0 if prev_eq else 0.0
        recs.append({"date": day, "ret": ret, "equity": eq, "n_open": len(open_pos)})
        prev_eq = eq
        equity = eq
    df = pd.DataFrame(recs)
    metrics = _port_metrics(df)
    metrics["slot_skips"] = int(skips)
    metrics["max_concurrent"] = int(max_open)
    return df, metrics


def _cagr_maxdd(eq: pd.Series, dates: pd.Series):
    if len(eq) < 2:
        return np.nan, np.nan
    yrs = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[0] > 0 else np.nan
    roll = eq.cummax()
    mdd = float((eq / roll - 1).min())
    return float(cagr), mdd


def _port_metrics(df):
    if df.empty:
        return {}
    m = {}
    full = df.set_index("date")["equity"]
    m["cagr_full"], m["maxdd_full"] = _cagr_maxdd(full.reset_index(drop=True),
                                                  df["date"].reset_index(drop=True))
    for win, mask in [("DEV", df["date"] < DEV_CUT), ("LIVE", df["date"] >= DEV_CUT)]:
        seg = df[mask]
        if len(seg) >= 2:
            # renormalize segment equity to its own start
            eq = seg["equity"] / seg["equity"].iloc[0]
            c, d = _cagr_maxdd(eq.reset_index(drop=True), seg["date"].reset_index(drop=True))
            m[f"cagr_{win}"] = c
            m[f"maxdd_{win}"] = d
        else:
            m[f"cagr_{win}"] = np.nan
            m[f"maxdd_{win}"] = np.nan
    # monthly returns
    mr = (1 + df.set_index("date")["ret"]).resample("ME").prod() - 1
    m["monthly"] = {str(k.date()): round(float(v), 5) for k, v in mr.items()}
    return m


# --------------------------------------------------------------------------- #
# Storm stress helpers
# --------------------------------------------------------------------------- #
def squeeze_tail(rows):
    """Fraction of short trades whose intra-hold 1h high moved > +15% vs entry."""
    n = 0
    hits = 0
    maxes = []
    for r in rows:
        c = daily(r["pair"])[0]
        e = float(c.loc[r["entry"]])
        hh = hourly_high(r["pair"])
        seg = hh.loc[r["entry"]:r["exit"] + pd.Timedelta(days=1)]
        if len(seg) == 0:
            continue
        adverse = float(seg.max()) / e - 1.0
        maxes.append(adverse)
        n += 1
        if adverse > 0.15:
            hits += 1
    return {"n": n, "frac_gt15": (hits / n if n else np.nan),
            "median_max_adverse": float(np.median(maxes)) if maxes else np.nan,
            "p95_max_adverse": float(np.percentile(maxes, 95)) if maxes else np.nan}


def worst_trades(rows, key, k=5):
    s = sorted(rows, key=lambda r: r[key])[:k]
    return [{"pair": r["pair"], "entry": str(r["entry"].date()),
             "exit": str(r["exit"].date()), "ret": round(float(r[key]), 4),
             "win": r["win"]} for r in s]


def btc_regime_buckets(rows, key):
    btc = daily("BTCUSDT")[0]
    out = {"btc_up": [], "btc_flat": [], "btc_down": []}
    for r in rows:
        ed = r["entry"]
        past = ed - pd.Timedelta(days=30)
        if ed in btc.index and past in btc.index:
            b30 = btc.loc[ed] / btc.loc[past] - 1
        else:
            continue
        bucket = "btc_up" if b30 > 0.05 else ("btc_down" if b30 < -0.05 else "btc_flat")
        out[bucket].append(float(r[key]))
    return {k2: {"n": len(v), "mean": (float(np.mean(v)) if v else np.nan)}
            for k2, v in out.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ev = pd.read_parquet(EVENTS_PATH)
    ev["ed"] = pd.to_datetime(ev["event_date"])
    kl_pairs = {p.stem for p in KL_DIR.glob("*.parquet")}
    fu_pairs = {p.stem for p in FUND_DIR.glob("*.parquet")}
    ev = ev[ev["pair"].isin(kl_pairs) & ev["pair"].isin(fu_pairs)].copy()
    universe = sorted(ev["pair"].unique())
    raw_counts = {"raw_events_total": int(len(pd.read_parquet(EVENTS_PATH))),
                  "events_with_data": int(len(ev)),
                  "universe_pairs": len(universe)}

    # per-year climate daily funding (fallback source)
    print("[setup] building per-year climate funding ...", flush=True)
    daily_sums = {}
    for p in fu_pairs:
        f = pd.read_parquet(FUND_DIR / f"{p}.parquet", columns=["fundingTime", "fundingRate"])
        ts = pd.to_datetime(f["fundingTime"], unit="ms")
        ds = pd.Series(f["fundingRate"].to_numpy(np.float64), index=ts).resample("1D").sum()
        for yr, g in ds.groupby(ds.index.year):
            daily_sums.setdefault(int(yr), []).append(g)
    for yr, lst in daily_sums.items():
        CLIMATE_DAILY[yr] = float(pd.concat(lst).mean())
    print("  climate daily funding:", {k: round(v, 6) for k, v in sorted(CLIMATE_DAILY.items())}, flush=True)

    # ---------------- S1 ----------------
    print("[S1] building token trades + controls ...", flush=True)
    token_trades, funnel = build_s1_token_trades(ev, universe)
    fb_count = sum(t["short"]["fb"] for tr in token_trades.values() for t in tr)

    grid = {}
    cell_rows = {}
    for thr in THRESHOLDS:
        for off in S1_ENTRIES:
            for hedge in S1_HEDGES:
                res, rows = s1_cell(token_trades, universe, thr, off, hedge)
                key = f"thr{thr}_e{off}_{hedge}"
                grid[key] = {"thr": thr, "entry": off, "hedge": hedge,
                             "DEV": res["DEV"], "LIVE": res["LIVE"]}
                cell_rows[key] = rows
    # selection on DEV
    def eligible(minn):
        return {k: v for k, v in grid.items()
                if v["DEV"]["n"] >= minn and not np.isnan(v["DEV"]["net_mean"])}
    elig = eligible(30)
    relax = False
    if not elig:
        elig = eligible(20)
        relax = True
    sel_key = max(elig, key=lambda k: elig[k]["DEV"]["net_mean"]) if elig else None
    print(f"[S1] selected cell = {sel_key} (relax_to_20={relax})", flush=True)

    # ---------------- S2 ----------------
    print("[S2] building long trades ...", flush=True)
    s2 = build_s2(ev)
    # S2 selection: pick threshold by best DEV mean (both cells reported)
    s2_sel = max(THRESHOLDS,
                 key=lambda t: (s2[t]["stats"]["DEV"]["net_mean"]
                                if not np.isnan(s2[t]["stats"]["DEV"]["net_mean"]) else -9))

    # ---------------- Portfolios ----------------
    print("[PORT] S1 portfolio ...", flush=True)
    sel = grid[sel_key]
    s1_rows_all = cell_rows[sel_key]
    df_s1, m_s1 = portfolio([dict(r) for r in s1_rows_all], "short", universe,
                            hedge=sel["hedge"])
    print("[PORT] S2 portfolio ...", flush=True)
    s2_rows_all = s2[s2_sel]["rows"]
    df_s2, m_s2 = portfolio([dict(r) for r in s2_rows_all], "long", universe, hedge="none")

    # ---------------- Verdicts ----------------
    s1_dev = sel["DEV"]
    s1_live = sel["LIVE"]
    s1_pass = bool(
        (np.sign(s1_dev["net_mean"]) == np.sign(s1_live["net_mean"])) and
        (s1_live["net_mean"] >= 0.01) and
        (s1_live["beat_ctrl"] >= 0.01) and
        (m_s1.get("cagr_DEV", -1) > 0) and (m_s1.get("cagr_LIVE", -1) > 0)
    )
    spec_dev_ok = bool(s1_dev["beat_ctrl"] >= 0.015)
    s2d = s2[s2_sel]["stats"]["DEV"]
    s2l = s2[s2_sel]["stats"]["LIVE"]
    s2_pass = bool((np.sign(s2d["net_mean"]) == np.sign(s2l["net_mean"])) and
                   (s2l["net_mean"] >= 0.005))

    # ---------------- Storm stress ----------------
    print("[STORM] computing ...", flush=True)
    sq = squeeze_tail(s1_rows_all)
    storm = {
        "s1_squeeze_tail": sq,
        "s1_worst5": worst_trades(s1_rows_all, "cell_net"),
        "s2_worst5": worst_trades(s2_rows_all, "net"),
        "s1_pctiles": {"p5": float(np.percentile([r["cell_net"] for r in s1_rows_all], 5)),
                       "p1": float(np.percentile([r["cell_net"] for r in s1_rows_all], 1))},
        "s2_pctiles": {"p5": float(np.percentile([r["net"] for r in s2_rows_all], 5)),
                       "p1": float(np.percentile([r["net"] for r in s2_rows_all], 1))},
        "s1_max_concurrent": m_s1.get("max_concurrent"),
        "s1_slot_skips": m_s1.get("slot_skips"),
        "s2_max_concurrent": m_s2.get("max_concurrent"),
        "s2_slot_skips": m_s2.get("slot_skips"),
        "s1_monthly": m_s1.get("monthly"),
        "s2_monthly": m_s2.get("monthly"),
        "s1_btc_regime": btc_regime_buckets(s1_rows_all, "cell_net"),
        "s2_btc_regime": btc_regime_buckets(s2_rows_all, "net"),
        "zoom_windows": {},
    }
    for mk in STORM_MONTHS:
        s1m = (m_s1.get("monthly") or {})
        s2m = (m_s2.get("monthly") or {})
        s1v = [v for k, v in s1m.items() if k.startswith(mk)]
        s2v = [v for k, v in s2m.items() if k.startswith(mk)]
        # open trades whose hold overlaps that month
        month_ts = pd.Timestamp(mk + "-01")
        month_end = month_ts + pd.offsets.MonthEnd(0)
        s1_open = [{"pair": r["pair"], "ret": round(float(r["cell_net"]), 4),
                    "entry": str(r["entry"].date())}
                   for r in s1_rows_all if r["entry"] <= month_end and r["exit"] >= month_ts]
        storm["zoom_windows"][mk] = {"s1_month_ret": s1v[0] if s1v else None,
                                     "s2_month_ret": s2v[0] if s2v else None,
                                     "s1_open_trades": s1_open}

    # ---------------- Assemble grid table ----------------
    def cell_compact(v):
        return {"thr": v["thr"], "entry": v["entry"], "hedge": v["hedge"],
                "DEV_n": v["DEV"]["n"], "DEV_net": _r(v["DEV"]["net_mean"]),
                "DEV_ctrl": _r(v["DEV"]["ctrl_mean"]), "DEV_beat": _r(v["DEV"]["beat_ctrl"]),
                "DEV_win": _r(v["DEV"]["win_rate"]),
                "LIVE_n": v["LIVE"]["n"], "LIVE_net": _r(v["LIVE"]["net_mean"]),
                "LIVE_ctrl": _r(v["LIVE"]["ctrl_mean"]), "LIVE_beat": _r(v["LIVE"]["beat_ctrl"]),
                "LIVE_win": _r(v["LIVE"]["win_rate"])}

    results = {
        "spec": "unlock_trade_sim v1 (question #18); see module docstring",
        "counts": raw_counts,
        "climate_daily_funding": {str(k): round(v, 6) for k, v in sorted(CLIMATE_DAILY.items())},
        "funnel_s1": {f"thr{k[0]}_e{k[1]}": v for k, v in funnel.items()},
        "funnel_s2": {str(t): s2[t]["funnel"] for t in THRESHOLDS},
        "fallback_funding_trades_s1": int(fb_count),
        "s1_grid": {k: cell_compact(v) for k, v in grid.items()},
        "s1_selected": {"cell": sel_key, "relaxed_to_20": relax,
                        "DEV": {kk: _r(vv) for kk, vv in s1_dev.items()},
                        "LIVE": {kk: _r(vv) for kk, vv in s1_live.items()},
                        "specificity_DEV_beat_ctrl>=150bps": spec_dev_ok},
        "s1_portfolio": {k: (v if k != "monthly" else "see storm") for k, v in m_s1.items()},
        "s2_cells": {str(t): {"DEV": {kk: _r(vv) for kk, vv in s2[t]["stats"]["DEV"].items()},
                              "LIVE": {kk: _r(vv) for kk, vv in s2[t]["stats"]["LIVE"].items()}}
                     for t in THRESHOLDS},
        "s2_selected_thr": s2_sel,
        "s2_portfolio": {k: (v if k != "monthly" else "see storm") for k, v in m_s2.items()},
        "storm": storm,
        "verdicts": {
            "S1": {"pass": s1_pass,
                   "bar": "LIVE same sign & net>=+100bps & beat_ctrl>=+100bps & port CAGR>0 both windows",
                   "DEV_net": _r(s1_dev["net_mean"]), "LIVE_net": _r(s1_live["net_mean"]),
                   "LIVE_beat_ctrl": _r(s1_live["beat_ctrl"]),
                   "port_cagr_DEV": _r(m_s1.get("cagr_DEV")), "port_cagr_LIVE": _r(m_s1.get("cagr_LIVE"))},
            "S2": {"pass": s2_pass,
                   "bar": "LIVE same sign & net>=+50bps",
                   "DEV_net": _r(s2d["net_mean"]), "LIVE_net": _r(s2l["net_mean"])},
        },
        "caveats": ["survivorship universe (2026 survivors)",
                    "DefiLlama self-reported supply denominators",
                    "anticipation is public knowledge",
                    "unlock windows cluster/overlap - dispersion read at event & month level",
                    "event_ts field is SECONDS not ms; event_date used as authoritative"],
    }

    UNLOCK_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNLOCK_DIR / "results_trade_sim.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    if not df_s1.empty:
        df_s1.to_parquet(UNLOCK_DIR / "daily_returns_s1.parquet", index=False)
    if not df_s2.empty:
        df_s2.to_parquet(UNLOCK_DIR / "daily_returns_s2.parquet", index=False)

    # ---------------- console summary ----------------
    print("\n===== FUNNEL (S1) =====")
    for k, v in funnel.items():
        print(f"  thr{k[0]} e-{k[1]}: dedup={v['dedup']} liqfail={v['liq_fail']} "
              f"fillfail={v['fill_fail']} filled={v['filled']} "
              f"(DEV={v['filled_DEV']} LIVE={v['filled_LIVE']})")
    print(f"  fallback-funding S1 trades: {fb_count}")
    print("\n===== S1 GRID (net bps/trade) =====")
    print(f"  {'cell':28s} {'DEVn':>4} {'DEVnet':>8} {'DEVbeat':>8} {'LIVEn':>5} {'LIVEnet':>8} {'LIVEbeat':>8}")
    for k, v in grid.items():
        print(f"  {k:28s} {v['DEV']['n']:>4} {_bps(v['DEV']['net_mean']):>8} "
              f"{_bps(v['DEV']['beat_ctrl']):>8} {v['LIVE']['n']:>5} "
              f"{_bps(v['LIVE']['net_mean']):>8} {_bps(v['LIVE']['beat_ctrl']):>8}")
    print(f"\n  SELECTED S1: {sel_key} relaxed20={relax}")
    print(f"    DEV net={_bps(s1_dev['net_mean'])} beat_ctrl={_bps(s1_dev['beat_ctrl'])} n={s1_dev['n']}")
    print(f"    LIVE net={_bps(s1_live['net_mean'])} beat_ctrl={_bps(s1_live['beat_ctrl'])} n={s1_live['n']}")
    print(f"    port CAGR DEV={_r(m_s1.get('cagr_DEV'))} LIVE={_r(m_s1.get('cagr_LIVE'))} "
          f"maxDD DEV={_r(m_s1.get('maxdd_DEV'))} LIVE={_r(m_s1.get('maxdd_LIVE'))}")
    print(f"    squeeze>+15%: {_r(sq['frac_gt15'])} (n={sq['n']})  S1 VERDICT={'PASS' if s1_pass else 'NO'}")
    print("\n===== S2 CELLS (net bps/trade) =====")
    for t in THRESHOLDS:
        st = s2[t]["stats"]
        print(f"  thr{t}: DEV n={st['DEV']['n']} net={_bps(st['DEV']['net_mean'])} | "
              f"LIVE n={st['LIVE']['n']} net={_bps(st['LIVE']['net_mean'])}")
    print(f"  SELECTED S2 thr={s2_sel}  port CAGR DEV={_r(m_s2.get('cagr_DEV'))} "
          f"LIVE={_r(m_s2.get('cagr_LIVE'))}  S2 VERDICT={'PASS' if s2_pass else 'NO'}")
    print("\nartifacts written to", UNLOCK_DIR)


def _r(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 5)


def _bps(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a"
    return f"{x*1e4:+.0f}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return str(o)


if __name__ == "__main__":
    main()
