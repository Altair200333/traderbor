"""Q20 - S1 DEPLOYMENT-REALISM SIM (one-shot, PRE-REGISTERED; ledger question #22,
internal name Q20 - keep "Q20" in outputs).

QUESTION: Is S1 (pre-cliff unlock SHORT, Q18 PASS cell) deployable in reality on a
small Bybit account - i.e. does the edge survive (a) a stop/liquidation model,
(b) margin realism, (c) hedge simplification - under pre-registered bars?

Q18 reference (research/scanner_lab/unlock_trade_sim.py, results_trade_sim.json):
DEV-selected S1 cell = thr>=0.01 supply, entry -30d, exit t-1d, EQUAL-WEIGHT-BASKET
hedge; DEV +396bps -> LIVE +319bps/trade; no stop; 55.8% of trades see >+15%
intra-hold adverse; worst trade -209% (= real liquidation); port maxDD -27% DEV /
-21% LIVE. BTC-hedge cell was DEV +168bps (reference).

================================ FROZEN SPEC ================================
Discipline: this FULL spec (grids, rules, verdict criteria) frozen BEFORE the first
run. DEV (event_date < 2025-01-01) selects, LIVE (>= 2025-01-01) gives the verdict.
ALL cells reported for both windows. Any post-hoc analysis labelled as such.

BASE CONFIG (= Q18 DEV-selected S1 cell, conventions REUSED via import of
unlock_trade_sim: loaders, event construction, dedup, liquidity gate, funding
with per-year climate fallback, return conventions):
  * Events research/data/unlocks/unlock_events.parquet, event_date authoritative
    (event_ts is SECONDS - ignored). Universe = unlock pairs with UM klines+funding.
  * thr frac_supply >= 0.01; entry = close of event_date-30; plan exit = close of
    event_date-1; 30d same-symbol dedupe; $1M 30d-median liquidity gate at entry;
    fill requires both daily closes.
  * COST_RT = 25bps PER LEG (hedge pays its own cost + funding). Short receives
    +funding, long pays +funding. Climate-fallback share reported.

INTRA-HOLD PATHS: 1m klines resampled to 1h. Position lives from the entry daily
close to the exit daily close; the stop/liquidation scan covers hourly bars
labelled [entry_date+1d 00:00, exit_date+1d 00:00) i.e. strictly after entry
close. Stops are checked on HOURLY HIGHS - optimistic vs tick wicks; conservative
fill assumption = fill at stop-price-or-worse (see haircut), and the 3%-worse
sensitivity quantifies the wick/gap risk. STATED HONESTLY: hourly highs can miss
sub-hour spikes that would have stopped (or liquidated) a live position.

STOP GRID on the short leg (pre-declared; DEV selects ONE):
  * Levels: +25%, +35%, +50% adverse from entry price, plus NO-STOP (reference
    only - never selectable). All levels sit outside the +17% median max-adverse
    noise zone (lab stop rule, docs/notes/2026-07-08).
  * Trigger: first hourly bar in the hold whose HIGH >= entry*(1+level).
  * Fill: entry*(1+level)*(1+haircut); BASE haircut = 1% (gap allowance),
    SENSITIVITY haircut = 3%. Verdict is evaluated at the 1% haircut; the 3% run
    is reported for all cells.
  * On a stop: BOTH legs close - short at the fill price above (funding accrued
    to the stop bar end), hedge at the stop bar's hourly close (own cost+funding
    to the same time). No re-entry. Post-stop forgone bounce reported per cell
    (= no-stop cell net minus stopped cell net, over stopped trades).

LIQUIDATION MODEL: each short leg is ISOLATED margin at 1x notional (margin =
notional). Bybit linear-perp approx for a 1x short: forced close at +98.5%
adverse (+99% minus 0.5% maintenance) with an additional 1% liquidation penalty
=> short gross at liquidation = -(0.985 + 0.01). Applies to every cell; with a
+25/35/50% stop it cannot fire (stop triggers first), so it is live only in
no-stop arms. Hedge-leg liquidation (long, needs -98.5%) is ignored as
unreachable - stated. A deployable cell must have ZERO simulated liquidations
on DEV+LIVE.

HEDGE ARMS (pre-declared; DEV selects ONE):
  (a) basket = full equal-weight long basket of the unlock universe (Q18
      reference), excl. the shorted token, notional = short notional;
  (b) btc    = single BTCUSDT long sized to leg notional;
  (c) none   = no hedge (REFERENCE ONLY, known beta-contaminated - never
      selectable).

SELECTION RULE (frozen): among the 6 candidate cells {+25,+35,+50} x
{basket,btc} at the 1% haircut, pick argmax DEV net mean/trade, requiring
DEV n>=30 (relax to >=20 if none, and say so) and zero DEV liquidations.
Tie-break: wider stop. no-stop and no-hedge arms are reference rows.

PORTFOLIO REALISM on $5,000: 10 slots; per-leg (short) notional = equity/10 at
entry, skipped+counted if below $5 min notional; hedge notional equal-sized ON
TOP of the short notional (gross exposure = 2x sum of open slot notionals when
hedged - reported daily; FLAG if gross/equity exceeds 2.0). Basket-hedge per-leg
notional (= slot/(universe-1)) reported with count below the $5 Bybit minimum -
descriptive deployability check. Continuous daily MTM as in Q18 (token+hedge
daily closes, cost as constant drag, token funding accrued daily; exact net
realized at effective exit = stop day or plan exit). CAGR/maxDD per window.

VERDICT RULE (frozen): S1 = DEPLOYABLE iff the DEV-selected (stop x hedge) cell
on LIVE, at the 1% haircut, achieves ALL of:
  (1) net mean >= +150 bps/trade;
  (2) ZERO liquidations (DEV+LIVE);
  (3) LIVE-window portfolio maxDD better than -15%;
  (4) worst single LIVE trade >= -40% of one slot's notional (cell_net >= -0.40).
Otherwise NOT DEPLOYABLE - report which clause failed. No reinterpretation.

ALSO REPORTED (descriptive): selected cell by-year table; DEV/LIVE trade counts;
edge cost vs the no-stop reference (and vs Q18's numbers); stop-out rate and
forgone bounce per cell; hedge-arm comparison; day-clustered (entry-date) SE of
the per-trade mean - unlock windows cluster/overlap; 3%-haircut sensitivity;
funding climate-fallback share; daily returns parquet of the selected cell for
portfolio-glue reuse.

HONESTY CAVEATS (frozen, restated in output): survivorship universe (2026
survivors - for a short mostly conservative, but squeeze tails on delisted names
are unobserved, cuts both ways); hourly-high stop checks are optimistic vs tick
wicks (3%-worse sensitivity quantifies); daily MTM understates intra-day DD
between stop checks; DefiLlama self-reported supply denominators; overlapping
events -> day-clustered SE reported; hedge MTM path ignores hedge funding until
realization (exact at exit, as in Q18).

Artifacts:
  research/data/unlocks/results_s1_realism.json      (all grids, both windows)
  research/data/unlocks/daily_returns_s1_realistic.parquet (selected cell, $5k)
Full results JSON printed at the end. Usage: python unlock_s1_realism.py
=============================================================================
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import unlock_trade_sim as uts  # REUSE Q18 loaders / conventions / constants

REPO_ROOT = Path(__file__).resolve().parents[2]
KL_DIR = uts.KL_DIR
FUND_DIR = uts.FUND_DIR
UNLOCK_DIR = uts.UNLOCK_DIR
COST_RT = uts.COST_RT
DEV_CUT = uts.DEV_CUT
DAY_NS = uts.DAY_NS

THR = 0.01
ENTRY_OFF = 30
STOP_LEVELS = [0.25, 0.35, 0.50]          # candidate stops (short-leg adverse)
HEDGES = ["basket", "btc", "none"]        # "none" = reference only
HAIRCUTS = [0.01, 0.03]                   # verdict at 0.01; 0.03 = sensitivity
LIQ_ADVERSE = 0.985                       # 1x isolated short forced-close level
LIQ_PENALTY = 0.01
SLOTS = 10
EQUITY0 = 5000.0
MIN_NOTIONAL = 5.0
H1 = pd.Timedelta(hours=1)
D1 = pd.Timedelta(days=1)

# --------------------------------------------------------------------------- #
# Hourly cache (high for stop scan, close for hedge exit at stop)
# --------------------------------------------------------------------------- #
_hourly: dict[str, pd.DataFrame | None] = {}


def hourly(pair: str) -> pd.DataFrame | None:
    if pair in _hourly:
        return _hourly[pair]
    p = KL_DIR / f"{pair}.parquet"
    if not p.exists():
        _hourly[pair] = None
        return None
    k = pd.read_parquet(p, columns=["open_time", "high", "close"])
    k.index = pd.to_datetime(k["open_time"], unit="ms")
    df = pd.DataFrame({"high": k["high"].resample("1h").max(),
                       "close": k["close"].resample("1h").last()}).dropna()
    _hourly[pair] = df
    return df


def scan_trigger(pair: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp,
                 level_price: float):
    """First hourly bar label in the hold whose high >= level_price, else None.
    Hold = bars labelled [entry_date+1d, exit_date+1d), i.e. strictly after the
    entry daily close up to the exit daily close."""
    h = hourly(pair)
    if h is None:
        return None
    seg = h["high"].loc[entry_date + D1: exit_date + D1 - H1]
    if seg.empty:
        return None
    hit = seg[seg.to_numpy() >= level_price]
    return None if hit.empty else hit.index[0]


def hourly_close_asof(pair: str, label: pd.Timestamp):
    h = hourly(pair)
    if h is None or h.empty:
        return None
    idx = h.index.searchsorted(label, side="right") - 1
    if idx < 0:
        return None
    return float(h["close"].iloc[idx])


# --------------------------------------------------------------------------- #
# Hedge legs (plan exit replicates Q18 long_leg/basket_long exactly)
# --------------------------------------------------------------------------- #
def hedge_leg_net(pair, entry_date, exit_date, stop_label):
    """Long hedge leg net per unit notional. Entry = daily close of entry_date.
    Exit = hourly close at the stop bar when stop_label given, else daily close
    of exit_date. Pays own COST_RT and funding (long pays +funding)."""
    d = uts.daily(pair)
    if d is None or entry_date not in d[0].index:
        return None
    e = float(d[0].loc[entry_date])
    if e <= 0:
        return None
    entry_ns = uts.close_ns(entry_date)
    if stop_label is None:
        if exit_date not in d[0].index:
            return None
        x = float(d[0].loc[exit_date])
        exit_ns = uts.close_ns(exit_date)
    else:
        x = hourly_close_asof(pair, stop_label)
        if x is None:
            return None
        exit_ns = int((stop_label + H1).value)
    if x <= 0:
        return None
    fund, fb = uts.funding_with_fallback(pair, entry_ns, exit_ns,
                                         entry_date.year, 1.0)
    return {"net": x / e - 1.0 - COST_RT - fund, "fb": fb}


def basket_net(universe, entry_date, exit_date, stop_label, exclude):
    nets, fb = [], 0
    for p in universe:
        if p == exclude:
            continue
        r = hedge_leg_net(p, entry_date, exit_date, stop_label)
        if r is not None:
            nets.append(r["net"])
            fb += int(r["fb"])
    if not nets:
        return None
    return {"net": float(np.mean(nets)), "n": len(nets), "fb": int(fb > 0)}


# --------------------------------------------------------------------------- #
# Trade construction (Q18 funnel: dedup -> liquidity -> fill) + trigger scan
# --------------------------------------------------------------------------- #
def build_trades(ev, universe):
    ded = uts.dedup_events(ev, THR)
    trades = []
    n_liqfail = n_fillfail = 0
    for _, r in ded.iterrows():
        d = r["ed"]
        entry = d - pd.Timedelta(days=ENTRY_OFF)
        exitd = d - D1
        if not uts.liquidity_ok(r["pair"], entry):
            n_liqfail += 1
            continue
        sh = uts.short_leg(r["pair"], entry, exitd)  # plan-exit short (Q18)
        if sh is None:
            n_fillfail += 1
            continue
        e = float(uts.daily(r["pair"])[0].loc[entry])
        stops = {s: scan_trigger(r["pair"], entry, exitd, e * (1 + s))
                 for s in STOP_LEVELS}
        liq = scan_trigger(r["pair"], entry, exitd, e * (1 + LIQ_ADVERSE))
        trades.append({"pair": r["pair"], "entry": entry, "exit": exitd,
                       "event": d, "win": "DEV" if d < DEV_CUT else "LIVE",
                       "e": e, "short": sh, "stops": stops, "liq": liq})
    funnel = {"dedup": int(len(ded)), "liq_fail": n_liqfail,
              "fill_fail": n_fillfail, "filled": len(trades),
              "filled_DEV": sum(t["win"] == "DEV" for t in trades),
              "filled_LIVE": sum(t["win"] == "LIVE" for t in trades)}
    return trades, funnel


def short_net_for(t, stop, haircut):
    """(net, exit_label, stopped, liquidated, fb) for the short leg under one
    stop arm. stop=None -> no-stop arm with liquidation model live."""
    entry_ns = uts.close_ns(t["entry"])
    if stop is not None and t["stops"][stop] is not None:
        lab = t["stops"][stop]
        gross = -(stop + haircut + stop * haircut)   # fill = e*(1+s)*(1+h)
        fund, fb = uts.funding_with_fallback(t["pair"], entry_ns,
                                             int((lab + H1).value),
                                             t["entry"].year, 1.0)
        return gross - COST_RT + fund, lab, True, False, fb
    if stop is None and t["liq"] is not None:
        lab = t["liq"]
        gross = -(LIQ_ADVERSE + LIQ_PENALTY)
        fund, fb = uts.funding_with_fallback(t["pair"], entry_ns,
                                             int((lab + H1).value),
                                             t["entry"].year, 1.0)
        return gross - COST_RT + fund, lab, False, True, fb
    return t["short"]["net"], None, False, False, t["short"]["fb"]


def eval_cell(trades, universe, stop, hedge, haircut, hedge_cache):
    rows = []
    for t in trades:
        snet, lab, stopped, liq, fb = short_net_for(t, stop, haircut)
        if hedge == "none":
            hnet = 0.0
        else:
            key = (hedge, t["pair"], t["entry"].value,
                   None if lab is None else lab.value)
            if key not in hedge_cache:
                hr = (hedge_leg_net("BTCUSDT", t["entry"], t["exit"], lab)
                      if hedge == "btc"
                      else basket_net(universe, t["entry"], t["exit"], lab,
                                      t["pair"]))
                hedge_cache[key] = hr["net"] if hr else 0.0
            hnet = hedge_cache[key]
        exit_eff = lab.normalize() if lab is not None else t["exit"]
        rows.append({"pair": t["pair"], "entry": t["entry"], "exit": t["exit"],
                     "exit_eff": exit_eff, "event": t["event"], "win": t["win"],
                     "short_net": snet, "hnet": hnet, "cell_net": snet + hnet,
                     "stopped": stopped, "liquidated": liq, "fb": fb})
    return rows


def clustered_se(vals, keys):
    """Entry-date-clustered SE of the mean (approx: SE across cluster means)."""
    if len(vals) < 2:
        return None
    g = pd.Series(vals).groupby(pd.Series(keys)).mean()
    if len(g) < 2:
        return None
    return float(g.std(ddof=1) / np.sqrt(len(g)))


def window_stats(rows, ref_rows=None):
    """Per-window stats; ref_rows = same-hedge NO-STOP rows (index-aligned) for
    the forgone-bounce metric."""
    out = {}
    for win in ("DEV", "LIVE"):
        idx = [i for i, r in enumerate(rows) if r["win"] == win]
        w = [rows[i] for i in idx]
        nets = np.array([r["cell_net"] for r in w])
        stopped = [i for i in idx if rows[i]["stopped"]]
        forgone = (float(np.mean([ref_rows[i]["cell_net"] - rows[i]["cell_net"]
                                  for i in stopped]))
                   if ref_rows is not None and stopped else None)
        out[win] = {
            "n": len(w),
            "net_mean": float(nets.mean()) if len(nets) else None,
            "net_med": float(np.median(nets)) if len(nets) else None,
            "win_rate": float((nets > 0).mean()) if len(nets) else None,
            "worst_trade": float(nets.min()) if len(nets) else None,
            "stop_rate": (len(stopped) / len(w)) if w else None,
            "n_liq": int(sum(r["liquidated"] for r in w)),
            "forgone_bounce_mean": forgone,
            "clustered_se": clustered_se(nets.tolist(),
                                         [str(r["entry"].date()) for r in w]),
            "fb_share": (float(np.mean([r["fb"] for r in w])) if w else None),
        }
    return out


# --------------------------------------------------------------------------- #
# $5k portfolio (10 slots, min notional, gross-exposure tracking; Q18 MTM)
# --------------------------------------------------------------------------- #
def portfolio_5k(rows, universe, hedge):
    rows = sorted((dict(r) for r in rows), key=lambda r: r["entry"])
    if not rows:
        return pd.DataFrame(), {}
    start = min(r["entry"] for r in rows)
    end = max(r["exit_eff"] for r in rows)
    days = pd.date_range(start, end, freq="1D")

    def path_for(r):
        seg = uts.daily(r["pair"])[0].loc[r["entry"]:r["exit_eff"]]
        e0 = float(seg.iloc[0])
        base = 1.0 - seg / e0
        if hedge == "btc":
            hseg = uts.daily("BTCUSDT")[0].loc[r["entry"]:r["exit_eff"]]
            h0 = float(hseg.iloc[0])
            hbase = (hseg / h0 - 1.0).reindex(seg.index, method="ffill").fillna(0.0)
        elif hedge == "basket":
            hb = []
            for p in universe:
                if p == r["pair"]:
                    continue
                cp = uts.daily(p)
                if cp is None:
                    continue
                s2 = cp[0].loc[r["entry"]:r["exit_eff"]]
                if len(s2) < 1:
                    continue
                hb.append((s2 / float(s2.iloc[0]) - 1.0)
                          .reindex(seg.index, method="ffill"))
            hbase = (pd.concat(hb, axis=1).mean(axis=1).fillna(0.0)
                     if hb else pd.Series(0.0, index=seg.index))
        else:
            hbase = pd.Series(0.0, index=seg.index)
        fundp = pd.Series(0.0, index=seg.index)
        for dt in seg.index:
            fundp.loc[dt] = uts.funding_sum(r["pair"], uts.close_ns(r["entry"]),
                                            uts.close_ns(dt))
        frac = base + hbase + fundp - COST_RT * (2 if hedge != "none" else 1)
        frac.loc[r["exit_eff"]] = r["cell_net"]   # exact realization
        return frac

    paths = {id(r): path_for(r) for r in rows}
    by_entry = {}
    for r in rows:
        by_entry.setdefault(r["entry"], []).append(r)

    cash = EQUITY0
    open_pos = []
    prev_eq = EQUITY0
    recs = []
    skips = min_not_skips = 0
    max_open = 0
    max_gross_ratio = 0.0
    days_gross_gt2 = 0
    basket_leg_min = None
    basket_legs_lt5 = 0
    n_basket = max(1, len(universe) - 1)
    lev = 2.0 if hedge != "none" else 1.0

    def mark(p, day):
        fr = paths[id(p)]
        v = fr.loc[day] if day in fr.index else fr.iloc[-1]
        return p["cap"] * (1 + v)

    for day in days:
        for r in by_entry.get(day, []):
            if len(open_pos) >= SLOTS:
                skips += 1
                continue
            eq_now = cash + sum(mark(p, day) for p in open_pos)
            cap = eq_now / SLOTS
            if cap < MIN_NOTIONAL:
                min_not_skips += 1
                continue
            if hedge == "basket":
                leg = cap / n_basket
                basket_leg_min = leg if basket_leg_min is None else min(basket_leg_min, leg)
                if leg < MIN_NOTIONAL:
                    basket_legs_lt5 += 1
            cash -= cap
            r["cap"] = cap
            open_pos.append(r)
        still = []
        for p in open_pos:
            if day >= p["exit_eff"]:
                cash += mark(p, day)
            else:
                still.append(p)
        open_pos = still
        max_open = max(max_open, len(open_pos))
        eq = cash + sum(mark(p, day) for p in open_pos)
        gross = lev * sum(p["cap"] for p in open_pos)
        ratio = gross / eq if eq > 0 else np.inf
        max_gross_ratio = max(max_gross_ratio, ratio)
        if ratio > 2.0 + 1e-9:
            days_gross_gt2 += 1
        recs.append({"date": day, "ret": eq / prev_eq - 1.0 if prev_eq else 0.0,
                     "equity": eq, "n_open": len(open_pos),
                     "gross_ratio": ratio})
        prev_eq = eq
    df = pd.DataFrame(recs)
    m = uts._port_metrics(df)
    m["slot_skips"] = int(skips)
    m["min_notional_skips"] = int(min_not_skips)
    m["max_concurrent"] = int(max_open)
    m["final_equity"] = float(df["equity"].iloc[-1])
    m["max_gross_over_equity"] = float(max_gross_ratio)
    m["days_gross_gt_2x"] = int(days_gross_gt2)
    m["basket_leg_min_notional"] = (None if basket_leg_min is None
                                    else float(basket_leg_min))
    m["basket_trades_with_leg_lt_5usd"] = int(basket_legs_lt5)
    return df, m


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ev = pd.read_parquet(uts.EVENTS_PATH)
    ev["ed"] = pd.to_datetime(ev["event_date"])
    kl_pairs = {p.stem for p in KL_DIR.glob("*.parquet")}
    fu_pairs = {p.stem for p in FUND_DIR.glob("*.parquet")}
    ev = ev[ev["pair"].isin(kl_pairs) & ev["pair"].isin(fu_pairs)].copy()
    universe = sorted(ev["pair"].unique())
    print(f"[setup] universe pairs = {len(universe)}", flush=True)

    # per-year climate daily funding (REUSED Q18 recipe -> fills uts.CLIMATE_DAILY)
    print("[setup] building per-year climate funding ...", flush=True)
    daily_sums: dict[int, list] = {}
    for p in fu_pairs:
        f = pd.read_parquet(FUND_DIR / f"{p}.parquet",
                            columns=["fundingTime", "fundingRate"])
        ts = pd.to_datetime(f["fundingTime"], unit="ms")
        ds = pd.Series(f["fundingRate"].to_numpy(np.float64), index=ts).resample("1D").sum()
        for yr, g in ds.groupby(ds.index.year):
            daily_sums.setdefault(int(yr), []).append(g)
    for yr, lst in daily_sums.items():
        uts.CLIMATE_DAILY[yr] = float(pd.concat(lst).mean())

    print("[trades] building S1 trades + trigger scan ...", flush=True)
    trades, funnel = build_trades(ev, universe)
    print(f"  funnel: {funnel}", flush=True)

    # ---------------- grid ----------------
    hedge_cache: dict = {}
    grid = {}          # key -> {"h0.01": stats, "h0.03": stats}
    rows_store = {}    # (stop, hedge, haircut) -> rows
    for hedge in HEDGES:
        ref_rows = eval_cell(trades, universe, None, hedge, HAIRCUTS[0], hedge_cache)
        rows_store[(None, hedge, HAIRCUTS[0])] = ref_rows
        rows_store[(None, hedge, HAIRCUTS[1])] = ref_rows  # haircut-invariant
        key = f"stopnone_{hedge}"
        st = window_stats(ref_rows, ref_rows)
        grid[key] = {f"h{h}": st for h in HAIRCUTS}
        for stop in STOP_LEVELS:
            key = f"stop{int(stop*100)}_{hedge}"
            grid[key] = {}
            for h in HAIRCUTS:
                rws = eval_cell(trades, universe, stop, hedge, h, hedge_cache)
                rows_store[(stop, hedge, h)] = rws
                grid[key][f"h{h}"] = window_stats(rws, ref_rows)
        print(f"[grid] hedge={hedge} done", flush=True)

    # ---------------- DEV selection (frozen rule) ----------------
    h0 = f"h{HAIRCUTS[0]}"
    cands = [(s, hg) for s in STOP_LEVELS for hg in ("basket", "btc")]
    minn, relax = 30, False
    elig = [(s, hg) for s, hg in cands
            if grid[f"stop{int(s*100)}_{hg}"][h0]["DEV"]["n"] >= minn
            and grid[f"stop{int(s*100)}_{hg}"][h0]["DEV"]["n_liq"] == 0]
    if not elig:
        minn, relax = 20, True
        elig = [(s, hg) for s, hg in cands
                if grid[f"stop{int(s*100)}_{hg}"][h0]["DEV"]["n"] >= minn
                and grid[f"stop{int(s*100)}_{hg}"][h0]["DEV"]["n_liq"] == 0]
    # argmax DEV net mean; tie-break wider stop
    sel_stop, sel_hedge = max(
        elig, key=lambda c: (grid[f"stop{int(c[0]*100)}_{c[1]}"][h0]["DEV"]["net_mean"], c[0]))
    sel_key = f"stop{int(sel_stop*100)}_{sel_hedge}"
    print(f"[select] DEV selected: {sel_key} (relaxed_to_20={relax})", flush=True)

    sel_rows = rows_store[(sel_stop, sel_hedge, HAIRCUTS[0])]
    sel_rows_h3 = rows_store[(sel_stop, sel_hedge, HAIRCUTS[1])]
    nostop_rows = rows_store[(None, sel_hedge, HAIRCUTS[0])]

    # ---------------- portfolios ($5k) ----------------
    print("[port] selected cell ...", flush=True)
    df_sel, m_sel = portfolio_5k(sel_rows, universe, sel_hedge)
    print("[port] selected cell @3% haircut ...", flush=True)
    df_h3, m_h3 = portfolio_5k(sel_rows_h3, universe, sel_hedge)
    print("[port] no-stop reference (same hedge) ...", flush=True)
    df_ns, m_ns = portfolio_5k(nostop_rows, universe, sel_hedge)

    # ---------------- verdict (frozen) ----------------
    sl = grid[sel_key][h0]["LIVE"]
    sd = grid[sel_key][h0]["DEV"]
    c1 = bool(sl["net_mean"] is not None and sl["net_mean"] >= 0.015)
    c2 = bool(sd["n_liq"] == 0 and sl["n_liq"] == 0)
    c3 = bool(m_sel.get("maxdd_LIVE") is not None
              and m_sel["maxdd_LIVE"] > -0.15)
    c4 = bool(sl["worst_trade"] is not None and sl["worst_trade"] >= -0.40)
    deployable = c1 and c2 and c3 and c4

    # ---------------- descriptive: by-year, Q18 comparison ----------------
    by_year = {}
    for r in sel_rows:
        y = int(r["event"].year)
        by_year.setdefault(y, []).append(r)
    by_year_tbl = {y: {"n": len(v),
                       "net_mean": float(np.mean([r["cell_net"] for r in v])),
                       "stop_rate": float(np.mean([r["stopped"] for r in v])),
                       "win_rate": float(np.mean([r["cell_net"] > 0 for r in v]))}
                   for y, v in sorted(by_year.items())}

    q18_ref = None
    try:
        with open(UNLOCK_DIR / "results_trade_sim.json") as f:
            q18 = json.load(f)
        q18_ref = {"cell": q18["s1_selected"]["cell"],
                   "DEV_net": q18["s1_selected"]["DEV"]["net_mean"],
                   "LIVE_net": q18["s1_selected"]["LIVE"]["net_mean"],
                   "port_maxdd_DEV": q18["s1_portfolio"].get("maxdd_DEV"),
                   "port_maxdd_LIVE": q18["s1_portfolio"].get("maxdd_LIVE")}
    except Exception as ex:  # pragma: no cover
        q18_ref = {"error": str(ex)}

    def _compact_port(m):
        return {k: (v if k != "monthly" else None) for k, v in m.items()
                if k != "monthly"}

    results = {
        "spec": "unlock_s1_realism v1 (Q20, ledger #22); full frozen spec in module docstring",
        "base_config": {"thr": THR, "entry_off": ENTRY_OFF, "exit": "t-1d",
                        "cost_rt_per_leg": COST_RT, "slots": SLOTS,
                        "equity0": EQUITY0, "min_notional": MIN_NOTIONAL,
                        "stop_levels": STOP_LEVELS, "haircuts": HAIRCUTS,
                        "liq_adverse": LIQ_ADVERSE, "liq_penalty": LIQ_PENALTY,
                        "universe_pairs": len(universe)},
        "funnel": funnel,
        "grid": grid,
        "selection": {"rule": "argmax DEV net_mean among {25,35,50}x{basket,btc}, h=1%, n>=30, 0 DEV liq; tie->wider stop",
                      "selected": sel_key, "relaxed_to_20": relax,
                      "DEV": sd, "LIVE": sl,
                      "LIVE_h3_sensitivity": grid[sel_key][f"h{HAIRCUTS[1]}"]["LIVE"]},
        "portfolio_selected": _compact_port(m_sel),
        "portfolio_selected_monthly": m_sel.get("monthly"),
        "portfolio_selected_h3": _compact_port(m_h3),
        "portfolio_nostop_ref": _compact_port(m_ns),
        "verdict": {
            "rule": "LIVE net>=+150bps AND zero liq DEV+LIVE AND LIVE port maxDD > -15% AND worst LIVE trade >= -40% of slot",
            "clause1_live_net_ge_150bps": {"pass": c1, "value": sl["net_mean"]},
            "clause2_zero_liquidations": {"pass": c2,
                                          "dev_liq": sd["n_liq"], "live_liq": sl["n_liq"]},
            "clause3_live_maxdd_gt_-15pct": {"pass": c3,
                                             "value": m_sel.get("maxdd_LIVE")},
            "clause4_worst_live_trade_ge_-40pct": {"pass": c4,
                                                   "value": sl["worst_trade"]},
            "DEPLOYABLE": deployable,
        },
        "by_year_selected": by_year_tbl,
        "q18_reference": q18_ref,
        "stop_economics": {
            "selected_vs_nostop_DEV": (sd["net_mean"] - grid[f"stopnone_{sel_hedge}"][h0]["DEV"]["net_mean"]),
            "selected_vs_nostop_LIVE": (sl["net_mean"] - grid[f"stopnone_{sel_hedge}"][h0]["LIVE"]["net_mean"]),
            "note": "no-stop reference includes the liquidation model (forced close at +98.5% +1% penalty)",
        },
        "caveats": [
            "survivorship universe (2026 survivors): for a short mostly conservative, but squeeze tails on delisted names unobserved - cuts both ways",
            "stops checked on HOURLY highs - optimistic vs tick wicks; 3%-worse-fill sensitivity reported as the quantification",
            "daily MTM understates intra-day drawdown between stop checks",
            "overlapping/clustered unlock windows - entry-date-clustered SE reported per cell",
            "hedge funding/cost enters daily MTM as constant drag, exact at realization (Q18 convention)",
            "DefiLlama self-reported supply denominators; event_ts is SECONDS, event_date used",
            "hedge-leg (long) liquidation ignored: needs -98.5%, unreachable for basket/BTC",
        ],
    }

    UNLOCK_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNLOCK_DIR / "results_s1_realism.json", "w") as f:
        json.dump(results, f, indent=2, default=uts._json_default)
    if not df_sel.empty:
        df_sel.to_parquet(UNLOCK_DIR / "daily_returns_s1_realistic.parquet",
                          index=False)

    # ---------------- console summary ----------------
    print("\n===== GRID (net bps/trade, h=1%) =====")
    print(f"  {'cell':22s} {'DEVn':>4} {'DEVnet':>8} {'DEVstop%':>8} {'DEVliq':>6} "
          f"{'LIVEn':>5} {'LIVEnet':>8} {'LIVEstop%':>9} {'LIVEliq':>7} {'LIVEworst':>9}")
    for hg in HEDGES:
        for sname in ["stopnone"] + [f"stop{int(s*100)}" for s in STOP_LEVELS]:
            k = f"{sname}_{hg}"
            g = grid[k][h0]
            print(f"  {k:22s} {g['DEV']['n']:>4} {uts._bps(g['DEV']['net_mean']):>8} "
                  f"{(g['DEV']['stop_rate'] or 0)*100:>7.1f}% {g['DEV']['n_liq']:>6} "
                  f"{g['LIVE']['n']:>5} {uts._bps(g['LIVE']['net_mean']):>8} "
                  f"{(g['LIVE']['stop_rate'] or 0)*100:>8.1f}% {g['LIVE']['n_liq']:>7} "
                  f"{uts._bps(g['LIVE']['worst_trade']):>9}")
    print(f"\n  SELECTED: {sel_key} (h=1%)  relaxed20={relax}")
    print(f"    DEV  net={uts._bps(sd['net_mean'])} se~{uts._bps(sd['clustered_se'])} "
          f"n={sd['n']} stop%={(sd['stop_rate'] or 0)*100:.1f} forgone={uts._bps(sd['forgone_bounce_mean'])}")
    print(f"    LIVE net={uts._bps(sl['net_mean'])} se~{uts._bps(sl['clustered_se'])} "
          f"n={sl['n']} stop%={(sl['stop_rate'] or 0)*100:.1f} forgone={uts._bps(sl['forgone_bounce_mean'])}")
    print(f"    LIVE @3% haircut net={uts._bps(grid[sel_key][f'h{HAIRCUTS[1]}']['LIVE']['net_mean'])}")
    print(f"    port $5k: CAGR DEV={uts._r(m_sel.get('cagr_DEV'))} LIVE={uts._r(m_sel.get('cagr_LIVE'))} "
          f"maxDD DEV={uts._r(m_sel.get('maxdd_DEV'))} LIVE={uts._r(m_sel.get('maxdd_LIVE'))}")
    print(f"    gross max={m_sel.get('max_gross_over_equity'):.2f}x "
          f"days>2x={m_sel.get('days_gross_gt_2x')} "
          f"basket_leg_min=${m_sel.get('basket_leg_min_notional') if m_sel.get('basket_leg_min_notional') is not None else 'n/a'} "
          f"legs<$5 trades={m_sel.get('basket_trades_with_leg_lt_5usd')}")
    print("\n===== VERDICT (frozen) =====")
    for ck in ["clause1_live_net_ge_150bps", "clause2_zero_liquidations",
               "clause3_live_maxdd_gt_-15pct", "clause4_worst_live_trade_ge_-40pct"]:
        v = results["verdict"][ck]
        print(f"  {ck}: {'PASS' if v['pass'] else 'FAIL'} ({ {kk: vv for kk, vv in v.items() if kk != 'pass'} })")
    print(f"  Q20 S1 DEPLOYABLE = {deployable}")
    print("\n===== BY-YEAR (selected) =====")
    for y, v in by_year_tbl.items():
        print(f"  {y}: n={v['n']:>3} net={uts._bps(v['net_mean'])} "
              f"stop%={v['stop_rate']*100:.1f} win%={v['win_rate']*100:.1f}")
    print("\nartifacts written to", UNLOCK_DIR)
    print("\n===== FULL RESULTS JSON =====")
    print(json.dumps(results, indent=2, default=uts._json_default))


if __name__ == "__main__":
    main()
