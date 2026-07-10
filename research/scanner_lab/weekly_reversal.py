"""Weekly SHORT-TERM CROSS-SECTIONAL REVERSAL study (one-shot, pre-registered).

QUESTION (never asked before in this lab): does buying last week's biggest
LOSERS pay on crypto perps? The academic short-term reversal / liquidity-
provision premium is strongest at the 1-week horizon; prior work here killed
weekly momentum (long-only alt TSMOM -51%/2y, long-short thin +2.7%/yr).
Reversal is momentum's mirror, so this is a genuinely new hypothesis.

============================ PRE-REGISTERED SPEC ============================
Everything below was fixed BEFORE the first run. The grid is run in full and
reported in full; NOTHING is added to the grid after seeing results.

Panel / universe:
  - Daily close panel resampled from 1h klines (research/data/v3/klines/1h),
    reusing weekly_tsmom.load_daily. Daily quote_volume = sum of 1h.
  - Eligible universe each Monday = liquidity top-75 by trailing 30d median
    daily quote_volume (same liquidity gate as weekly_tsmom).
  - Rebalance: Monday 00:00 UTC daily close. Costs on drifted weights, same
    mechanism as weekly_tsmom.py (turnover = sum|w_new - w_drifted|).

Signal (trailing 7d return), causal, decided at close of rebalance day t:
  - skip OFF: 7d return over days t-7..t-1  = px[t-1]/px[t-8] - 1
  - skip ON : 7d return over days t-8..t-2  = px[t-2]/px[t-9] - 1  (drops the
    single most recent day t-1 as a bid-ask-bounce / microstructure guard)
  Both use only prices strictly before the rebalance close -> fully causal.

Declared grid (2x2x2x2 = 16 cells, all run):
  - portfolio : long bottom-N losers, N in {10, 20}
  - weighting : {equal-weight; inverse-vol capped 10%}  (invvol = 1/sigma(20d
    daily rets), norm, clip 0.10, renorm w/ cap-shortfall-to-cash, per
    weekly_tsmom.build_weights)
  - skip-day  : {off; on}
  - side      : {long-only; long-short = long bottom-N + short top-N, 50/50
    capital per leg (per weekly_tsmom_short.py)}

Funding (perps): realized 8h funding from research/data/perp/funding, applied
  with weekly_tsmom_short.py's sign convention to EVERY held leg:
  net = (r_long - fund_long) - r_short + fund_short  -> longs PAY positive
  funding / receive negative; shorts RECEIVE positive / pay negative. NB this
  differs from weekly_tsmom.py (which omitted funding on its long leg); it is
  the physically correct perp accounting. 2025-26 funding is NEGATIVE, i.e. a
  TAILWIND for the long-loser leg and a HEADWIND for shorts.

Costs: PRIMARY 25 bps per side x turnover (drifted). Sensitivity: 10 bps per
  side, run only on the DEV-best cell. Reversal turnover is EXPECTED high
  (~80-100%/wk one leg); turnover reported per cell.

Controls (declared): (a) EW basket of the eligible top-75, weekly rebalanced;
  (b) BTC buy-and-hold; (c) MIRROR diagnostic = long TOP-N WINNERS (= weekly
  momentum), long-only EW skip-off, N in {10,20} -- prior TSMOM says this must
  be ~flat/negative; it is a sanity check that the panel/plumbing is sane.

Metrics per cell: total return, CAGR, annualized Sharpe (WEEKLY returns,
  x sqrt(52)), maxDD (daily), worst week, avg weekly turnover -- for windows:
    full_4y  2022-06-01 .. panel end (2026-07-06)
    DEV      2022-06-01 .. 2024-12-31
    HOLDOUT  2025-01-01 .. panel end
  plus by-calendar-year for the DEV-best cell.

============================= HONESTY PROTOCOL =============================
  - DEV-best cell is picked on DEV ONLY (by ann. weekly Sharpe, tie-break
    total return). The VERDICT is that one cell's HOLDOUT number. The full
    16-cell grid is reported on both windows as pure diagnostic.
  - *** SURVIVORSHIP BIAS -- THE STUDY'S MAIN THREAT TO VALIDITY ***
    The 149-coin universe was chosen in 2026: 68/149 coins list only after
    2024-01. Coins that crashed and DIED are absent. For a LONG-loser
    strategy this bias is SEVERE and direction-known: we buy last week's
    biggest crashers KNOWING (by construction) they survived and mean-
    reverted. Dead crashers -- the ones that would have bankrupted the trade
    -- are silently excluded. Early years (2022-23) are the most inflated.
    Therefore: treat every positive LONG number, especially pre-2025 and
    especially DEV, as an UPPER bound, not an estimate. HOLDOUT (2025+, near-
    contemporaneous universe) is the least-biased read. The SHORT leg carries
    the opposite (favorable-to-us) bias: dead coins we'd have shorted are
    missing, so short profit is UNDER-stated.

Usage: python weekly_reversal.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402
from weekly_tsmom import load_daily, bench_btc, bench_ew  # noqa: E402
from carry_study import load_funding  # noqa: E402

ART_DIR = REPO_ROOT / "research" / "data" / "reversal"

FULL_START = "2022-06-01"
DEV_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
LIQ_TOP = 75
COST_PRIMARY = 25.0   # bps per side (pre-registered)
COST_SENS = 10.0      # bps per side, DEV-best only


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def rank_weights(px: pd.DataFrame, dv: pd.DataFrame, t: pd.Timestamp, n: int,
                 weighting: str, skip: bool, pick: str) -> pd.Series:
    """Causal cross-sectional pick at close of day t.

    pick='bottom' -> N biggest losers (reversal long / mirror short-leg winners
    handled by caller); pick='top' -> N biggest winners.
    """
    hist = px.loc[:t]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    liq = dv.loc[:t].tail(30).median()
    live = liq.dropna().nlargest(LIQ_TOP).index
    h = hist[live]
    if skip:                                   # days t-8..t-2
        sig = h.iloc[-3] / h.iloc[-10] - 1.0
    else:                                      # days t-7..t-1
        sig = h.iloc[-2] / h.iloc[-9] - 1.0
    sig = sig.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sig) < n:
        return pd.Series(dtype=float)
    sel = (sig.nsmallest(n) if pick == "bottom" else sig.nlargest(n)).index
    if weighting == "ew":
        return pd.Series(1.0 / n, index=sel)
    vol = hist[sel].pct_change().tail(20).std()
    vol = vol[(vol > 0) & vol.notna()]
    if vol.empty:
        return pd.Series(dtype=float)
    w = 1.0 / vol
    w = w / w.sum()
    w = w.clip(upper=0.10)
    return w / max(w.sum(), 1.0)               # cap shortfall -> cash


def _drift(w: pd.Series, day_ret_row: pd.Series, legret: float) -> pd.Series:
    if not len(w):
        return w
    gross = w * (1.0 + day_ret_row.reindex(w.index).fillna(0.0))
    return gross / max(1.0 + legret, 1e-9)


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def metrics(eq: pd.Series, turnover_log: list[float]) -> dict:
    dr_w = eq.resample("W-SUN").last().pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1.0).min()
    term = float(eq.iloc[-1])
    return {
        "total_return": round(term - 1.0, 4),
        "cagr": round(float(max(term, 1e-9) ** (1.0 / years) - 1.0), 4),
        "sharpe_wk": round(float(dr_w.mean() / dr_w.std() * np.sqrt(52)), 3)
        if dr_w.std() > 0 else None,
        "max_dd": round(float(dd), 4),
        "worst_week": round(float(dr_w.min()), 4) if len(dr_w) else None,
        "turnover_wk": round(float(np.mean(turnover_log)), 3) if turnover_log else None,
    }


def run(px: pd.DataFrame, dv: pd.DataFrame, fund_daily: pd.DataFrame, rets: pd.DataFrame,
        n: int, weighting: str, skip: bool, side: str, cost_bps: float,
        start: str, end: str, pick: str = "bottom",
        return_equity: bool = False) -> dict:
    days = px.loc[start:end].index
    wl = pd.Series(dtype=float)   # long weights (positive)
    ws = pd.Series(dtype=float)   # short weights (positive numbers = short notional)
    eq, idx, turnover_log = [1.0], [days[0]], []
    for d in days[1:]:
        row = rets.loc[d]
        rl = float((row.reindex(wl.index).fillna(0.0) * wl).sum()) if len(wl) else 0.0
        rs = float((row.reindex(ws.index).fillna(0.0) * ws).sum()) if len(ws) else 0.0
        frow = fund_daily.loc[d] if d in fund_daily.index else None
        fl = float((frow.reindex(wl.index).fillna(0.0) * wl).sum()) if (len(wl) and frow is not None) else 0.0
        fs = float((frow.reindex(ws.index).fillna(0.0) * ws).sum()) if (len(ws) and frow is not None) else 0.0
        r = (rl - fl) - rs + fs      # longs pay funding, shorts receive
        cost = 0.0
        if d.dayofweek == 0:
            wl_d = _drift(wl, row, rl)
            ws_d = _drift(ws, row, rs)
            nl = rank_weights(px, dv, d, n, weighting, skip, pick)
            if side == "ls":
                short_pick = "top" if pick == "bottom" else "bottom"
                nsr = rank_weights(px, dv, d, n, weighting, skip, short_pick) * 0.5
                nl = nl * 0.5
            else:
                nsr = pd.Series(dtype=float)
            turn = 0.0
            for old, new in ((wl_d, nl), (ws_d, nsr)):
                ix = old.index.union(new.index)
                turn += float((new.reindex(ix, fill_value=0.0)
                               - old.reindex(ix, fill_value=0.0)).abs().sum())
            cost = turn * cost_bps / 1e4
            turnover_log.append(turn)
            wl, ws = nl, nsr
        step = r - cost
        eq.append(eq[-1] * (1.0 + step))
        idx.append(d)
    s = pd.Series(eq, index=pd.DatetimeIndex(idx))
    out = metrics(s, turnover_log)
    if return_equity:
        out["_equity"] = s
    return out


def cell_id(n, weighting, skip, side) -> str:
    return f"N{n}_{weighting}_{'skipon' if skip else 'skipoff'}_{side}"


def grid_cells():
    for n in (10, 20):
        for weighting in ("ew", "invvol"):
            for skip in (False, True):
                for side in ("long", "ls"):
                    yield dict(n=n, weighting=weighting, skip=skip, side=side)


# --------------------------------------------------------------------------- #
def main() -> None:
    px, dv = load_daily(REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h")
    rets = px.pct_change()
    fr = load_funding()
    fund_daily = fr.resample("1D").sum().reindex(px.index).fillna(0.0)
    end = str(px.index[-1].date())
    print(f"panel {px.shape} ({px.index[0].date()}..{end}); funding {fr.shape}")

    windows = [("full_4y", FULL_START, end),
               ("DEV", FULL_START, DEV_END),
               ("HOLDOUT", HOLDOUT_START, end)]

    results = {"grid": [], "controls": [], "mirror": [], "dev_best": {}}

    # ---- controls: BTC + EW basket (reuse weekly_tsmom benchmarks) ----
    for wname, ws_, we_ in windows:
        b = bench_btc(px, ws_, we_)
        e = bench_ew(px, dv, ws_, we_, liq_top=LIQ_TOP)
        results["controls"].append({"window": wname, **b})
        results["controls"].append({"window": wname, **e})
        print(f"[{wname}] BTC total={b['total_return']:+.2%} shp={b['sharpe']}  "
              f"EW total={e['total_return']:+.2%} shp={e['sharpe']}")

    # ---- MIRROR (momentum sanity): long top-N winners, EW, skip-off, long-only
    for n in (10, 20):
        for wname, ws_, we_ in windows:
            m = run(px, dv, fund_daily, rets, n=n, weighting="ew", skip=False,
                    side="long", cost_bps=COST_PRIMARY, start=ws_, end=we_, pick="top")
            results["mirror"].append({"window": wname, "cell": f"MIRROR_top{n}_ew_skipoff_long", **m})

    # ---- full grid (primary cost) ----
    dev_scores = {}
    for cfg in grid_cells():
        cid = cell_id(**cfg)
        for wname, ws_, we_ in windows:
            m = run(px, dv, fund_daily, rets, cost_bps=COST_PRIMARY,
                    start=ws_, end=we_, **cfg)
            results["grid"].append({"window": wname, "cell": cid, **cfg, **m})
            if wname == "DEV":
                dev_scores[cid] = (m.get("sharpe_wk") or -1e9, m["total_return"], cfg)
        print(f"  {cid:>28s} done")

    # ---- pick DEV-best (Sharpe, tie-break total) ----
    best_cid = max(dev_scores, key=lambda k: (dev_scores[k][0], dev_scores[k][1]))
    best_cfg = dev_scores[best_cid][2]
    # by-year over full range + holdout verdict + 10bps sensitivity
    full_run = run(px, dv, fund_daily, rets, cost_bps=COST_PRIMARY,
                   start=FULL_START, end=end, return_equity=True, **best_cfg)
    eqf = full_run.pop("_equity")
    yr = eqf.resample("YE").last()
    by_year = {}
    prev = 1.0
    # start-of-2022 base is eqf.iloc[0]; compute calendar-year returns
    for y in sorted({d.year for d in eqf.index}):
        seg = eqf[eqf.index.year == y]
        base = eqf[eqf.index < seg.index[0]]
        b0 = float(base.iloc[-1]) if len(base) else 1.0
        by_year[str(y)] = round(float(seg.iloc[-1] / b0 - 1.0), 4)
    holdout = {r["cell"]: r for r in results["grid"]
               if r["window"] == "HOLDOUT" and r["cell"] == best_cid}[best_cid]
    dev = {r["cell"]: r for r in results["grid"]
           if r["window"] == "DEV" and r["cell"] == best_cid}[best_cid]
    sens10 = run(px, dv, fund_daily, rets, cost_bps=COST_SENS,
                 start=HOLDOUT_START, end=end, **best_cfg)
    results["dev_best"] = {
        "cell": best_cid, "cfg": best_cfg,
        "DEV": {k: dev[k] for k in ("total_return", "cagr", "sharpe_wk", "max_dd", "worst_week", "turnover_wk")},
        "HOLDOUT_verdict": {k: holdout[k] for k in ("total_return", "cagr", "sharpe_wk", "max_dd", "worst_week", "turnover_wk")},
        "HOLDOUT_cost10bps": sens10,
        "by_year_full": by_year,
    }
    print(f"\nDEV-best = {best_cid}")
    print(f"  DEV     : {results['dev_best']['DEV']}")
    print(f"  HOLDOUT : {results['dev_best']['HOLDOUT_verdict']}")
    print(f"  HOLDOUT 10bps: {sens10}")
    print(f"  by_year : {by_year}")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "weekly cross-sectional reversal; pre-registered, see module docstring",
        "windows": {"full_4y": [FULL_START, end], "DEV": [FULL_START, DEV_END],
                    "HOLDOUT": [HOLDOUT_START, end]},
        "cost_primary_bps_per_side": COST_PRIMARY,
        **results,
    }
    (ART_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nartifacts -> {ART_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
