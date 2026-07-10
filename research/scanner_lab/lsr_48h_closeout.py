"""GLOBAL account long/short ratio @ 48h -> tradable contrarian sim: CLOSE-OUT.

=========================== PRE-REGISTERED SPEC ============================
Frozen 2026-07-09 BEFORE the first run. This is a SINGLE declared spec: NO
grid, NO tuning, NO variant search. The bucket-study shape is already known
(see lsr_study.py) -- we are only PRICING the tradable version at 48h to close
a footnote left by that study and retire this idea for good.

PRIOR EVIDENCE (stated honestly, from lsr_study.py results):
  NO positioning ratio cleared the deployment bar. The GLOBAL account
  long/short ratio (column `count_long_short_ratio` in metrics_5m) was the ONE
  footnote: a consistent CONTRARIAN tilt -- Q5-Q1 forward-return spread
  -46 bps/24h on DEV, -27 bps HOLDOUT, sign-stable at all horizons, and
  -91 bps at 48h on DEV. Sign-stability is real; magnitude is small and this
  is a gross bucket spread (no costs, no funding, no execution). The prior
  therefore EXPECTS this close-out to return NO -- a small gross tilt does not
  survive 25 bps round-trip + funding drag on the short leg. We price it to
  be definitive, not because we expect a pass.

DATA (UTC): metrics_5m/{PAIR}.parquet (5m, col count_long_short_ratio),
  v3/klines/1h/{PAIR}.parquet (spot 1h), perp/funding/{PAIR}.parquet.
  Universe: universe.load_universe() (149 coins).

SIGNAL (per symbol, on the 1h grid; reuses lsr_study.py machinery):
  - ratio = count_long_short_ratio resampled 5m->1h last(), reindexed onto a
    complete hourly grid spanning klines, ffill(limit=12h) to bridge tiny gaps.
  - pctrank = ratio.rolling(720h=30d, min_periods=360h=15d).rank(pct=True)
    -> point-in-time trailing 30d percentile (min 15d history).
  - liquidity: daily quote_volume sum -> rolling(30d).median() > $1M,
    ffill to 1h; observations failing this are dropped.

EVENTS (contrarian, DIRECTION PRE-DECLARED from the known Q5-Q1<0 shape):
  - LONG  event: pctrank crosses BELOW 0.05 (prev>=0.05, cur<0.05).
                 crowd least long -> fade -> position = +1 (long).
  - SHORT event: pctrank crosses ABOVE 0.95 (prev<=0.95, cur>0.95).
                 crowd most long -> fade -> position = -1 (short).
  - 24h per-symbol cooldown PER SIDE (long and short have independent timers).

TRADE: entry = next 1h open; hold 48h to close (exit = close of entry_bar+47).
  - costs 25 bps round-trip (PRIMARY); 10 bps RT reported as maker-in
    sensitivity.
  - realized funding applied to BOTH legs: funding_pnl = -pos * sum(fundingRate
    over [entry_ts, exit_ts)); longs pay when funding>0 (crowd long), shorts
    receive. (loader pattern from carry_study.py; window convention identical
    to lsr_study.py sim.)

REPORT: LONG side, SHORT side, and combined 50/50 -- n, net mean, median, win,
  p5; DEV (<2025-01-01) vs 2025-26 (>=2025-01-01); by-year. 15-slot portfolio
  (liqrev_v2.portfolio pattern, 1/15 equity/slot, slot frees at exit) for each
  side + combined. RANDOM-BARS control: same liquidity filter, sampled liquid
  bars every 4h, 48h hold, both directions, 25 bps + funding -> drift baseline.

PRE-REGISTERED VERDICT RULE: DEPLOYABLE iff a side clears net >= +25 bps/trade
  (0.0025) in BOTH DEV and 2025-26 with the SAME sign, AND beats the
  same-direction random control by >= 25 bps. Else NO (footnote closed).

Live shadow (small size) remains the final validator regardless of outcome.
Usage: python lsr_48h_closeout.py   (prints progress; ~few min)
=============================================================================
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

METRICS_DIR = REPO_ROOT / "research" / "data" / "perp" / "metrics_5m"
KL_DIR = REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h"
FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
ART_DIR = REPO_ROOT / "research" / "data" / "lsr"

LSR_COL = "count_long_short_ratio"          # GLOBAL account long/short ratio
HOLD_BARS = 48                              # 48h hold to close
SLOTS = 15
RT_PRIMARY = 0.0025                         # 25 bps round-trip (taker)
RT_MAKER = 0.0010                           # 10 bps maker-in sensitivity
LO_X, HI_X = 0.05, 0.95                     # extreme percentile thresholds
COOLDOWN_BARS = 24                          # 24h per-side symbol cooldown
ROLL_W, ROLL_MIN = 720, 360                 # 30d / 15d in hours
SAMPLE_EVERY = 4                            # random-control sampling (every 4h)
LIQ_MIN = 1e6
WIN_SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
DEPLOY_BAR = 0.0025                         # +25 bps/trade verdict threshold


# ---------------------------------------------------------------- data load
def load_symbol(pair: str):
    """Return (grid, open[], close[], pctrank[], liq[], cf[]) or None.

    cf is a length-(n+1) exclusive prefix sum of hourly funding rates aligned
    to the grid: funding over grid index range [a, b) = cf[b] - cf[a].
    Reuses lsr_study.load_symbol logic; adds grid-aligned funding prefix sum.
    """
    kp, mp = KL_DIR / f"{pair}.parquet", METRICS_DIR / f"{pair}.parquet"
    if not kp.exists() or not mp.exists():
        return None
    k = pd.read_parquet(kp, columns=["open_time", "open", "close", "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    k = k[~k.index.duplicated(keep="last")]
    grid = pd.date_range(k.index.min(), k.index.max(), freq="1h", tz="UTC")
    k = k.reindex(grid)

    # liquidity mask (30d median daily quote_volume > $1M), identical to lsr_study
    dvol = k["quote_volume"].resample("1D").sum()
    liq = (dvol.rolling(30, min_periods=20).median() > LIQ_MIN)
    liq_h = liq.reindex(grid, method="ffill").fillna(False).to_numpy()

    # ratio pctrank (point-in-time trailing 30d percentile, min 15d)
    m = pd.read_parquet(mp, columns=["ts_ms", LSR_COL])
    m.index = pd.to_datetime(m["ts_ms"], unit="ms", utc=True)
    r1h = m[LSR_COL].resample("1h").last().reindex(grid).ffill(limit=12)
    pr = r1h.rolling(ROLL_W, min_periods=ROLL_MIN).rank(pct=True).to_numpy()

    # grid-aligned funding prefix sum (loader pattern from carry_study.py)
    n = len(grid)
    fr_grid = np.zeros(n, dtype=float)
    fp = FUND_DIR / f"{pair}.parquet"
    if fp.exists():
        f = pd.read_parquet(fp, columns=["fundingTime", "fundingRate"])
        ft = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("1h")
        fs = pd.Series(f["fundingRate"].to_numpy(), index=ft)
        fs = fs[~fs.index.duplicated(keep="last")].reindex(grid).fillna(0.0)
        fr_grid = fs.to_numpy()
    cf = np.concatenate([[0.0], np.cumsum(fr_grid)])   # cf[k] = sum rates[0:k]

    return grid, k["open"].to_numpy(), k["close"].to_numpy(), pr, liq_h, cf


# ---------------------------------------------------------------- event sim
def collect_trades(pairs):
    """Both-side contrarian events; returns trade DataFrame + random control."""
    trades, ctrl_rows = [], []
    for idx, pair in enumerate(pairs, 1):
        res = load_symbol(pair)
        if res is None:
            continue
        grid, opn, cls, pr, liq, cf = res
        n = len(grid)
        prev = np.r_[np.nan, pr[:-1]]
        lo_cross = (pr < LO_X) & (prev >= LO_X) & np.isfinite(prev)  # -> LONG
        hi_cross = (pr > HI_X) & (prev <= HI_X) & np.isfinite(prev)  # -> SHORT
        last_long = last_short = -10 ** 9
        for i in range(n):
            is_long, is_short = lo_cross[i], hi_cross[i]
            if not (is_long or is_short):
                continue
            if not liq[i]:
                continue
            if i + 1 + HOLD_BARS >= n:           # need full 48h window ahead
                continue
            if is_long:
                if i - last_long < COOLDOWN_BARS:
                    continue
                pos = 1
            else:
                if i - last_short < COOLDOWN_BARS:
                    continue
                pos = -1
            entry = opn[i + 1]
            exit_px = cls[i + 1 + HOLD_BARS - 1]
            if not (np.isfinite(entry) and np.isfinite(exit_px)) or entry <= 0:
                continue
            gross_dir = pos * (exit_px / entry - 1.0)
            # funding window [entry_ts, exit_ts): indices [i+1, i+HOLD_BARS)
            fsum = cf[i + 1 + HOLD_BARS - 1] - cf[i + 1]
            fpnl = -pos * fsum
            trades.append({
                "symbol": pair, "ts": grid[i + 1],
                "exit_ts": grid[i + 1 + HOLD_BARS - 1],
                "side": "long" if pos == 1 else "short", "pos": pos,
                "gross": gross_dir, "fpnl": fpnl})
            if is_long:
                last_long = i
            else:
                last_short = i

        # random-bars control: sampled liquid bars w/ full window, both dirs
        imax = n - HOLD_BARS - 2
        if imax > 0:
            ii = np.arange(0, imax + 1)
            sel = ii[(ii % SAMPLE_EVERY == 0) & liq[ii]]
            if sel.size:
                entry = opn[sel + 1]
                exit_px = cls[sel + 1 + HOLD_BARS - 1]
                gr = exit_px / entry - 1.0
                fsum = cf[sel + 1 + HOLD_BARS - 1] - cf[sel + 1]
                ok = np.isfinite(gr) & np.isfinite(entry) & (entry > 0)
                for ts_, g_, fs_ in zip(grid[sel + 1][ok], gr[ok], fsum[ok]):
                    ctrl_rows.append({"ts": ts_, "gross": g_, "fsum": fs_})
        if idx % 25 == 0:
            print(f"[{idx}/{len(pairs)}] trades={len(trades)} "
                  f"ctrl={len(ctrl_rows)}", flush=True)

    tr = pd.DataFrame(trades)
    if len(tr):
        tr["year"] = tr["ts"].dt.year
    ctrl = pd.DataFrame(ctrl_rows)
    if len(ctrl):
        ctrl["year"] = ctrl["ts"].dt.year
    return tr, ctrl


# ---------------------------------------------------------------- reporting
def _stats(net: np.ndarray) -> dict:
    if net.size == 0:
        return {"n": 0, "net_mean": None, "median": None, "win": None, "p5": None}
    return {"n": int(net.size),
            "net_mean": round(float(np.mean(net)), 5),
            "median": round(float(np.median(net)), 5),
            "win": round(float(np.mean(net > 0)), 3),
            "p5": round(float(np.percentile(net, 5)), 5)}


def side_report(tr: pd.DataFrame, cost: float) -> dict:
    """Per-trade table for a set of trades at a given cost: overall + splits."""
    if len(tr) == 0:
        return {"overall": _stats(np.array([])), "dev": _stats(np.array([])),
                "recent_2025_26": _stats(np.array([])), "by_year": {}}
    net = (tr["gross"] - cost + tr["fpnl"]).to_numpy()
    dev = tr["ts"] < WIN_SPLIT
    out = {"overall": _stats(net),
           "dev": _stats(net[dev.to_numpy()]),
           "recent_2025_26": _stats(net[(~dev).to_numpy()]),
           "by_year": {}}
    for y, g in tr.groupby("year"):
        gn = (g["gross"] - cost + g["fpnl"]).to_numpy()
        out["by_year"][str(int(y))] = _stats(gn)
    return out


def portfolio(tr: pd.DataFrame, cost: float) -> dict:
    """15-slot portfolio (liqrev_v2 pattern): 1/SLOTS equity/slot, frees @ exit."""
    if len(tr) == 0:
        return {}
    t = tr.sort_values("ts").copy()
    t["ret"] = t["gross"] - cost + t["fpnl"]
    eq, busy, curve, n_taken = 1.0, [], [], 0
    for _, r in t.iterrows():
        busy = [b for b in busy if b > r["ts"]]
        if len(busy) < SLOTS:
            eq *= (1 + r["ret"] / SLOTS)
            busy.append(r["exit_ts"])
            n_taken += 1
        curve.append((r["ts"], eq))
    c = pd.Series(dict(curve))
    if c.empty:
        return {}
    years = max((c.index[-1] - c.index[0]).days / 365.25, 1e-9)
    roll_max = c.cummax()
    maxdd = float((c / roll_max - 1.0).min())
    yr = (c.groupby(c.index.year).last() / c.groupby(c.index.year).first() - 1.0)
    return {"n_taken": int(n_taken), "n_events": int(len(t)),
            "total_return": round(float(c.iloc[-1] - 1.0), 4),
            "cagr": round(float(c.iloc[-1] ** (1 / years) - 1.0), 4),
            "max_drawdown": round(maxdd, 4),
            "by_year": {str(int(k)): round(float(v), 4) for k, v in yr.items()}}


def control_report(ctrl: pd.DataFrame) -> dict:
    """Random-bars drift baseline, both directions, at PRIMARY 25 bps + funding.

    long  net = gross - RT - fsum   (long pays funding when fsum>0)
    short net = -gross - RT + fsum  (short receives funding when fsum>0)
    """
    if len(ctrl) == 0:
        return {}
    g, fs = ctrl["gross"].to_numpy(), ctrl["fsum"].to_numpy()
    dev = (ctrl["ts"] < WIN_SPLIT).to_numpy()
    net_long = g - RT_PRIMARY - fs
    net_short = -g - RT_PRIMARY + fs
    out = {"n": int(len(ctrl))}
    for name, net in (("long", net_long), ("short", net_short)):
        by_year = {}
        for y, mask in ctrl.groupby("year").groups.items():
            m = ctrl.index.isin(mask)
            by_year[str(int(y))] = round(float(np.mean(net[m])), 5)
        out[name] = {"overall_mean": round(float(np.mean(net)), 5),
                     "dev_mean": round(float(np.mean(net[dev])), 5),
                     "recent_mean": round(float(np.mean(net[~dev])), 5),
                     "by_year": by_year}
    return out


def verdict(side_rep: dict, ctrl_dir: dict, side_name: str) -> dict:
    """Pre-registered rule at PRIMARY 25 bps cost."""
    dev_m = side_rep["dev"]["net_mean"]
    rec_m = side_rep["recent_2025_26"]["net_mean"]
    over_m = side_rep["overall"]["net_mean"]
    ctrl_m = ctrl_dir["overall_mean"] if ctrl_dir else None
    checks = {
        "dev_ge_25bps": dev_m is not None and dev_m >= DEPLOY_BAR,
        "recent_ge_25bps": rec_m is not None and rec_m >= DEPLOY_BAR,
        "same_sign_dev_recent": (dev_m is not None and rec_m is not None
                                 and np.sign(dev_m) == np.sign(rec_m)),
        "beats_random_by_25bps": (over_m is not None and ctrl_m is not None
                                  and (over_m - ctrl_m) >= DEPLOY_BAR),
    }
    deployable = all(checks.values())
    return {"side": side_name, "deployable": bool(deployable),
            "dev_net_mean": dev_m, "recent_net_mean": rec_m,
            "overall_net_mean": over_m, "random_same_dir_mean": ctrl_m,
            "edge_vs_random": (round(over_m - ctrl_m, 5)
                               if (over_m is not None and ctrl_m is not None)
                               else None),
            "checks": checks}


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = datetime.now(timezone.utc)
    pairs = [c.pair for c in load_universe()]
    print(f"universe: {len(pairs)} pairs; signal col: {LSR_COL}", flush=True)

    tr, ctrl = collect_trades(pairs)
    print(f"trades: {len(tr)}  (long={int((tr['side']=='long').sum()) if len(tr) else 0}, "
          f"short={int((tr['side']=='short').sum()) if len(tr) else 0})  "
          f"control bars: {len(ctrl)}", flush=True)

    long_tr = tr[tr["side"] == "long"] if len(tr) else tr
    short_tr = tr[tr["side"] == "short"] if len(tr) else tr

    ctrl_rep = control_report(ctrl)

    per_trade = {}
    for cost_name, cost in (("primary_25bps", RT_PRIMARY), ("maker_10bps", RT_MAKER)):
        lr = side_report(long_tr, cost)
        sr = side_report(short_tr, cost)
        cr = side_report(tr, cost)            # combined pooled
        # combined 50/50 = equal-weight of the two side means
        def _5050(a, b):
            if a is None or b is None:
                return None
            return round(0.5 * a + 0.5 * b, 5)
        combined_5050 = {
            "overall": _5050(lr["overall"]["net_mean"], sr["overall"]["net_mean"]),
            "dev": _5050(lr["dev"]["net_mean"], sr["dev"]["net_mean"]),
            "recent_2025_26": _5050(lr["recent_2025_26"]["net_mean"],
                                    sr["recent_2025_26"]["net_mean"]),
        }
        per_trade[cost_name] = {"long": lr, "short": sr,
                                "combined_pooled": cr,
                                "combined_5050_net_mean": combined_5050}

    portfolios = {
        "long": portfolio(long_tr, RT_PRIMARY),
        "short": portfolio(short_tr, RT_PRIMARY),
        "combined": portfolio(tr, RT_PRIMARY),
        "long_maker10bps": portfolio(long_tr, RT_MAKER),
        "short_maker10bps": portfolio(short_tr, RT_MAKER),
    }

    verdicts = {
        "long": verdict(per_trade["primary_25bps"]["long"],
                        ctrl_rep.get("long"), "long"),
        "short": verdict(per_trade["primary_25bps"]["short"],
                         ctrl_rep.get("short"), "short"),
    }
    any_deploy = any(v["deployable"] for v in verdicts.values())
    final = "DEPLOYABLE" if any_deploy else "NO -- footnote closed"

    result = {
        "run_utc": t0.isoformat(),
        "study": "GLOBAL account long/short ratio (count_long_short_ratio) "
                 "contrarian @ 48h -- pre-registered close-out",
        "spec": "frozen in module docstring before first run; single spec, no tuning",
        "prior_expectation": "NO -- small gross bucket tilt (-91bps/48h DEV, "
                             "sign-stable) not expected to survive costs+funding",
        "config": {"signal_col": LSR_COL, "hold_bars": HOLD_BARS,
                   "rt_primary": RT_PRIMARY, "rt_maker": RT_MAKER,
                   "extreme": [LO_X, HI_X], "cooldown_bars_per_side": COOLDOWN_BARS,
                   "roll_window_h": ROLL_W, "roll_min_h": ROLL_MIN,
                   "liq_min_usd": LIQ_MIN, "slots": SLOTS,
                   "sample_every_h_control": SAMPLE_EVERY,
                   "win_split": str(WIN_SPLIT), "deploy_bar_bps": 25,
                   "long_dir": "pctrank cross BELOW 0.05 -> +1",
                   "short_dir": "pctrank cross ABOVE 0.95 -> -1",
                   "funding": "both legs: fpnl = -pos * sum(rate over hold)"},
        "counts": {"n_trades": int(len(tr)),
                   "n_long": int((tr["side"] == "long").sum()) if len(tr) else 0,
                   "n_short": int((tr["side"] == "short").sum()) if len(tr) else 0,
                   "n_control_bars": int(len(ctrl)),
                   "n_symbols": int(tr["symbol"].nunique()) if len(tr) else 0,
                   "first_ts": str(tr["ts"].min()) if len(tr) else None,
                   "last_ts": str(tr["ts"].max()) if len(tr) else None},
        "per_trade": per_trade,
        "random_bars_control": ctrl_rep,
        "portfolios_15slot": portfolios,
        "verdict_rule": "deployable iff a side net>=+25bps in BOTH DEV and "
                        "2025-26 (same sign) AND beats same-dir random by>=25bps",
        "verdicts": verdicts,
        "final_verdict": final,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    outp = ART_DIR / "results_48h.json"
    outp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"\nFINAL VERDICT: {final}")
    print(f"artifacts -> {outp}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
