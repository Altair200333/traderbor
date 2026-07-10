"""
majors_trend.py — Defensive trend-overlay study on crypto MAJORS (BTC, ETH).

PRE-REGISTERED ONE-SHOT STUDY (spec frozen before first run; 2026-07-08 consult).
=================================================================================
This is the last unasked question from the 2026-07-08 strategy consultation.

QUESTION (defensive overlay, NOT an alpha hunt):
    Do canonical, literature-STANDARD trend rules applied to MAJORS reduce
    drawdown enough to be worth running on a BTC/ETH core holding?
    Established priors in this lab: alt-basket strategies are dead; BTC
    buy&hold was the best performer of 2022-26 but carried a -53% maxDD.
    We are looking for a *defensive overlay* that keeps most of the upside
    while cutting the worst drawdowns.

The rules below are LITERATURE-STANDARD time-series-momentum / trend filters
(Faber "A Quantitative Approach to Tactical Asset Allocation" 10-month/200-day
SMA; Moskowitz-Ooi-Pedersen "Time Series Momentum" 12-/3-month lookbacks).
They are NOT tuned or optimized on this data — no parameter search is run here.
The point of a pre-registered test is precisely that we commit to the canonical
parameters up front and report every cell.

DATA
----
research/data/binance_um/klines_1m/{BTCUSDT,ETHUSDT}.parquet
Binance USD-M futures 1m klines, 2020-01 .. 2026-07 (open_time ms UTC, close).
Load only open_time+close; resample to DAILY UTC close (last 1m close of each
UTC day). ~6.5 years spanning 2020-21 bull, 2022 bear, 2023-25 cycle, 2026 bear
— a genuine multi-regime span.

RULES  (long-or-cash; evaluated on daily close; position applied from the NEXT
day's close-to-close return => strict 1-day lag, no lookahead):
    R1: close > SMA200         (Faber 200-day trend filter)
    R2: close > SMA50          (faster SMA trend filter)
    R3: ret_90d > 0            (~3-month time-series momentum)
    R4: ret_28d > 0            (~1-month time-series momentum)
    => 8 cells total (2 assets x 4 rules). ALL are reported.

MECHANICS
    - Signal s_t computed on close of day t.
    - Position for day t+1 return = s_t  (1-day lag).
    - Strategy daily return on day t = pos_t * r_t  minus cost on days the
      position changes.
    - COSTS: 10 bps per SIDE, charged on every position change (entries and
      exits both cost 10 bps; a full round-trip = 20 bps). Trend flips are
      infrequent and maker-able, so 10 bps/side is realistic.
    - WARMUP: no signal is acted on until 200 days of data exist. All four
      rules use the SAME 200-day warmup so every cell is scored on an
      IDENTICAL span, and each cell's buy&hold benchmark uses that same span.

METRICS (per cell, vs buy&hold of the SAME asset on the SAME span):
    total return, CAGR, annualized vol, Sharpe (daily, rf=0), maxDD,
    time-in-market %, number of round-trips, worst calendar-year return;
    by-year returns; sub-period table (2020-21, 2022, 2023-24, 2025-26).

PRE-REGISTERED SUCCESS CRITERION (defensive overlay):
    A rule is USEFUL iff BOTH hold vs its own buy&hold:
        (a) maxDD reduced by >= 1/3   (|DD_strat| <= 2/3 * |DD_bh|), AND
        (b) CAGR >= (CAGR_bh - 3.0pp).
    i.e. it must cut a third of the drawdown while giving up no more than
    3 percentage points of CAGR.

HONEST NOTE (frozen with the spec): 6.5 years contains only ~3 roughly
independent bear episodes (2022, mid-2024 shakeouts, 2026). This test
validates the RECENT regime only; the deeper prior for trend/TSMOM comes
from decades of cross-asset literature evidence, not from this window.

Artifacts: research/data/majors_trend/results.json  (all cells + benchmarks).
Run:  .venv/Scripts/python.exe research/scanner_lab/majors_trend.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Frozen configuration
# --------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO, "research", "data", "binance_um", "klines_1m")
OUT_DIR = os.path.join(REPO, "research", "data", "majors_trend")

ASSETS = {"BTC": "BTCUSDT.parquet", "ETH": "ETHUSDT.parquet"}
RULES = ("R1", "R2", "R3", "R4")
RULE_DESC = {
    "R1": "close > SMA200",
    "R2": "close > SMA50",
    "R3": "ret_90d > 0",
    "R4": "ret_28d > 0",
}

COST_PER_SIDE = 0.0010          # 10 bps per side on position changes
WARMUP_DAYS = 200               # first *return* day (0-based index); needs SMA200 at t-1
ANN = 365.0                     # crypto trades every day
DD_REDUCTION_MIN = 1.0 / 3.0    # maxDD must fall by >= 1/3
CAGR_GIVEUP_MAX = 0.03          # CAGR may lag B&H by at most 3pp

# Sub-periods (calendar-year buckets), frozen in spec.
SUBPERIODS = {
    "2020-21": (2020, 2021),
    "2022": (2022, 2022),
    "2023-24": (2023, 2024),
    "2025-26": (2025, 2026),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_daily_close(fname: str) -> pd.Series:
    """1m klines -> daily UTC close (last 1m close of each UTC day)."""
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_parquet(path, columns=["open_time", "close"])
    dt = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    s = pd.Series(df["close"].to_numpy(), index=dt).sort_index()
    daily = s.resample("1D").last().dropna()
    return daily


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def build_signal(close: pd.Series, rule: str) -> np.ndarray:
    """Long-or-cash {0,1} signal on daily close. NaN during each rule's warmup."""
    c = close
    if rule == "R1":
        sig = c > c.rolling(200).mean()
        warm = c.rolling(200).mean().isna()
    elif rule == "R2":
        sig = c > c.rolling(50).mean()
        warm = c.rolling(50).mean().isna()
    elif rule == "R3":
        sig = (c / c.shift(90) - 1.0) > 0.0
        warm = c.shift(90).isna()
    elif rule == "R4":
        sig = (c / c.shift(28) - 1.0) > 0.0
        warm = c.shift(28).isna()
    else:
        raise ValueError(rule)
    out = sig.astype(float).to_numpy().copy()
    out[warm.to_numpy()] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _cagr(equity_end: float, years: float) -> float:
    if years <= 0 or equity_end <= 0:
        return float("nan")
    return equity_end ** (1.0 / years) - 1.0


def _max_dd(ret: np.ndarray) -> float:
    eq = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def metrics(ret: np.ndarray, dates: pd.DatetimeIndex, pos: np.ndarray | None) -> dict:
    """ret/dates/pos are already sliced to the scored span."""
    eq_end = float(np.prod(1.0 + ret))
    years = (dates[-1] - dates[0]).days / 365.25
    mu, sd = float(np.mean(ret)), float(np.std(ret, ddof=1))
    m = {
        "total_return": eq_end - 1.0,
        "cagr": _cagr(eq_end, years),
        "ann_vol": sd * np.sqrt(ANN),
        "sharpe": (mu / sd * np.sqrt(ANN)) if sd > 0 else float("nan"),
        "max_dd": _max_dd(ret),
        "n_obs": int(len(ret)),
        "years": years,
    }
    if pos is not None:
        m["time_in_market"] = float(np.mean(pos))
    return m


def by_year(ret: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    yrs = dates.year.to_numpy()
    out = {}
    for y in np.unique(yrs):
        r = ret[yrs == y]
        out[int(y)] = float(np.prod(1.0 + r) - 1.0)
    return out


def subperiods(ret: np.ndarray, dates: pd.DatetimeIndex, pos: np.ndarray | None) -> dict:
    yrs = dates.year.to_numpy()
    out = {}
    for name, (lo, hi) in SUBPERIODS.items():
        mask = (yrs >= lo) & (yrs <= hi)
        if not mask.any():
            out[name] = None
            continue
        r = ret[mask]
        rec = {
            "total_return": float(np.prod(1.0 + r) - 1.0),
            "max_dd": _max_dd(r),
            "n_days": int(mask.sum()),
        }
        if pos is not None:
            rec["time_in_market"] = float(np.mean(pos[mask]))
        out[name] = rec
    return out


# --------------------------------------------------------------------------- #
# Core backtest for one asset
# --------------------------------------------------------------------------- #
def run_asset(name: str, fname: str) -> dict:
    close = load_daily_close(fname)
    dates = close.index
    c = close.to_numpy()
    n = len(c)

    # daily close-to-close return
    r = np.empty(n)
    r[0] = np.nan
    r[1:] = c[1:] / c[:-1] - 1.0

    start = WARMUP_DAYS  # first scored return day (0-based); pos uses signal[start-1]
    span = slice(start, n)
    span_dates = dates[span]
    bh_ret = r[span]

    result = {
        "asset": name,
        "data": {
            "first_date": dates[0].isoformat(),
            "last_date": dates[-1].isoformat(),
            "n_daily": int(n),
            "scored_first_date": span_dates[0].isoformat(),
            "scored_n_days": int(len(bh_ret)),
        },
        "buy_and_hold": {
            **metrics(bh_ret, span_dates, None),
            "by_year": by_year(bh_ret, span_dates),
            "subperiods": subperiods(bh_ret, span_dates, None),
        },
        "cells": {},
    }

    bh = result["buy_and_hold"]

    for rule in RULES:
        sig = build_signal(close, rule)  # aligned to close index

        # position held for day t's return = signal at close of t-1
        pos = np.zeros(n)
        pos[start:] = sig[start - 1 : n - 1]
        assert not np.isnan(pos[start:]).any(), f"NaN in traded pos {name}/{rule}"

        # costs on position changes; implicit prior position = cash(0)
        pos_span = pos[span]
        prev = np.concatenate(([0.0], pos_span[:-1]))
        changes = pos_span != prev
        cost = changes.astype(float) * COST_PER_SIDE
        entries = int(np.sum((pos_span == 1.0) & (prev == 0.0)))  # round-trips

        strat_ret = pos_span * bh_ret - cost

        m = metrics(strat_ret, span_dates, pos_span)
        yr = by_year(strat_ret, span_dates)
        m["round_trips"] = entries
        m["n_position_changes"] = int(changes.sum())
        m["worst_year"] = min(yr.values())

        # pre-registered success criterion vs this asset's B&H
        dd_bh, dd_s = bh["max_dd"], m["max_dd"]
        dd_reduction = (abs(dd_bh) - abs(dd_s)) / abs(dd_bh) if dd_bh != 0 else 0.0
        cagr_gap = m["cagr"] - bh["cagr"]  # negative => lagging
        passes_dd = dd_reduction >= DD_REDUCTION_MIN
        passes_cagr = cagr_gap >= -CAGR_GIVEUP_MAX
        m["criterion"] = {
            "dd_reduction": dd_reduction,
            "cagr_gap": cagr_gap,
            "passes_dd_third": bool(passes_dd),
            "passes_cagr_within_3pp": bool(passes_cagr),
            "USEFUL": bool(passes_dd and passes_cagr),
        }
        m["by_year"] = yr
        m["subperiods"] = subperiods(strat_ret, span_dates, pos_span)
        m["rule_desc"] = RULE_DESC[rule]
        result["cells"][rule] = m

    return result


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def pct(x, nd=1):
    return "nan" if x != x else f"{x*100:.{nd}f}%"


def print_report(results: dict) -> None:
    print("\n" + "=" * 92)
    print("MAJORS TREND OVERLAY — pre-registered study (BTC, ETH)")
    print("Costs 10bps/side | 1-day lag | 200d warmup | long-or-cash")
    print("=" * 92)

    # 8-cell table
    hdr = f"{'asset':5} {'rule':4} {'desc':16} {'CAGR':>8} {'Sharpe':>7} {'maxDD':>8} {'TiM':>6} {'trips':>6} {'USEFUL':>7}"
    print("\n(1) 8-CELL TABLE vs BUY & HOLD")
    print(hdr)
    print("-" * len(hdr))
    for a in ASSETS:
        bh = results[a]["buy_and_hold"]
        print(f"{a:5} {'B&H':4} {'buy & hold':16} {pct(bh['cagr']):>8} {bh['sharpe']:>7.2f} "
              f"{pct(bh['max_dd']):>8} {'100.0%':>6} {'-':>6} {'-':>7}")
        for rule in RULES:
            m = results[a]["cells"][rule]
            print(f"{a:5} {rule:4} {RULE_DESC[rule]:16} {pct(m['cagr']):>8} {m['sharpe']:>7.2f} "
                  f"{pct(m['max_dd']):>8} {pct(m['time_in_market']):>6} {m['round_trips']:>6} "
                  f"{('YES' if m['criterion']['USEFUL'] else 'no'):>7}")

    # best DD reducer
    best = None
    for a in ASSETS:
        for rule in RULES:
            m = results[a]["cells"][rule]
            key = m["criterion"]["dd_reduction"]
            if best is None or key > best[0]:
                best = (key, a, rule, m)
    _, ba, br, bm = best
    print(f"\n(2) BY-YEAR — best DD reducer: {ba} {br} ({RULE_DESC[br]}), "
          f"DD cut {pct(bm['criterion']['dd_reduction'])} vs B&H")
    years = sorted(bm["by_year"].keys())
    bhby = results[ba]["buy_and_hold"]["by_year"]
    print(f"{'year':6} {'strat':>9} {'B&H':>9}")
    for y in years:
        print(f"{y:6} {pct(bm['by_year'][y]):>9} {pct(bhby.get(y, float('nan'))):>9}")

    # criterion pass list
    print("\n(3) PRE-REGISTERED CRITERION (DD cut >=1/3 AND CAGR >= B&H-3pp)")
    passers = []
    for a in ASSETS:
        for rule in RULES:
            m = results[a]["cells"][rule]
            cr = m["criterion"]
            tag = "PASS" if cr["USEFUL"] else "fail"
            if cr["USEFUL"]:
                passers.append(f"{a}/{rule}")
            print(f"  {a}/{rule} {RULE_DESC[rule]:16} DDcut={pct(cr['dd_reduction'])} "
                  f"CAGRgap={pct(cr['cagr_gap'])} -> {tag}")
    print(f"  PASSERS: {passers if passers else 'NONE'}")

    # subperiods
    print("\n(4) SUB-PERIODS (total return; strat TiM in parens)")
    for a in ASSETS:
        print(f"  --- {a} ---")
        bhsp = results[a]["buy_and_hold"]["subperiods"]
        line = f"  {'sub':9} {'B&H':>10}"
        for rule in RULES:
            line += f" {rule:>14}"
        print(line)
        for sp in SUBPERIODS:
            row = f"  {sp:9} {pct(bhsp[sp]['total_return']):>10}"
            for rule in RULES:
                m = results[a]["cells"][rule]["subperiods"][sp]
                row += f" {pct(m['total_return'])+'('+pct(m['time_in_market'],0)+')':>14}"
            print(row)
    print("=" * 92)


# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {a: run_asset(a, f) for a, f in ASSETS.items()}

    out = {
        "meta": {
            "study": "majors_trend defensive overlay",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "assets": list(ASSETS.keys()),
            "rules": RULE_DESC,
            "cost_per_side": COST_PER_SIDE,
            "warmup_days": WARMUP_DAYS,
            "ann_factor": ANN,
            "criterion": {
                "dd_reduction_min": DD_REDUCTION_MIN,
                "cagr_giveup_max_pp": CAGR_GIVEUP_MAX,
            },
            "note": "Literature-standard rules (Faber 200d SMA; MOP TSMOM). "
                    "Not tuned here. 6.5y ~ 3 independent bears; validates recent "
                    "regime only, deeper prior from decades of TSMOM literature.",
        },
        "results": results,
    }
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    print_report(results)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
