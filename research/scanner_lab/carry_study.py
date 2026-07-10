"""Funding-carry measurement study (one-shot, pre-registered).

QUESTION: how much delta-neutral funding rent (short perp + long spot,
collect positive funding) was harvestable on our 149-coin universe,
2022-07..2026-07, net of realistic entry/exit costs?

PRE-REGISTERED spec (fixed before the first run):
  - climate baseline: cross-coin average annualized funding by month (no
    trading, no costs) — majors (BTC/ETH/BNB/SOL/XRP) vs rest.
  - PRIMARY portfolio: weekly (Monday 00:00 UTC) review; rank by trailing 7d
    mean annualized funding (21 settlements); hold coins with trailing APR >
    10%; top-10 by that same trailing APR, equal weight. Position collects
    the realized funding of the NEXT week (8h settlements, sign as received
    by the perp-short: positive funding = we receive, negative = we pay).
  - costs: membership change = full round trip of 2 legs x 2 sides; taker
    20bps/leg RT total 40bps primary; maker 8bps RT reported.
  - sensitivities (all reported): threshold 5%/20% APR; top-5/top-20;
  - metrics: net APR by year, avg names held, weekly membership turnover,
    worst month, majors-only and ex-majors variants.
NOT measured (stated): basis convergence noise, ADL events, margin drag —
this measures the funding stream itself; a live carry book has extra tail
risk on top (Oct-2025-cascade class).

Usage: python carry_study.py
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

FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
ART_DIR = REPO_ROOT / "research" / "data" / "carry"
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
SETTLES_PER_YEAR = 3 * 365


def load_funding() -> pd.DataFrame:
    """Panel: rows = settlement grid (8h), cols = symbols, values = fundingRate."""
    series = {}
    for c in load_universe():
        p = FUND_DIR / f"{c.pair}.parquet"
        if not p.exists():
            continue
        f = pd.read_parquet(p)
        ts = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("1h")
        s = pd.Series(f["fundingRate"].to_numpy(), index=ts)
        series[c.pair] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(series).sort_index()


def climate(fr: pd.DataFrame) -> pd.DataFrame:
    apr = fr * SETTLES_PER_YEAR
    by_month = apr.resample("MS").mean()
    out = pd.DataFrame({
        "all_mean_apr": by_month.mean(axis=1),
        "majors_mean_apr": by_month[[c for c in MAJORS if c in apr.columns]].mean(axis=1),
        "exmajors_mean_apr": by_month[[c for c in apr.columns
                                       if c not in MAJORS]].mean(axis=1),
        "frac_coins_pos": (by_month > 0).mean(axis=1),
    })
    return out.round(4)


def run_portfolio(fr: pd.DataFrame, thresh_apr: float, top_n: int,
                  rt_cost: float, universe: list[str] | None = None) -> dict:
    f = fr[universe] if universe else fr
    trail = f.rolling(21, min_periods=15).mean() * SETTLES_PER_YEAR
    mondays = [d for d in pd.date_range(f.index[0], f.index[-1], freq="W-MON",
                                        tz="UTC") if d in f.index]
    held: set[str] = set()
    pnl = pd.Series(0.0, index=f.index)
    costs = pd.Series(0.0, index=f.index)
    n_names, turnover = [], []
    reb = {d: None for d in mondays}
    for d in mondays:
        t = trail.loc[d].dropna()
        elig = t[t > thresh_apr].nlargest(top_n).index
        new = set(elig)
        changed = len(new ^ held)
        turnover.append(changed)
        costs.loc[d] += changed * rt_cost / 2  # half RT per membership change side
        held = new
        n_names.append(len(held))
        reb[d] = sorted(held)
    # walk settlements: between rebalances, collect realized funding of held set
    cur: set[str] = set()
    ri = iter(sorted(reb))
    nxt = next(ri, None)
    for ts in f.index:
        while nxt is not None and ts >= nxt:
            cur = set(reb[nxt])
            nxt = next(ri, None)
        if cur:
            pnl.loc[ts] = float(f.loc[ts, list(cur)].fillna(0.0).mean())
    daily = (pnl - costs).resample("1D").sum()
    eq = (1 + daily).cumprod()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    by_year = ((1 + daily).groupby(daily.index.year).prod() - 1).round(4)
    monthly = (1 + daily).resample("MS").prod() - 1
    return {"thresh_apr": thresh_apr, "top_n": top_n, "rt_cost_bps": rt_cost * 1e4,
            "universe": "majors" if universe == MAJORS else
            ("exmajors" if universe else "all"),
            "net_total": round(float(eq.iloc[-1] - 1), 4),
            "net_apr": round(float(eq.iloc[-1] ** (1 / years) - 1), 4),
            "worst_month": round(float(monthly.min()), 4),
            "avg_names": round(float(np.mean(n_names)), 1),
            "avg_weekly_membership_changes": round(float(np.mean(turnover)), 2),
            "by_year": {str(k): float(v) for k, v in by_year.items()}}


def main() -> None:
    fr = load_funding()
    print(f"funding panel: {fr.shape[1]} symbols x {fr.shape[0]} settlements "
          f"({fr.index[0]} .. {fr.index[-1]})")
    cl = climate(fr)
    print("\n=== CLIMATE: mean annualized funding by year (no trading) ===")
    print(cl.groupby(cl.index.year).mean().round(4).to_string())

    runs = []
    grid = [
        dict(thresh_apr=0.10, top_n=10, rt_cost=0.0040, tag="PRIMARY_taker"),
        dict(thresh_apr=0.10, top_n=10, rt_cost=0.0008, tag="maker"),
        dict(thresh_apr=0.05, top_n=10, rt_cost=0.0040, tag="thresh5"),
        dict(thresh_apr=0.20, top_n=10, rt_cost=0.0040, tag="thresh20"),
        dict(thresh_apr=0.10, top_n=5, rt_cost=0.0040, tag="top5"),
        dict(thresh_apr=0.10, top_n=20, rt_cost=0.0040, tag="top20"),
        dict(thresh_apr=0.10, top_n=10, rt_cost=0.0040, universe=MAJORS, tag="majors_only"),
        dict(thresh_apr=0.10, top_n=10, rt_cost=0.0040,
             universe=[c for c in fr.columns if c not in MAJORS], tag="exmajors"),
    ]
    print("\n=== PORTFOLIOS ===")
    for g in grid:
        tag = g.pop("tag")
        r = run_portfolio(fr, **g)
        r["tag"] = tag
        runs.append(r)
        print(f"  {tag:>14s}: net_apr={r['net_apr']:+7.2%} total={r['net_total']:+8.2%} "
              f"worst_mo={r['worst_month']:+6.2%} names={r['avg_names']} "
              f"chg/wk={r['avg_weekly_membership_changes']} by_year={r['by_year']}")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "results.json").write_text(json.dumps(
        {"run_utc": datetime.now(timezone.utc).isoformat(),
         "climate_by_year": cl.groupby(cl.index.year).mean().round(4).to_dict(),
         "portfolios": runs}, indent=2), encoding="utf-8")
    cl.to_csv(ART_DIR / "climate_monthly.csv")
    print(f"\nartifacts -> {ART_DIR}")


if __name__ == "__main__":
    main()
