"""Weekly long-only time-series-momentum study on the 4y 1h klines cache.

PRE-REGISTERED primary spec (parameters fixed from the 2026-07-07 research
notes BEFORE the first run; sensitivities are reported in full, never
cherry-picked):
  - signal: per-coin 28d total return > 0 (time-series, no cross-sectional rank)
  - liquidity: trailing 30d median daily quote-volume, top 75 at rebalance;
    if >25 coins qualify, keep the 25 most liquid (liquidity cut, not momentum)
  - weights: 1/sigma(20d daily returns), normalized, clipped at 10%, renorm;
    shortfall stays in cash
  - rebalance: Monday 00:00 UTC close; costs = per-side bps x |dweight| vs
    drifted prior weights (primary 12.5bps/side = 25bps RT; 5 and 25 reported)
  - benchmarks: BTC buy-and-hold; weekly-rebalanced equal-weight of the
    liquidity top-75
  - KNOWN BIAS, stated up front: the 149-coin universe was selected in 2026 ->
    pre-2024 results carry survivorship inflation. The 2024-07..2026-07
    subperiod (near-contemporaneous universe) is reported separately and is
    the honest headline.

Usage: python weekly_tsmom.py [--klines research/data/v3/klines/1h]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402

ART_DIR = REPO_ROOT / "research" / "data" / "tsmom"


def load_daily(kl_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, dvols = {}, {}
    for c in load_universe():
        p = kl_dir / f"{c.pair}.parquet"
        if not p.exists():
            continue
        k = pd.read_parquet(p, columns=["open_time", "close", "quote_volume"])
        ts = pd.to_datetime(k["open_time"], unit="ms", utc=True)
        s = k.set_index(ts).sort_index()
        closes[c.pair] = s["close"].resample("1D").last()
        dvols[c.pair] = s["quote_volume"].resample("1D").sum(min_count=1)
    px = pd.DataFrame(closes)
    dv = pd.DataFrame(dvols)
    return px, dv


def build_weights(px: pd.DataFrame, dv: pd.DataFrame, t: pd.Timestamp,
                  formation_d: int, use_ma_filter: bool,
                  liq_top: int = 75, max_names: int = 25,
                  w_cap: float = 0.10) -> pd.Series:
    """Causal weights decided at close of day t."""
    hist = px.loc[:t]
    if len(hist) < formation_d + 2:
        return pd.Series(dtype=float)
    liq = dv.loc[:t].tail(30).median()
    live = liq.dropna().nlargest(liq_top).index
    form = hist[live].iloc[-1] / hist[live].iloc[-(formation_d + 1)] - 1.0
    qual = form[form > 0].index
    if use_ma_filter:
        ma50 = hist[live].tail(50).mean()
        qual = [s for s in qual if hist[s].iloc[-1] > ma50[s]]
    if len(qual) == 0:
        return pd.Series(dtype=float)
    if len(qual) > max_names:  # liquidity cut, never momentum rank
        qual = liq[qual].nlargest(max_names).index
    rets = hist[qual].pct_change().tail(20)
    sig = rets.std()
    sig = sig[(sig > 0) & sig.notna()]
    if sig.empty:
        return pd.Series(dtype=float)
    w = (1.0 / sig)
    w = w / w.sum()
    w = w.clip(upper=w_cap)
    w = w / max(w.sum(), 1.0)  # cap shortfall -> cash, never leverage
    return w


def run_backtest(px: pd.DataFrame, dv: pd.DataFrame, formation_d: int,
                 side_cost_bps: float, use_ma_filter: bool,
                 start: str, end: str) -> dict:
    days = px.loc[start:end].index
    mondays = [d for d in days if d.dayofweek == 0]
    rets = px.pct_change()
    w = pd.Series(dtype=float)
    equity, dates = [1.0], [days[0]]
    turnover_log, n_names_log = [], []
    for i, d in enumerate(days[1:], 1):
        r = float((rets.loc[d].reindex(w.index).fillna(0.0) * w).sum()) if len(w) else 0.0
        cost = 0.0
        if d in mondays:
            # drift old weights through the day's return before diffing
            if len(w):
                gross = w * (1.0 + rets.loc[d].reindex(w.index).fillna(0.0))
                w_drift = gross / max(1.0 + r, 1e-9)
            else:
                w_drift = w
            w_new = build_weights(px, dv, d, formation_d, use_ma_filter)
            all_idx = w_drift.index.union(w_new.index)
            turn = float((w_new.reindex(all_idx, fill_value=0.0)
                          - w_drift.reindex(all_idx, fill_value=0.0)).abs().sum())
            cost = turn * side_cost_bps / 1e4
            turnover_log.append(turn)
            n_names_log.append(int(len(w_new)))
            w = w_new
        equity.append(equity[-1] * (1.0 + r - cost))
        dates.append(d)
    eq = pd.Series(equity, index=pd.DatetimeIndex(dates))
    dr = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1.0).min()
    out = {
        "formation_d": formation_d, "side_cost_bps": side_cost_bps,
        "ma_filter": use_ma_filter, "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "total_return": round(float(eq.iloc[-1] - 1.0), 4),
        "cagr": round(float(eq.iloc[-1] ** (1 / years) - 1.0), 4),
        "ann_vol": round(float(dr.std() * np.sqrt(365)), 4),
        "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(365)), 3)
        if dr.std() > 0 else None,
        "max_dd": round(float(dd), 4),
        "avg_weekly_turnover": round(float(np.mean(turnover_log)), 4)
        if turnover_log else None,
        "avg_n_names": round(float(np.mean(n_names_log)), 1) if n_names_log else None,
        "pct_weeks_invested": round(float(np.mean([n > 0 for n in n_names_log])), 3)
        if n_names_log else None,
    }
    return {"summary": out, "equity": eq}


def bench_btc(px: pd.DataFrame, start: str, end: str) -> dict:
    eq = px["BTCUSDT"].loc[start:end].dropna()
    eq = eq / eq.iloc[0]
    dr = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {"name": "BTC_BH", "total_return": round(float(eq.iloc[-1] - 1.0), 4),
            "cagr": round(float(eq.iloc[-1] ** (1 / years) - 1.0), 4),
            "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(365)), 3),
            "max_dd": round(float((eq / eq.cummax() - 1.0).min()), 4)}


def bench_ew(px: pd.DataFrame, dv: pd.DataFrame, start: str, end: str,
             liq_top: int = 75) -> dict:
    """Weekly-rebalanced equal-weight of the liquidity top-75 — the 'asset
    class itself, signal off' benchmark (declared in the spec)."""
    days = px.loc[start:end].index
    rets = px.pct_change()
    w = pd.Series(dtype=float)
    eq = [1.0]
    for d in days[1:]:
        r = float((rets.loc[d].reindex(w.index).fillna(0.0) * w).sum()) if len(w) else 0.0
        if d.dayofweek == 0:
            live = dv.loc[:d].tail(30).median().dropna().nlargest(liq_top).index
            w = pd.Series(1.0 / len(live), index=live)
        eq.append(eq[-1] * (1.0 + r))
    s = pd.Series(eq, index=days)
    dr = s.pct_change().dropna()
    years = (days[-1] - days[0]).days / 365.25
    return {"name": "EW_liquid75_weekly", "total_return": round(float(s.iloc[-1] - 1.0), 4),
            "cagr": round(float(s.iloc[-1] ** (1 / years) - 1.0), 4),
            "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(365)), 3),
            "max_dd": round(float((s / s.cummax() - 1.0).min()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines", default=str(REPO_ROOT / "research" / "data" / "v3"
                                            / "klines" / "1h"))
    args = ap.parse_args()

    px, dv = load_daily(Path(args.klines))
    print(f"panel: {px.shape[1]} symbols x {px.shape[0]} days "
          f"({px.index[0].date()} .. {px.index[-1].date()})")

    windows = [("full_4y", "2022-06-01", "2026-07-06"),
               ("honest_2y", "2024-07-01", "2026-07-06")]
    runs = []
    # PRIMARY first, then pre-declared sensitivities — all reported
    grid = [dict(formation_d=28, side_cost_bps=12.5, use_ma_filter=False, tag="PRIMARY"),
            dict(formation_d=28, side_cost_bps=5.0, use_ma_filter=False, tag="cost5"),
            dict(formation_d=28, side_cost_bps=25.0, use_ma_filter=False, tag="cost25"),
            dict(formation_d=14, side_cost_bps=12.5, use_ma_filter=False, tag="form14"),
            dict(formation_d=56, side_cost_bps=12.5, use_ma_filter=False, tag="form56"),
            dict(formation_d=28, side_cost_bps=12.5, use_ma_filter=True, tag="ma50")]
    for wname, ws, we in windows:
        b = bench_btc(px, ws, we)
        bew = bench_ew(px, dv, ws, we)
        print(f"\n=== window {wname} ({ws}..{we})  BTC B&H: {b} ===")
        print(f"    asset-class control {bew}")
        runs.append({"window": wname, "tag": "bench_ew", **bew})
        for g in grid:
            tag = g.pop("tag")
            r = run_backtest(px, dv, start=ws, end=we, **g)
            g["tag"] = tag
            row = {"window": wname, "tag": tag, **r["summary"]}
            runs.append(row)
            print(f"  {tag:>8s}: ret={row['total_return']:+8.2%} cagr={row['cagr']:+7.2%} "
                  f"shp={row['sharpe']} dd={row['max_dd']:.1%} "
                  f"turn/wk={row['avg_weekly_turnover']} names={row['avg_n_names']} "
                  f"inv={row['pct_weeks_invested']}")
            if tag == "PRIMARY":
                ART_DIR.mkdir(parents=True, exist_ok=True)
                r["equity"].to_csv(ART_DIR / f"equity_{wname}_primary.csv")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = {"run_utc": datetime.now(timezone.utc).isoformat(),
           "spec": "pre-registered primary + declared sensitivities, see docstring",
           "results": runs}
    (ART_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nartifacts -> {ART_DIR}")


if __name__ == "__main__":
    main()
