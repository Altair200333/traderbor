"""Weekly TSMOM short-side / long-short study (one-shot, pre-registered).

Follow-up to weekly_tsmom.py (long-only NO-GO): does the SHORT side of weekly
trend pay on perps, with realized funding included?

PRE-REGISTERED spec (fixed before first run):
  - signal: per-coin 28d return < 0 -> short candidate (mirror of long spec);
    liquidity top-75, max 25 names by liquidity, inverse-vol weights cap 10%
  - variants: SHORT_ONLY; LONG_SHORT (50% capital each leg, legs built
    independently); both on perps -> realized 8h funding applied from the
    funding panel: shorts RECEIVE positive funding / PAY negative; longs the
    opposite. (carry study: avg funding 2025-26 is NEGATIVE = headwind for
    shorts now.)
  - costs: per-side 12.5bps x turnover (25bps RT primary); cost5 sensitivity
  - windows: full_4y 2022-06.. / honest_2y 2024-07.. ; benchmark: short the
    EW liquid-75 basket (with funding) = "short the asset class" control
  - KNOWN BIAS, works AGAINST shorts here: 2026-survivor universe excludes
    delisted/dead coins -> short returns are UNDERestimated. Long-only had the
    opposite bias. No intraweek stops -> squeeze weeks hit at full weight;
    worst week reported.

Usage: python weekly_tsmom_short.py
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
from weekly_tsmom import load_daily, build_weights  # noqa: E402
from carry_study import load_funding  # noqa: E402

ART_DIR = REPO_ROOT / "research" / "data" / "tsmom"


def short_weights(px: pd.DataFrame, dv: pd.DataFrame, t: pd.Timestamp,
                  formation_d: int = 28) -> pd.Series:
    """Mirror of build_weights: 28d return < 0, inverse-vol, cap 10%."""
    hist = px.loc[:t]
    if len(hist) < formation_d + 2:
        return pd.Series(dtype=float)
    liq = dv.loc[:t].tail(30).median()
    live = liq.dropna().nlargest(75).index
    form = hist[live].iloc[-1] / hist[live].iloc[-(formation_d + 1)] - 1.0
    qual = form[form < 0].index
    if len(qual) == 0:
        return pd.Series(dtype=float)
    if len(qual) > 25:
        qual = liq[qual].nlargest(25).index
    sig = hist[qual].pct_change().tail(20).std()
    sig = sig[(sig > 0) & sig.notna()]
    if sig.empty:
        return pd.Series(dtype=float)
    w = 1.0 / sig
    w = (w / w.sum()).clip(upper=0.10)
    return w / max(w.sum(), 1.0)


def run(px: pd.DataFrame, dv: pd.DataFrame, fund_daily: pd.DataFrame,
        mode: str, side_cost_bps: float, start: str, end: str,
        ew_short: bool = False) -> dict:
    days = px.loc[start:end].index
    rets = px.pct_change()
    wl = pd.Series(dtype=float)   # long weights (positive)
    ws = pd.Series(dtype=float)   # short weights (positive numbers, short exposure)
    eq, weekly_r = [1.0], []
    turnover_log = []
    wk_accum = 0.0
    for d in days[1:]:
        rl = float((rets.loc[d].reindex(wl.index).fillna(0.0) * wl).sum()) if len(wl) else 0.0
        rs = float((rets.loc[d].reindex(ws.index).fillna(0.0) * ws).sum()) if len(ws) else 0.0
        fl = float((fund_daily.loc[d].reindex(wl.index).fillna(0.0) * wl).sum()) \
            if (len(wl) and d in fund_daily.index) else 0.0
        fs = float((fund_daily.loc[d].reindex(ws.index).fillna(0.0) * ws).sum()) \
            if (len(ws) and d in fund_daily.index) else 0.0
        # longs pay positive funding, shorts receive it
        r = (rl - fl) - rs + fs
        cost = 0.0
        if d.dayofweek == 0:
            if ew_short:
                live = dv.loc[:d].tail(30).median().dropna().nlargest(75).index
                ns = pd.Series(1.0 / len(live), index=live)
                nl = pd.Series(dtype=float)
            elif mode == "short_only":
                ns = short_weights(px, dv, d)
                nl = pd.Series(dtype=float)
            else:  # long_short: half capital each leg
                nl = build_weights(px, dv, d, 28, False) * 0.5
                ns = short_weights(px, dv, d) * 0.5
            turn = 0.0
            for old, new in ((wl, nl), (ws, ns)):
                idx = old.index.union(new.index)
                turn += float((new.reindex(idx, fill_value=0.0)
                               - old.reindex(idx, fill_value=0.0)).abs().sum())
            cost = turn * side_cost_bps / 1e4
            turnover_log.append(turn)
            wl, ws = nl, ns
        step = r - cost
        eq.append(eq[-1] * (1.0 + step))
        wk_accum += step
        if d.dayofweek == 6:
            weekly_r.append(wk_accum)
            wk_accum = 0.0
    s = pd.Series(eq, index=days)
    dr = s.pct_change().dropna()
    years = (days[-1] - days[0]).days / 365.25
    return {"mode": "EW_SHORT_control" if ew_short else mode,
            "side_cost_bps": side_cost_bps,
            "total": round(float(s.iloc[-1] - 1), 4),
            "cagr": round(float(max(s.iloc[-1], 1e-6) ** (1 / years) - 1), 4),
            "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(365)), 3)
            if dr.std() > 0 else None,
            "max_dd": round(float((s / s.cummax() - 1).min()), 4),
            "worst_week": round(float(min(weekly_r)), 4) if weekly_r else None,
            "avg_turnover_wk": round(float(np.mean(turnover_log)), 3)
            if turnover_log else None}


def main() -> None:
    px, dv = load_daily(REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h")
    fr = load_funding()
    fund_daily = fr.resample("1D").sum().reindex(px.index).fillna(0.0)
    print(f"panel {px.shape}, funding panel {fr.shape}")

    windows = [("full_4y", "2022-06-01", "2026-07-06"),
               ("honest_2y", "2024-07-01", "2026-07-06")]
    grid = [dict(mode="short_only", side_cost_bps=12.5, tag="SHORT_PRIMARY"),
            dict(mode="short_only", side_cost_bps=5.0, tag="short_cost5"),
            dict(mode="long_short", side_cost_bps=12.5, tag="LONG_SHORT"),
            dict(mode="short_only", side_cost_bps=12.5, ew_short=True, tag="EW_SHORT_ctl")]
    runs = []
    for wname, ws_, we_ in windows:
        print(f"\n=== window {wname} ({ws_}..{we_}) ===")
        for g in grid:
            tag = g.pop("tag")
            r = run(px, dv, fund_daily, start=ws_, end=we_, **g)
            g["tag"] = tag
            row = {"window": wname, "tag": tag, **r}
            runs.append(row)
            print(f"  {tag:>14s}: total={r['total']:+8.2%} cagr={r['cagr']:+7.2%} "
                  f"shp={r['sharpe']} dd={r['max_dd']:.1%} worst_wk={r['worst_week']} "
                  f"turn={r['avg_turnover_wk']}")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "results_short.json").write_text(json.dumps(
        {"run_utc": datetime.now(timezone.utc).isoformat(), "results": runs},
        indent=2), encoding="utf-8")
    print(f"\nartifacts -> {ART_DIR / 'results_short.json'}")


if __name__ == "__main__":
    main()
