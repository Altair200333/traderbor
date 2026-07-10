"""Liquidation-cascade reversion event study (one-shot, pre-registered).

QUESTION: after a symbol-level forced-flow cascade (sharp price drop WITH
open-interest collapse = positions being closed, not opened), does price
mean-revert over 1h-48h enough to trade long at retail costs?

PRE-REGISTERED event definition (fixed before the first run):
  - grid: 1h bars from the 4y cache (research/data/v3/klines/1h)
  - OI: sum_open_interest (coin units, price-independent) resampled 1h last
  - trigger at bar t close: ret_6h <= -8%  AND  d_oi_6h <= -10%
  - liquidity: trailing 30d median daily quote volume > $1M (causal)
  - dedup: per symbol, 24h cooldown (first trigger wins)
  - entry: next 1h open; forward returns to close at +1h/+4h/+12h/+24h/+48h
  - practical variant: long next open -> +24h close, 25bps RT cost
  - CONTROLS (same table): (a) price-only events (ret_6h <= -8%, no OI
    condition) — isolates OI's contribution; (b) random same-symbol bars,
    20x oversampled, matched to event-year distribution.
  - splits: by year; mean/median/win-rate vs controls.
Event study only (no stops); a stop-managed variant is a follow-up question.

Usage: python liqrev_study.py
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
ART_DIR = REPO_ROOT / "research" / "data" / "liqrev"
HORIZONS = [1, 4, 12, 24, 48]
RT_COST = 0.0025
RNG = np.random.default_rng(7)


def one_symbol(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, price_only_events) rows with forward returns."""
    kp, op = KL_DIR / f"{pair}.parquet", OI_DIR / f"{pair}.parquet"
    if not kp.exists() or not op.exists():
        return pd.DataFrame(), pd.DataFrame()
    k = pd.read_parquet(kp, columns=["open_time", "open", "close", "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    oi = pd.read_parquet(op, columns=["ts_ms", "sum_open_interest"])
    oi_s = pd.Series(oi["sum_open_interest"].to_numpy(),
                     index=pd.to_datetime(oi["ts_ms"], unit="ms", utc=True))
    oi_h = oi_s.resample("1h").last().reindex(k.index).ffill()

    ret6 = k["close"].pct_change(6)
    doi6 = oi_h.pct_change(6)
    dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
    liq_ok = dvol30.reindex(k.index, method="ffill") > 1e6

    def build(mask: pd.Series) -> pd.DataFrame:
        rows = []
        last_t = None
        for t in k.index[mask.fillna(False)]:
            if last_t is not None and (t - last_t) < pd.Timedelta("24h"):
                continue
            i = k.index.get_loc(t)
            if i + 49 >= len(k):
                continue
            last_t = t
            entry = k["open"].iloc[i + 1]
            row = {"symbol": pair, "ts": t, "ret6": float(ret6.loc[t]),
                   "doi6": float(doi6.loc[t]) if np.isfinite(doi6.loc[t]) else np.nan}
            for h in HORIZONS:
                row[f"fwd_{h}h"] = float(k["close"].iloc[i + h] / entry - 1.0)
            rows.append(row)
        return pd.DataFrame(rows)

    base = (ret6 <= -0.08) & liq_ok
    return build(base & (doi6 <= -0.10)), build(base)


def summarize(df: pd.DataFrame, name: str) -> dict:
    if df.empty:
        return {"name": name, "n": 0}
    out = {"name": name, "n": int(len(df)),
           "n_symbols": int(df["symbol"].nunique())}
    for h in HORIZONS:
        c = df[f"fwd_{h}h"]
        out[f"fwd_{h}h"] = {"mean": round(float(c.mean()), 4),
                            "median": round(float(c.median()), 4),
                            "win": round(float((c > 0).mean()), 3)}
    net24 = df["fwd_24h"] - RT_COST
    out["practical_24h_net"] = {"mean": round(float(net24.mean()), 4),
                                "median": round(float(net24.median()), 4),
                                "win": round(float((net24 > 0).mean()), 3),
                                "sum": round(float(net24.sum()), 3)}
    out["by_year_mean_24h"] = {str(y): round(float(g["fwd_24h"].mean()), 4)
                               for y, g in df.groupby(df["ts"].dt.year)}
    return out


def random_control(pairs: list[str], events: pd.DataFrame) -> pd.DataFrame:
    """Same-symbol random bars, 20x, matched to the event year distribution."""
    if events.empty:
        return pd.DataFrame()
    rows = []
    per_sym = events.groupby("symbol").size()
    for pair, n_ev in per_sym.items():
        kp = KL_DIR / f"{pair}.parquet"
        k = pd.read_parquet(kp, columns=["open_time", "open", "close"])
        k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
        k = k.set_index("ts").sort_index()
        years = events[events["symbol"] == pair]["ts"].dt.year.tolist()
        for y in years:
            idx = k.index[(k.index.year == y)]
            idx = idx[(np.arange(len(idx)) + 49) < len(k)]
            if len(idx) < 50:
                continue
            for t in RNG.choice(idx[:-50], size=min(20, len(idx) - 50), replace=False):
                i = k.index.get_loc(t)
                entry = k["open"].iloc[i + 1]
                row = {"symbol": pair, "ts": t}
                for h in HORIZONS:
                    row[f"fwd_{h}h"] = float(k["close"].iloc[i + h] / entry - 1.0)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ev_parts, po_parts = [], []
    pairs = [c.pair for c in load_universe()]
    for i, pair in enumerate(pairs, 1):
        e, p = one_symbol(pair)
        if len(e):
            ev_parts.append(e)
        if len(p):
            po_parts.append(p)
        if i % 30 == 0:
            print(f"[{i}/{len(pairs)}] scanned, events so far: "
                  f"{sum(len(x) for x in ev_parts)}", flush=True)
    events = pd.concat(ev_parts, ignore_index=True) if ev_parts else pd.DataFrame()
    price_only = pd.concat(po_parts, ignore_index=True) if po_parts else pd.DataFrame()
    print(f"\ncascade events: {len(events)}  price-only events: {len(price_only)}")

    ctrl = random_control(pairs, events)
    reports = [summarize(events, "CASCADE (price+OI)"),
               summarize(price_only, "price_only_control"),
               summarize(ctrl, "random_control")]
    for r in reports:
        print(json.dumps(r, indent=2))

    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "results.json").write_text(json.dumps(
        {"run_utc": datetime.now(timezone.utc).isoformat(), "rt_cost": RT_COST,
         "reports": reports}, indent=2), encoding="utf-8")
    if len(events):
        events.to_parquet(ART_DIR / "events.parquet", index=False)
    print(f"\nartifacts -> {ART_DIR}")


if __name__ == "__main__":
    main()
