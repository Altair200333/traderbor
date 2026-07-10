"""Liquidation-cascade reversion v2: stop-managed, execution-aware (one-shot).

PRE-REGISTERED spec (frozen from the 2026-07-08 design pass BEFORE this run;
liqrev_study.py = v1 event study, this adds stops/execution/portfolio):
  - events: identical to v1 (ret_6h <= -8% AND OI_6h <= -10%, $1M liquidity,
    24h symbol cooldown, 1h grid, 4y span)
  - PRIMARY: entry market at next 1h open; STOP = trigger-bar low (gap-aware:
    exit at min(open, stop) when breached; entry bar itself can stop); else
    exit +24h close; costs 25bps RT; portfolio 15 slots x 1/15 equity,
    slot frees at actual exit
  - declared variants (all reported, none hidden):
      no_stop        - v1 comparability control
      stop_buffer1   - stop at trigger low x 0.99 (liquidation lows are
                       stop-hunt magnets; buffer may cut noise stopouts)
      maker          - post-only limit at trigger close, valid through next
                       1h bar, filled iff next bar low < limit (trade-through);
                       unfilled = skipped; 10bps RT; fill rate reported
      deep           - subset ret_6h <= -12% (severity branch)
      cost 50/100bps - storm-slippage haircuts on PRIMARY
  - reporting: per-event net mean/median/win/stop-rate; portfolio total/CAGR/
    maxDD/worst-month; by-year + 2025-26 subperiod for PRIMARY
Live shadow (small size) remains the final validator regardless of outcome.

Usage: python liqrev_v2.py
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
HOLD_BARS = 24
SLOTS = 15


def detect_events(pair: str) -> pd.DataFrame:
    """v1-identical trigger detection; returns rows with kline context."""
    kp, op = KL_DIR / f"{pair}.parquet", OI_DIR / f"{pair}.parquet"
    if not kp.exists() or not op.exists():
        return pd.DataFrame()
    k = pd.read_parquet(kp, columns=["open_time", "open", "high", "low", "close",
                                     "quote_volume"])
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
    mask = ((ret6 <= -0.08) & (doi6 <= -0.10) & liq_ok).fillna(False)

    rows, last_t = [], None
    for t in k.index[mask]:
        if last_t is not None and (t - last_t) < pd.Timedelta("24h"):
            continue
        i = k.index.get_loc(t)
        if i + HOLD_BARS + 1 >= len(k):
            continue
        last_t = t
        rows.append({"symbol": pair, "ts": t, "i": i, "ret6": float(ret6.loc[t]),
                     "trig_low": float(k["low"].iloc[i]),
                     "trig_close": float(k["close"].iloc[i])})
    if not rows:
        return pd.DataFrame()
    ev = pd.DataFrame(rows)
    ev["_k"] = pair  # marker; klines fetched again in simulate for memory economy
    return ev


def simulate(ev: pd.DataFrame, kcache: dict, entry_mode: str, stop_mode: str,
             rt_cost: float, subset_deep: bool) -> pd.DataFrame:
    out = []
    for _, r in ev.iterrows():
        if subset_deep and r["ret6"] > -0.12:
            continue
        k = kcache[r["symbol"]]
        i = int(r["i"])
        bar1_open = float(k["open"].iloc[i + 1])
        bar1_low = float(k["low"].iloc[i + 1])
        if entry_mode == "market":
            entry, entry_bar = bar1_open, i + 1
        else:  # maker: limit at trigger close, next bar only, trade-through fill
            limit = r["trig_close"]
            if bar1_low < limit:
                entry, entry_bar = limit, i + 1
            else:
                out.append({"symbol": r["symbol"], "ts": r["ts"], "filled": False})
                continue
        stop = (None if stop_mode == "none"
                else r["trig_low"] * (0.99 if stop_mode == "buffer1" else 1.0))
        exit_px, exit_bar, stopped = None, entry_bar + HOLD_BARS, False
        for j in range(entry_bar, entry_bar + HOLD_BARS):
            o, lo = float(k["open"].iloc[j]), float(k["low"].iloc[j])
            if stop is not None and (o <= stop or lo <= stop):
                exit_px, exit_bar, stopped = min(o, stop), j, True
                break
        if exit_px is None:
            exit_px = float(k["close"].iloc[entry_bar + HOLD_BARS - 1])
            exit_bar = entry_bar + HOLD_BARS - 1
        ret = exit_px / entry - 1.0 - rt_cost
        out.append({"symbol": r["symbol"], "ts": r["ts"], "filled": True,
                    "ret": ret, "stopped": stopped,
                    "exit_ts": k.index[exit_bar]})
    return pd.DataFrame(out)


def portfolio(tr: pd.DataFrame) -> dict:
    t = tr[tr["filled"]].sort_values("ts")
    eq, busy, curve = 1.0, [], []
    n_taken = 0
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
    years = (c.index[-1] - c.index[0]).days / 365.25
    m = c.resample("MS").last().ffill().pct_change().dropna()
    yearly = c.groupby(c.index.year).last() / c.groupby(c.index.year).first() - 1
    return {"total": round(float(eq - 1), 4),
            "cagr": round(float(eq ** (1 / years) - 1), 4),
            "maxDD": round(float((c / c.cummax() - 1).min()), 4),
            "worst_month": round(float(m.min()), 4) if len(m) else None,
            "n_taken": n_taken,
            "by_year": {str(k): round(float(v), 3) for k, v in yearly.items()}}


def report(tr: pd.DataFrame, tag: str) -> dict:
    f = tr[tr["filled"]]
    rec = f[f["ts"] >= "2025-01-01"]
    out = {"tag": tag, "n_events": int(len(tr)), "n_filled": int(len(f)),
           "fill_rate": round(float(len(f) / len(tr)), 3) if len(tr) else None,
           "net_mean": round(float(f["ret"].mean()), 4) if len(f) else None,
           "net_median": round(float(f["ret"].median()), 4) if len(f) else None,
           "win": round(float((f["ret"] > 0).mean()), 3) if len(f) else None,
           "stop_rate": round(float(f["stopped"].mean()), 3) if len(f) else None,
           "p5": round(float(f["ret"].quantile(0.05)), 4) if len(f) else None,
           "recent25_26": {"n": int(len(rec)),
                           "net_mean": round(float(rec["ret"].mean()), 4)
                           if len(rec) else None,
                           "win": round(float((rec["ret"] > 0).mean()), 3)
                           if len(rec) else None},
           "portfolio_15slots": portfolio(tr)}
    return out


def main() -> None:
    evs, kcache = [], {}
    pairs = [c.pair for c in load_universe()]
    for i, pair in enumerate(pairs, 1):
        e = detect_events(pair)
        if len(e):
            evs.append(e)
            kcache[pair] = pd.read_parquet(
                KL_DIR / f"{pair}.parquet",
                columns=["open_time", "open", "high", "low", "close"])
            kcache[pair]["ts"] = pd.to_datetime(kcache[pair]["open_time"],
                                                unit="ms", utc=True)
            kcache[pair] = kcache[pair].set_index("ts").sort_index()
        if i % 40 == 0:
            print(f"[{i}/{len(pairs)}] events {sum(len(x) for x in evs)}", flush=True)
    ev = pd.concat(evs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    print(f"events total: {len(ev)}")

    grid = [
        ("PRIMARY_stop_25bps", "market", "low", 0.0025, False),
        ("cost50", "market", "low", 0.0050, False),
        ("cost100", "market", "low", 0.0100, False),
        ("no_stop_control", "market", "none", 0.0025, False),
        ("stop_buffer1pct", "market", "buffer1", 0.0025, False),
        ("maker_10bps", "maker", "low", 0.0010, False),
        ("deep_-12pct", "market", "low", 0.0025, True),
    ]
    reports = []
    for tag, em, sm, cost, deep in grid:
        sm_ = {"low": "low", "none": "none", "buffer1": "buffer1"}[sm]
        tr = simulate(ev, kcache, em, sm_, cost, deep)
        rep = report(tr, tag)
        reports.append(rep)
        print(json.dumps(rep, indent=1, default=str))

    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "results_v2.json").write_text(json.dumps(
        {"run_utc": datetime.now(timezone.utc).isoformat(),
         "slots": SLOTS, "hold_bars": HOLD_BARS, "reports": reports},
        indent=2, default=str), encoding="utf-8")
    print(f"\nartifacts -> {ART_DIR / 'results_v2.json'}")


if __name__ == "__main__":
    main()
