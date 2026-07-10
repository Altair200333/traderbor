"""PRE-REGISTERED EVENT STUDY — token unlock (vesting cliff) price pressure.

Written & committed BEFORE looking at any result. Hypothesis (Keyrock 2024): large
vesting-cliff unlocks precede/accompany negative price pressure, with front-running
from ~ -30 days.

DATA
    Events : research/data/unlocks/unlock_events.parquet  (from fetch_unlocks.py)
             one row per discrete CLIFF unlock; frac_supply = cliff_tokens/maxSupply.
    Prices : research/data/v3/klines/1h/{PAIR}.parquet, UTC, resampled to daily close.

EVENT DEFINITION (pre-registered filters, ALL must hold)
    1. frac_supply >= 0.01  (tranche >= 1% of max supply). Large subset: >= 0.03.
    2. token maps to our 149-pair universe (symbol -> PAIR).
    3. event_date lies inside our klines span with room for the measured window.
    4. token had >= 60 daily bars of price history BEFORE the event.

ABNORMAL RETURN MODEL (pre-registered)
    r_i(t)   = simple daily return of token i (close-to-close, UTC).
    m(t)     = equal-weight mean daily return across all universe pairs trading that day
               (removes market drift; market-adjusted / beta=1 model).
    AR_i(t)  = r_i(t) - m(t).
    CAR over a window = arithmetic sum of AR over the window's daily bars.

MEASUREMENT WINDOWS (trading-day offsets around day 0 = unlock date)
    PRE30 [-30,-1]   PRE7 [-7,-1]   POST7 [+1,+7]   POST30 [+1,+30]

CONTROLS
    For each qualifying event, draw 20 random valid dates from the SAME token's history
    (positions with full -30/+30 room), compute the same window CARs. Pool per subset;
    report control mean and +/-1 se band  (se = std / sqrt(n_control)).  Seed = 12345.

SPLITS
    DEV = event year < 2025 ; LIVE = event year in {2025, 2026}.
    Size buckets: MID = [1%,3%) ; LARGE = >=3%.

REPORTED PER SUBSET/WINDOW
    n events, mean CAR, median CAR, pct_negative (win rate for the short hypothesis),
    control mean, control se.

PRE-REGISTERED INTERPRETATION RULE
    A tradable pattern exists for a window IFF, on the >=1% sample:
        |mean CAR| >= 0.03  AND  same sign in DEV and LIVE
        AND event mean CAR lies OUTSIDE the pooled random-control +/-1 se band.
    This is an EVENT STUDY, not a strategy: no portfolio simulation, no costs,
    no execution model. A pass is necessary-not-sufficient evidence to justify a
    strategy backtest later.

Outputs: research/data/unlocks/results.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from universe import load_universe  # noqa: E402

DATA = HERE.parents[0] / "data"
KLINES = DATA / "v3" / "klines" / "1h"
OUT = DATA / "unlocks"
SEED = 12345
WINDOWS = {"PRE30": (-30, -1), "PRE7": (-7, -1), "POST7": (1, 7), "POST30": (1, 30)}
N_CTRL = 20


def daily_close(pair: str) -> pd.Series | None:
    fp = KLINES / f"{pair}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp, columns=["open_time", "close"])
    idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    s = pd.Series(df["close"].values, index=idx).sort_index()
    return s.resample("1D").last().dropna()


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = load_universe()
    closes = {}
    for c in universe:
        s = daily_close(c.pair)
        if s is not None and len(s) > 0:
            closes[c.pair] = s
    px = pd.DataFrame(closes).sort_index()
    ret = px.pct_change()
    market = ret.mean(axis=1)               # equal-weight universe mean per day
    ar = ret.sub(market, axis=0)            # abnormal return panel
    return ret, ar


def car_at(ar_pair: pd.Series, pos: int, lo: int, hi: int) -> float:
    """Sum AR over positional offsets [pos+lo, pos+hi] inclusive; NaN if out of range."""
    a, b = pos + lo, pos + hi
    if a < 0 or b >= len(ar_pair):
        return np.nan
    seg = ar_pair.iloc[a : b + 1]
    if seg.isna().any():
        return np.nan
    return float(seg.sum())


def all_windows(ar_pair: pd.Series, pos: int) -> dict:
    return {w: car_at(ar_pair, pos, lo, hi) for w, (lo, hi) in WINDOWS.items()}


def summarize(rows: list[dict], key: str) -> dict:
    out = {}
    for w in WINDOWS:
        vals = np.array([r[key][w] for r in rows if not np.isnan(r[key][w])], float)
        if len(vals) == 0:
            out[w] = {"n": 0}
            continue
        out[w] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 5),
            "median": round(float(np.median(vals)), 5),
            "pct_negative": round(float((vals < 0).mean()), 3),
        }
    return out


def control_stats(ctrl_rows: list[dict]) -> dict:
    out = {}
    for w in WINDOWS:
        vals = np.array([v for r in ctrl_rows for v in r[w] if not np.isnan(v)], float)
        if len(vals) == 0:
            out[w] = {"n": 0}
            continue
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        out[w] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 5),
            "se": round(se, 5),
            "band": [round(float(vals.mean() - se), 5), round(float(vals.mean() + se), 5)],
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    ret, ar = build_panel()
    span = (str(ar.index.min().date()), str(ar.index.max().date()))
    print("panel:", ar.shape, "span", span)

    ev = pd.read_parquet(OUT / "unlock_events.parquet")
    ev = ev[ev["frac_supply"] >= 0.01].copy()
    ev["event_dt"] = pd.to_datetime(ev["event_date"], utc=True)
    print("events >=1% (pre-filter):", len(ev))

    events, ctrl_rows = [], []
    dropped = {"no_pair": 0, "lt60_hist": 0, "no_pos": 0}
    for _, r in ev.iterrows():
        pair = r["pair"]
        if pair not in ar.columns:
            dropped["no_pair"] += 1
            continue
        ap = ar[pair].dropna()
        if len(ap) < 61:
            dropped["lt60_hist"] += 1
            continue
        pos = ap.index.searchsorted(r["event_dt"])
        if pos <= 0 or pos >= len(ap):
            dropped["no_pos"] += 1
            continue
        # require >=60 prior daily bars before the event position
        if pos < 60:
            dropped["lt60_hist"] += 1
            continue
        cars = all_windows(ap, pos)
        year = int(r["event_date"][:4])
        events.append({
            "symbol": r["symbol"], "pair": pair, "event_date": r["event_date"],
            "frac_supply": float(r["frac_supply"]),
            "split": "DEV" if year < 2025 else "LIVE",
            "bucket": "LARGE" if r["frac_supply"] >= 0.03 else "MID",
            "car": cars,
        })
        # matched random controls from valid positions [60, len-31]
        lo_p, hi_p = 60, len(ap) - 31
        if hi_p > lo_p:
            picks = rng.integers(lo_p, hi_p + 1, size=N_CTRL)
            crow = {w: [] for w in WINDOWS}
            for p in picks:
                for w, (a, b) in WINDOWS.items():
                    crow[w].append(car_at(ap, int(p), a, b))
            ctrl_rows.append(crow)

    print("qualifying events:", len(events), "dropped:", dropped)

    def subset(pred):
        return [e for e in events if pred(e)]

    subsets = {
        "ALL_ge1pct": subset(lambda e: True),
        "DEV": subset(lambda e: e["split"] == "DEV"),
        "LIVE": subset(lambda e: e["split"] == "LIVE"),
        "MID_1to3pct": subset(lambda e: e["bucket"] == "MID"),
        "LARGE_ge3pct": subset(lambda e: e["bucket"] == "LARGE"),
        "DEV_LARGE": subset(lambda e: e["split"] == "DEV" and e["bucket"] == "LARGE"),
        "LIVE_LARGE": subset(lambda e: e["split"] == "LIVE" and e["bucket"] == "LARGE"),
    }
    tables = {k: {"n_events": len(v), "car": summarize(v, "car")} for k, v in subsets.items()}
    ctrl = control_stats(ctrl_rows)

    # size distribution
    fr = np.array([e["frac_supply"] for e in events], float)
    size_dist = {
        "n": len(fr),
        "quantiles": {q: round(float(np.quantile(fr, q)), 4)
                      for q in (0.5, 0.75, 0.9, 0.95, 0.99)} if len(fr) else {},
        "max": round(float(fr.max()), 4) if len(fr) else None,
    }

    # interpretation rule (on ALL_ge1pct), per window
    verdict = {}
    dev, live = tables["DEV"]["car"], tables["LIVE"]["car"]
    allc = tables["ALL_ge1pct"]["car"]
    for w in WINDOWS:
        a = allc.get(w, {})
        if a.get("n", 0) == 0 or dev.get(w, {}).get("n", 0) == 0 or live.get(w, {}).get("n", 0) == 0:
            verdict[w] = {"pass": False, "reason": "insufficient n in a split"}
            continue
        mean = a["mean"]
        same_sign = np.sign(dev[w]["mean"]) == np.sign(live[w]["mean"]) and dev[w]["mean"] != 0
        band = ctrl.get(w, {}).get("band")
        outside = band is not None and (mean < band[0] or mean > band[1])
        verdict[w] = {
            "pass": bool(abs(mean) >= 0.03 and same_sign and outside),
            "all_mean": mean, "dev_mean": dev[w]["mean"], "live_mean": live[w]["mean"],
            "same_sign": bool(same_sign), "ctrl_band": band, "outside_ctrl_band": bool(outside),
            "abs_ge_3pct": bool(abs(mean) >= 0.03),
        }

    results = {
        "panel_span": span,
        "n_events_qualifying": len(events),
        "dropped": dropped,
        "size_distribution_frac_supply": size_dist,
        "control": ctrl,
        "tables": tables,
        "interpretation_rule": verdict,
        "notes": "Event study only; no portfolio sim, no costs. Unlock dates are known "
                 "in advance (anticipation/front-running expected). Overlapping windows "
                 "for tokens with frequent cliffs are not independent.",
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "n": len(events)}, indent=1))


if __name__ == "__main__":
    main()
