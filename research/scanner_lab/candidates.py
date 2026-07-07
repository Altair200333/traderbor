"""Vectorized candidate generation replicating production screener semantics.

Production reference: agents-v2/traderbot_ai/screener/ (patterns.py, gates.py,
plan.py, config.py @ screener-1.3.1). We generate a RELAXED pool (wider than
production hard+marginal) so the ML layer can learn beyond current gates, and
record production-parity flags (is_hard / is_marginal) per event as baselines.

Pool admission (relaxed):
  pattern hit (P1/P1H/P2/P3, priority P2>P1>P1H>P3)
  AND directional roc_4h >= 1.5% (production S1: 2.5%)
  AND vol_ratio >= 1.5        (production S3: 2.0)
  AND atr_pct in (0.003, 0.06) (production S5: 0.005..0.040)
  AND >= 168 bars history, AND 4h dedup per symbol+side.

Every production gate value is recorded so is_hard/is_marginal can be derived
and gate margins feed the model as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators_vec import (
    atr_wilder, ema, median_range_pct, range_expansion, roc, rolling_median_prev,
    rsi_wilder, zscore,
)

# production config values (config.py, screener-1.3.1)
P1_LOOKBACK = 20
P1_HOLD_BARS = 6
P2_TREND_WINDOW = 24
P2_TREND_SHARE = 0.80
P2_PULLBACK_BARS = 3
P3_ATR_COMPRESSION = 0.7
P3_ATR_LOOKBACK = 72
P3_RANGE_BARS = 48
STOP_ATR_MULT = 2.0
STRUCT_BUFFER_ATR = 0.5
STOP_PCT_MIN = 0.010
STOP_ATR_FLOOR_MULT = 1.5
STOP_NOISE_MULT = 1.5
STOP_PCT_MAX = 0.040
TP_RR = {"P1": 2.5, "P1H": 2.0, "P2": 2.0, "P3": 2.5}

# pool (relaxed) thresholds
POOL_ROC4H_MIN = 0.015
POOL_VOL_RATIO_MIN = 1.5
POOL_ATR_MIN, POOL_ATR_MAX = 0.003, 0.06
DEDUP_BARS = 4          # per symbol+side, matches production cooldown_candidate=4h
MIN_HISTORY = 168


def compute_frame(df: pd.DataFrame) -> pd.DataFrame:
    """df: 1h klines (open_time ms, open/high/low/close/volume/...). Returns df + indicator cols."""
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]
    out["roc_1h"] = roc(c, 1)
    out["roc_4h"] = roc(c, 4)
    out["roc_24h"] = roc(c, 24)
    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["rsi14"] = rsi_wilder(c, 14)
    out["atr"] = atr_wilder(h, l, c, 14)
    out["atr_pct"] = out["atr"] / c
    out["vol_med24"] = rolling_median_prev(v, 24)
    out["vol_ratio"] = v / out["vol_med24"].replace(0.0, np.nan)
    out["z20"] = zscore(c, 20)
    out["rexp72"] = range_expansion(h, l, 72)
    out["med_range24"] = median_range_pct(h, l, c, 24)
    # pattern building blocks
    out["hh20"] = h.shift(1).rolling(P1_LOOKBACK).max()      # prior 20-bar high
    out["ll20"] = l.shift(1).rolling(P1_LOOKBACK).min()
    out["hh48"] = h.shift(1).rolling(P3_RANGE_BARS).max()
    out["ll48"] = l.shift(1).rolling(P3_RANGE_BARS).min()
    out["atr_72ago"] = out["atr"].shift(P3_ATR_LOOKBACK)
    out["above_ema50_share"] = (c > out["ema50"]).rolling(P2_TREND_WINDOW).mean()
    out["below_ema50_share"] = (c < out["ema50"]).rolling(P2_TREND_WINDOW).mean()
    # ema20 touch: bar range crosses ema20, within last P2_PULLBACK_BARS bars
    touch = (l <= out["ema20"]) & (out["ema20"] <= h)
    out["ema20_touch_recent"] = touch.rolling(P2_PULLBACK_BARS).max().astype(bool)
    out["prev_high"] = h.shift(1)
    out["prev_low"] = l.shift(1)
    return out


def _patterns_side(f: pd.DataFrame, side: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (pattern_id Series of str|'' , boundary Series, p1h_age Series)."""
    c, h, l = f["close"], f["high"], f["low"]
    n = len(f)
    if side == "long":
        p1 = c > f["hh20"]
        p1_boundary = f["hh20"]
        p2 = (
            (f["above_ema50_share"] >= P2_TREND_SHARE)
            & f["ema20_touch_recent"]
            & (c > f["prev_high"])
        )
        p3 = (f["atr"] <= P3_ATR_COMPRESSION * f["atr_72ago"]) & (c > f["hh48"])
        p3_boundary = f["hh48"]
    else:
        p1 = c < f["ll20"]
        p1_boundary = f["ll20"]
        p2 = (
            (f["below_ema50_share"] >= P2_TREND_SHARE)
            & f["ema20_touch_recent"]
            & (c < f["prev_low"])
        )
        p3 = (f["atr"] <= P3_ATR_COMPRESSION * f["atr_72ago"]) & (c < f["ll48"])
        p3_boundary = f["ll48"]

    # P1H: a P1 breakout in last P1_HOLD_BARS bars, close still beyond ORIGINAL boundary
    p1_np = p1.to_numpy(copy=False)
    bound_np = p1_boundary.to_numpy(copy=False)
    close_np = c.to_numpy(copy=False)
    p1h = np.zeros(n, dtype=bool)
    p1h_age = np.zeros(n, dtype=np.int32)
    p1h_bound = np.full(n, np.nan)
    for age in range(1, P1_HOLD_BARS + 1):
        past_p1 = np.roll(p1_np, age)
        past_p1[:age] = False
        past_bound = np.roll(bound_np, age)
        past_bound[:age] = np.nan
        if side == "long":
            holds = past_p1 & (close_np > past_bound)
        else:
            holds = past_p1 & (close_np < past_bound)
        newly = holds & ~p1h
        p1h |= newly
        p1h_age[newly] = age
        p1h_bound[newly] = past_bound[newly]

    # priority P2 > P1 > P1H > P3
    pat = np.full(n, "", dtype=object)
    boundary = np.full(n, np.nan)
    m = p3.to_numpy(copy=False) & ~np.isnan(p3_boundary.to_numpy(copy=False))
    pat[m] = "P3"
    boundary[m] = p3_boundary.to_numpy(copy=False)[m]
    m = p1h & (pat != "P2") & (pat != "P1")
    pat[m] = "P1H"
    boundary[m] = p1h_bound[m]
    m = p1.to_numpy(copy=False)
    pat[m] = "P1"
    boundary[m] = bound_np[m]
    m = p2.to_numpy(copy=False)
    pat[m] = "P2"
    # P2 invalidation uses pullback extreme, boundary not defined; keep NaN there
    boundary[m] = np.nan
    return pd.Series(pat, index=f.index), pd.Series(boundary, index=f.index), pd.Series(p1h_age, index=f.index)


def scan_symbol(df1h: pd.DataFrame, symbol: str, btc_roc4h: pd.Series | None) -> pd.DataFrame:
    """Generate relaxed-pool events for one symbol. btc_roc4h indexed by open_time (ms)."""
    f = compute_frame(df1h)
    f["bar_idx"] = np.arange(len(f))
    events = []
    for side in ("long", "short"):
        pat, boundary, p1h_age = _patterns_side(f, side)
        sign = 1.0 if side == "long" else -1.0
        droc4 = sign * f["roc_4h"]
        pool = (
            (pat != "")
            & (droc4 >= POOL_ROC4H_MIN)
            & (f["vol_ratio"] >= POOL_VOL_RATIO_MIN)
            & f["atr_pct"].between(POOL_ATR_MIN, POOL_ATR_MAX)
            & (f["bar_idx"] >= MIN_HISTORY)
        )
        idx = np.flatnonzero(pool.to_numpy(copy=False))
        if len(idx) == 0:
            continue
        # 4h dedup per side
        kept = []
        last = -10**9
        for i in idx:
            if i - last >= DEDUP_BARS:
                kept.append(i)
                last = i
        sub = f.iloc[kept]
        pat_s = pat.iloc[kept]
        bound_s = boundary.iloc[kept]
        age_s = p1h_age.iloc[kept]

        c = sub["close"]
        atr = sub["atr"]
        # plan: invalidation & stop distance
        if side == "long":
            p2_extreme = f["low"].rolling(P2_PULLBACK_BARS + 1).min().iloc[kept]
            inval_pat = bound_s - STRUCT_BUFFER_ATR * atr
            inval_p2 = p2_extreme - STRUCT_BUFFER_ATR * atr
        else:
            p2_extreme = f["high"].rolling(P2_PULLBACK_BARS + 1).max().iloc[kept]
            inval_pat = bound_s + STRUCT_BUFFER_ATR * atr
            inval_p2 = p2_extreme + STRUCT_BUFFER_ATR * atr
        inval = inval_pat.where(pat_s != "P2", inval_p2)
        d_struct = (c - inval).abs() / c
        d_atr = STOP_ATR_MULT * atr / c
        d_noise = np.maximum.reduce([
            np.full(len(sub), STOP_PCT_MIN),
            (STOP_ATR_FLOOR_MULT * sub["atr_pct"]).to_numpy(),
            (STOP_NOISE_MULT * sub["med_range24"]).to_numpy(),
        ])
        d_final = np.maximum.reduce([d_atr.to_numpy(), d_struct.to_numpy(), d_noise])

        breakout_dist_atr = sign * (c - bound_s) / atr  # S9c value (P1/P1H/P3 only)
        ev = pd.DataFrame({
            "symbol": symbol,
            "side": side,
            "open_time": sub["open_time"].to_numpy(),
            "as_of": sub["open_time"].to_numpy() + 3_600_000,
            "bar_idx": sub["bar_idx"].to_numpy(),
            "pattern": pat_s.to_numpy(),
            "p1h_age": age_s.to_numpy(),
            "entry": c.to_numpy(),
            "boundary": bound_s.to_numpy(),
            "d_final": d_final,
            "d_atr": d_atr.to_numpy(),
            "d_struct": d_struct.to_numpy(),
            "d_noise": d_noise,
            "tp_rr": pat_s.map(TP_RR).to_numpy(),
            "roc_1h": sub["roc_1h"].to_numpy(),
            "roc_4h": sub["roc_4h"].to_numpy(),
            "roc_24h": sub["roc_24h"].to_numpy(),
            "vol_ratio": sub["vol_ratio"].to_numpy(),
            "rsi14": sub["rsi14"].to_numpy(),
            "atr_pct": sub["atr_pct"].to_numpy(),
            "ema20_ext_atr": (sign * (c - sub["ema20"]) / atr).to_numpy(),
            "breakout_dist_atr": breakout_dist_atr.to_numpy(),
            "z20": sub["z20"].to_numpy(),
            "rexp72": sub["rexp72"].to_numpy(),
            "med_range24": sub["med_range24"].to_numpy(),
            "above_ema50": (sub["close"] > sub["ema50"]).to_numpy(),
        })
        events.append(ev)
    if not events:
        return pd.DataFrame()
    out = pd.concat(events, ignore_index=True)
    if btc_roc4h is not None:
        out["btc_roc_4h"] = out["open_time"].map(btc_roc4h)
    else:
        out["btc_roc_4h"] = np.nan
    return _add_production_flags(out)


def _add_production_flags(ev: pd.DataFrame) -> pd.DataFrame:
    """Production gate booleans + is_hard / is_marginal parity flags."""
    sign = np.where(ev["side"] == "long", 1.0, -1.0)
    droc4 = sign * ev["roc_4h"]
    droc24 = sign * ev["roc_24h"]
    s1 = droc4 >= 0.025
    s2 = droc24 >= 0.020
    s3 = ev["vol_ratio"] >= 2.0
    s4 = np.where(ev["side"] == "long",
                  (ev["rsi14"] > 55) & (ev["rsi14"] < 78),
                  (ev["rsi14"] > 22) & (ev["rsi14"] < 45))
    s5 = (ev["atr_pct"] > 0.005) & (ev["atr_pct"] < 0.040)
    s6 = np.where(ev["side"] == "long", ev["above_ema50"], ~ev["above_ema50"])
    s7 = np.where(ev["side"] == "long", ev["btc_roc_4h"] >= -0.010, ev["btc_roc_4h"] <= 0.010)
    s7 = np.where(ev["btc_roc_4h"].isna(), True, s7)  # missing BTC -> pass (matrix parity)
    last_share = (ev["roc_1h"].abs() / ev["roc_4h"].abs().replace(0.0, np.nan)).fillna(np.inf)
    s9a = last_share <= 0.45
    s9b = ev["ema20_ext_atr"].abs() <= 1.5
    s9c_applic = ev["pattern"].isin(["P1", "P1H", "P3"])
    s9c = np.where(s9c_applic, ev["breakout_dist_atr"] <= 0.8, True)
    s10 = np.where(ev["rexp72"].isna(), True, ev["rexp72"] <= 0.35)
    s11 = np.where(ev["side"] == "long", ev["z20"] <= 3.0, ev["z20"] >= -3.0)
    stop_feasible = (ev["d_final"] >= STOP_PCT_MIN) & (ev["d_final"] <= STOP_PCT_MAX)

    gates = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7,
             "s9a": s9a, "s9b": s9b, "s9c": s9c, "s10": s10, "s11": s11}
    for k, v in gates.items():
        ev[f"g_{k}"] = np.asarray(v, dtype=bool)
    ev["stop_feasible"] = stop_feasible.to_numpy()

    all_pass = np.logical_and.reduce([ev[f"g_{k}"].to_numpy() for k in gates])
    ev["is_hard"] = all_pass & ev["stop_feasible"].to_numpy()
    fail_only_ext = np.logical_and.reduce(
        [ev[f"g_{k}"].to_numpy() for k in gates if k not in ("s9b", "s9c")]
    ) & ~all_pass
    ev["is_marginal"] = (
        fail_only_ext
        & ev["g_s4"].to_numpy() & ev["g_s9a"].to_numpy()
        & (ev["ema20_ext_atr"].abs().to_numpy() <= 2.5)
        & (np.where(s9c_applic, ev["breakout_dist_atr"], 0.0) <= 1.6)
        & (ev["side"] == "long").to_numpy()
        & ev["pattern"].isin(["P1", "P1H", "P3"]).to_numpy()
        & ev["stop_feasible"].to_numpy()
    )
    return ev
