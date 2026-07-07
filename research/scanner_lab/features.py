"""Feature enrichment: per-symbol extras + market/cross-sectional features.

All features use only data up to and including the (closed) trigger bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators_vec import atr_wilder, ema, roc, zscore


def extra_symbol_features(df1h: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Adds per-symbol features indexed via events.bar_idx."""
    c, h, v = df1h["close"], df1h["high"], df1h["volume"]
    qv, tr = df1h["quote_volume"], df1h["trades"]
    atr = atr_wilder(h, df1h["low"], c, 14)
    e20, e50, e100, e200 = ema(c, 20), ema(c, 50), ema(c, 100), ema(c, 200)
    # multi-horizon trend alignment (CTREND motif): count of bullish MA relations
    align = (
        (c > e20).astype(int) + (e20 > e50).astype(int)
        + (e50 > e100).astype(int) + (e100 > e200).astype(int)
    )
    f = pd.DataFrame({
        "ma_align": align,
        "ema50_slope_24h": e50 / e50.shift(24) - 1.0,
        "roc_12h": roc(c, 12),
        "roc_72h": roc(c, 72),
        "roc_168h": roc(c, 168),
        "atr_ratio_72": atr / atr.shift(72),
        "qvol_z168": (qv - qv.rolling(168).mean()) / qv.rolling(168).std(ddof=0),
        "trades_z168": (tr - tr.rolling(168).mean()) / tr.rolling(168).std(ddof=0),
        "taker_share": (df1h["taker_buy_base"] / v.replace(0.0, np.nan)),
        "dist_hh168_atr": (h.rolling(168).max() - c) / atr,
        "bbw20": c.rolling(20).std(ddof=0) / c,
    })
    f["taker_share_4h"] = f["taker_share"].rolling(4).mean()
    # squeeze percentile: bb width vs trailing 30d
    f["bbw_pctile_720"] = f["bbw20"].rolling(720, min_periods=240).rank(pct=True)
    # hour-of-day-normalized relative volume (lit: the "in play" filter):
    # volume vs median of same-UTC-hour volume over prior 20 days
    hod = pd.to_datetime(df1h["open_time"], unit="ms", utc=True).dt.hour
    f["rvol_hod"] = v / v.groupby(hod).transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).median()
    ).replace(0.0, np.nan)
    took = f.iloc[events["bar_idx"].to_numpy()].reset_index(drop=True)
    ev = events.reset_index(drop=True).join(took)
    ts = pd.to_datetime(ev["as_of"], unit="ms", utc=True)
    ev["hour"] = ts.dt.hour
    ev["dow"] = ts.dt.dayofweek
    return ev


def build_market_frames(frames_1h: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frames_1h: pair -> 1h klines. Returns (market_df indexed by open_time,
    xsec_ranks stacked Series-frame indexed by (open_time, pair))."""
    closes = {}
    vols = {}
    for pair, df in frames_1h.items():
        s = df.set_index("open_time")
        closes[pair] = s["close"]
        vols[pair] = s["volume"]
    close_m = pd.DataFrame(closes).sort_index()
    roc4_m = close_m / close_m.shift(4) - 1.0
    roc24_m = close_m / close_m.shift(24) - 1.0
    ema50_m = close_m.ewm(span=50, adjust=False).mean()
    vol_m = pd.DataFrame(vols).sort_index()
    volratio_m = vol_m / vol_m.shift(1).rolling(24).median().replace(0.0, np.nan)

    btc = close_m["BTCUSDT"]
    btc_ret = btc.pct_change()
    # denominators = listed coins only (NaN pre-listing must not count as False)
    market = pd.DataFrame({
        "breadth_ema50": (close_m > ema50_m).sum(axis=1) / close_m.notna().sum(axis=1),
        "breadth_roc24_pos": (roc24_m > 0).sum(axis=1) / roc24_m.notna().sum(axis=1),
        "median_roc24": roc24_m.median(axis=1),
        "median_roc4": roc4_m.median(axis=1),
        "btc_roc_24h": roc24_m["BTCUSDT"],
        "btc_z20": zscore(btc, 20),
        "btc_vol_24h": btc_ret.rolling(24).std(ddof=0),
        "btc_above_ema50": (btc > ema50_m["BTCUSDT"]).astype(float),
    })
    market["altseason"] = market["median_roc24"] - market["btc_roc_24h"]

    ranks = pd.DataFrame({
        "rank_roc4": roc4_m.rank(axis=1, pct=True).stack(),
        "rank_roc24": roc24_m.rank(axis=1, pct=True).stack(),
        "rank_volratio": volratio_m.rank(axis=1, pct=True).stack(),
    })
    return market, ranks


def join_market(ev: pd.DataFrame, market: pd.DataFrame, ranks: pd.DataFrame,
                universe_meta: pd.DataFrame) -> pd.DataFrame:
    ev = ev.merge(market, left_on="open_time", right_index=True, how="left")
    ev = ev.merge(ranks, left_on=["open_time", "symbol"], right_index=True, how="left")
    ev = ev.merge(universe_meta, on="symbol", how="left")
    return ev
