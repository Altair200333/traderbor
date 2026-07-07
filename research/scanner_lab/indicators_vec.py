"""Vectorized indicators matching production screener semantics
(agents-v2/traderbot_ai/screener/indicators.py) on pandas Series.

Wilder RSI/ATR via ewm(alpha=1/n, adjust=False): converges to production
values after ~5n bars; all events require >=168 bars of history, so the
seed difference is negligible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def roc(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1.0


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ru / rd.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(rd != 0, 100.0)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def rolling_median_prev(s: pd.Series, n: int) -> pd.Series:
    """Median of the previous n values (excluding current)."""
    return s.shift(1).rolling(n).median()


def zscore(close: pd.Series, n: int = 20) -> pd.Series:
    m = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return (close - m) / sd.replace(0.0, np.nan)


def range_expansion(high: pd.Series, low: pd.Series, n: int = 72) -> pd.Series:
    """(maxH - minL) / minL over trailing n bars (incl current)."""
    mx = high.rolling(n).max()
    mn = low.rolling(n).min()
    return (mx - mn) / mn


def median_range_pct(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 24) -> pd.Series:
    return ((high - low) / close).rolling(n).median()
