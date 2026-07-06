from __future__ import annotations

import math
from statistics import median


Number = float | int


def roc(values: list[Number], periods: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if periods <= 0:
        raise ValueError("periods must be positive")
    for index in range(periods, len(values)):
        previous = float(values[index - periods])
        if previous == 0 or not math.isfinite(previous):
            result[index] = None
        else:
            current = float(values[index])
            result[index] = current / previous - 1.0 if math.isfinite(current) else None
    return result


def ema(values: list[Number], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return result
    seed_values = [float(value) for value in values[:period]]
    if not all(math.isfinite(value) for value in seed_values):
        return result
    value = sum(seed_values) / period
    result[period - 1] = value
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        price = float(values[index])
        previous = result[index - 1]
        if not math.isfinite(price) or previous is None:
            result[index] = None
        else:
            value = price * alpha + previous * (1.0 - alpha)
            result[index] = value
    return result


def rsi_wilder(values: list[Number], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) <= period:
        return result
    closes = [float(value) for value in values]
    deltas = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    first = deltas[:period]
    if not all(math.isfinite(delta) for delta in first):
        return result
    avg_gain = sum(max(delta, 0.0) for delta in first) / period
    avg_loss = sum(max(-delta, 0.0) for delta in first) / period
    result[period] = _rsi_from_avgs(avg_gain, avg_loss)
    for delta_index in range(period, len(deltas)):
        delta = deltas[delta_index]
        if not math.isfinite(delta):
            result[delta_index + 1] = None
            continue
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        result[delta_index + 1] = _rsi_from_avgs(avg_gain, avg_loss)
    return result


def atr_wilder(high: list[Number], low: list[Number], close: list[Number], period: int) -> list[float | None]:
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have equal length")
    result: list[float | None] = [None] * len(close)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(close) <= period:
        return result
    highs = [float(value) for value in high]
    lows = [float(value) for value in low]
    closes = [float(value) for value in close]
    true_ranges: list[float] = []
    for index in range(1, len(closes)):
        tr = max(highs[index] - lows[index], abs(highs[index] - closes[index - 1]), abs(lows[index] - closes[index - 1]))
        true_ranges.append(tr)
    first = true_ranges[:period]
    if not all(math.isfinite(value) for value in first):
        return result
    atr = sum(first) / period
    result[period] = atr
    for tr_index in range(period, len(true_ranges)):
        value = true_ranges[tr_index]
        if not math.isfinite(value):
            result[tr_index + 1] = None
            continue
        atr = (atr * (period - 1) + value) / period
        result[tr_index + 1] = atr
    return result


def sma(values: list[Number], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be positive")
    for index in range(period - 1, len(values)):
        sample = [float(value) for value in values[index - period + 1 : index + 1]]
        if all(math.isfinite(value) for value in sample):
            result[index] = sum(sample) / period
    return result


def rolling_std(values: list[Number], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be positive")
    for index in range(period - 1, len(values)):
        sample = [float(value) for value in values[index - period + 1 : index + 1]]
        if not all(math.isfinite(value) for value in sample):
            continue
        mean = sum(sample) / period
        result[index] = math.sqrt(sum((value - mean) ** 2 for value in sample) / period)
    return result


def zscore_last(values: list[Number], period: int) -> float | None:
    if len(values) < period:
        return None
    mean_series = sma(values, period)
    std_series = rolling_std(values, period)
    mean = mean_series[-1]
    std = std_series[-1]
    if mean is None or std is None or std == 0.0:
        return None
    return (float(values[-1]) - mean) / std


def range_expansion_last(high: list[Number], low: list[Number], window: int) -> float | None:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(high) < window or len(low) < window:
        return None
    highs = [float(value) for value in high[-window:]]
    lows = [float(value) for value in low[-window:]]
    if not all(math.isfinite(value) for value in highs + lows):
        return None
    lowest = min(lows)
    if lowest <= 0:
        return None
    return (max(highs) - lowest) / lowest


def median_range_pct(high: list[Number], low: list[Number], close: list[Number], window: int) -> float | None:
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have equal length")
    start = max(0, len(close) - window)
    ranges = [
        (float(high[index]) - float(low[index])) / abs(float(close[index]))
        for index in range(start, len(close))
        if float(close[index]) != 0
    ]
    if not ranges:
        return None
    return float(median(ranges))


def chande_kroll_stop(
    high: list[Number],
    low: list[Number],
    close: list[Number],
    p: int = 10,
    x: float = 1.0,
    q: int = 9,
) -> tuple[list[float | None], list[float | None]]:
    """Chande Kroll stop: double-smoothed ATR stop bands (long stop below, short stop above)."""
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have equal length")
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")
    length = len(close)
    atr_series = atr_wilder(high, low, close, p)
    first_high: list[float | None] = [None] * length
    first_low: list[float | None] = [None] * length
    for index in range(p - 1, length):
        atr_value = atr_series[index]
        if atr_value is None:
            continue
        window_high = [float(value) for value in high[index - p + 1 : index + 1]]
        window_low = [float(value) for value in low[index - p + 1 : index + 1]]
        if not all(math.isfinite(value) for value in window_high + window_low):
            continue
        first_high[index] = max(window_high) - x * atr_value
        first_low[index] = min(window_low) + x * atr_value
    long_stop: list[float | None] = [None] * length
    short_stop: list[float | None] = [None] * length
    for index in range(q - 1, length):
        high_window = first_high[index - q + 1 : index + 1]
        low_window = first_low[index - q + 1 : index + 1]
        if all(value is not None for value in high_window):
            long_stop[index] = max(value for value in high_window if value is not None)
        if all(value is not None for value in low_window):
            short_stop[index] = min(value for value in low_window if value is not None)
    return long_stop, short_stop


def rolling_median_previous(values: list[Number], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if window <= 0:
        raise ValueError("window must be positive")
    for index in range(window, len(values)):
        sample = [float(value) for value in values[index - window : index]]
        if all(math.isfinite(value) for value in sample):
            result[index] = float(median(sample))
    return result


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
