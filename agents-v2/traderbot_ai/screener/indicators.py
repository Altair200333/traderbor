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
