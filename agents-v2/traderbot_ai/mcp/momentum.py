from __future__ import annotations

import os
from statistics import median
from typing import Any, Literal

from traderbot_ai.simulator.clock import guarded_simulation_as_of
from traderbot_ai.simulator.market_cache import DEFAULT_CACHE_PATH, Candle, LocalMarketCache, candle_freshness
from traderbot_ai.tools.market import normalize_symbol


def scan_momentum_universe_impl(
    symbols: list[str] | str,
    as_of: str | int | float,
    decision_interval: str = "4h",
) -> dict[str, Any]:
    as_of_ms = guarded_simulation_as_of(as_of)
    cache = _cache()
    rows = []
    for symbol in _split_symbols(symbols):
        candles = cache.get_candles(symbol=symbol, interval="4h", as_of_ms=as_of_ms, limit=60)
        metrics = _coarse_metrics(candles, as_of_ms)
        side = _coarse_side(metrics)
        rows.append({"symbol": normalize_symbol(symbol), "side": side, **metrics})
    candidates = [row for row in rows if row["side"] in {"long", "short"}]
    candidates.sort(key=lambda row: abs(row.get("roc_4h") or 0.0) * max(row.get("coarse_4h_volume_ratio") or 0.0, 0.0), reverse=True)
    fresh_count = sum(1 for row in rows if row.get("data_fresh") is True)
    return {
        "ok": fresh_count > 0,
        "as_of_ms": as_of_ms,
        "decision_interval": decision_interval,
        "source": "local_cache",
        "fresh_symbol_count": fresh_count,
        "error": None if fresh_count > 0 else "no fresh 4h candles at as_of",
        "candidates": candidates,
        "rejected": [row for row in rows if row["side"] == "hold"],
    }


def get_candidate_detail_impl(symbol: str, side: Literal["long", "short"], as_of: str | int | float) -> dict[str, Any]:
    as_of_ms = guarded_simulation_as_of(as_of)
    cache = _cache()
    candles = cache.get_candles(symbol=symbol, interval="1h", as_of_ms=as_of_ms, limit=170)
    freshness = candle_freshness(candles[-1] if candles else None, "1h", as_of_ms)
    if not freshness["fresh"]:
        return {
            "ok": False,
            "symbol": normalize_symbol(symbol),
            "side": side,
            "as_of_ms": as_of_ms,
            "source": "local_cache",
            "freshness": freshness,
            "error": "stale 1h candles at as_of",
        }
    if len(candles) < 60:
        return {"ok": False, "symbol": normalize_symbol(symbol), "side": side, "as_of_ms": as_of_ms, "error": "not enough 1h candles"}
    closes = [candle.close for candle in candles]
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]
    volumes = [candle.volume for candle in candles]
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    atr = _atr(candles, 14)
    rsi = _rsi(closes, 14)
    close = closes[-1]
    roc_1h = _roc(closes, 1)
    roc_4h = _roc(closes, 4)
    volume_ratio = _ratio(volumes[-1], median(volumes[-25:-1]) if len(volumes) >= 25 else None)
    extension_atr = abs(close - ema20) / atr if atr and ema20 is not None else None
    trend_ok = close > ema50 if side == "long" else close < ema50
    pattern = _pattern(side, candles, ema20, ema50, atr)
    return {
        "ok": True,
        "symbol": normalize_symbol(symbol),
        "side": side,
        "as_of_ms": as_of_ms,
        "source": "local_cache",
        "close": close,
        "roc_1h": roc_1h,
        "roc_4h_from_1h": roc_4h,
        "last_1h_share_of_4h": (abs(roc_1h) / abs(roc_4h)) if roc_1h is not None and roc_4h not in (None, 0.0) else None,
        "volume_ratio_1h": volume_ratio,
        "rsi_14_1h": rsi,
        "atr_14_1h": atr,
        "atr_pct_of_price": (atr / close) if atr is not None and close > 0 else None,
        "ema20_1h": ema20,
        "ema50_1h": ema50,
        "extension_atr_vs_ema20": extension_atr,
        "trend_ok": trend_ok,
        "pattern": pattern,
        "recent_high_20": max(highs[-21:-1]),
        "recent_low_20": min(lows[-21:-1]),
    }


def _cache() -> LocalMarketCache:
    return LocalMarketCache(os.getenv("TRADERBOT_MARKET_CACHE_PATH") or DEFAULT_CACHE_PATH)


def _split_symbols(symbols: list[str] | str) -> list[str]:
    if isinstance(symbols, str):
        values = symbols.split(",")
    else:
        values = symbols
    return [normalize_symbol(symbol) for symbol in values if str(symbol).strip()]


def _coarse_metrics(candles: list[Candle], as_of_ms: int | None) -> dict[str, Any]:
    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]
    freshness = candle_freshness(candles[-1] if candles else None, "4h", as_of_ms)
    return {
        "candles_4h": len(candles),
        "close": closes[-1] if closes else None,
        "data_fresh": freshness["fresh"],
        "freshness": freshness,
        "roc_4h": _roc(closes, 1),
        "roc_24h": _roc(closes, 6),
        "coarse_4h_volume_ratio": _ratio(volumes[-1] if volumes else None, median(volumes[-21:-1]) if len(volumes) >= 21 else None),
    }


def _coarse_side(metrics: dict[str, Any]) -> str:
    if metrics.get("data_fresh") is not True:
        return "hold"
    roc_4h = metrics.get("roc_4h")
    roc_24h = metrics.get("roc_24h")
    if roc_4h is None or roc_24h is None:
        return "hold"
    if roc_4h >= 0.025 and roc_24h >= 0.02:
        return "long"
    if roc_4h <= -0.025 and roc_24h <= -0.02:
        return "short"
    return "hold"


def _roc(values: list[float], periods: int) -> float | None:
    if len(values) <= periods:
        return None
    previous = values[-1 - periods]
    if previous == 0:
        return None
    return (values[-1] - previous) / previous


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    alpha = 2 / (period + 1)
    value = sum(values[:period]) / period
    for price in values[period:]:
        value = alpha * price + (1 - alpha) * value
    return value


def _rsi(values: list[float], period: int) -> float | None:
    if len(values) <= period:
        return None
    gains = []
    losses = []
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    first = deltas[:period]
    gains.append(sum(max(delta, 0.0) for delta in first) / period)
    losses.append(sum(abs(min(delta, 0.0)) for delta in first) / period)
    avg_gain = gains[-1]
    avg_loss = losses[-1]
    for delta in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + abs(min(delta, 0.0))) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(candles: list[Candle], period: int) -> float | None:
    if len(candles) <= period:
        return None
    true_ranges = []
    for previous, current in zip(candles[:-1], candles[1:]):
        true_ranges.append(max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close)))
    first = true_ranges[:period]
    atr = sum(first) / period
    for value in true_ranges[period:]:
        atr = (atr * (period - 1) + value) / period
    return atr


def _pattern(side: str, candles: list[Candle], ema20: float | None, ema50: float | None, atr: float | None) -> dict[str, Any]:
    close = candles[-1].close
    high_20 = max(candle.high for candle in candles[-21:-1])
    low_20 = min(candle.low for candle in candles[-21:-1])
    breakout = close > high_20 if side == "long" else close < low_20
    bars_on_trend_side = 0
    if ema50 is not None:
        bars_on_trend_side = sum(1 for candle in candles[-24:] if (candle.close > ema50 if side == "long" else candle.close < ema50))
    ema20_touch = False
    if ema20 is not None:
        ema20_touch = any(candle.low <= ema20 <= candle.high for candle in candles[-3:])
    previous = candles[-2]
    continuation = bars_on_trend_side >= 20 and ema20_touch and (close > previous.high if side == "long" else close < previous.low)
    atr_now = atr
    atr_then = _atr(candles[:-72], 14) if len(candles) > 90 else None
    range_48h_high = max(candle.high for candle in candles[-49:-1])
    range_48h_low = min(candle.low for candle in candles[-49:-1])
    compression = atr_now is not None and atr_then is not None and atr_now <= 0.7 * atr_then and (close > range_48h_high if side == "long" else close < range_48h_low)
    labels = []
    if breakout:
        labels.append("P1")
    if continuation:
        labels.append("P2")
    if compression:
        labels.append("P3")
    return {
        "ids": labels,
        "p1_range_breakout": breakout,
        "p2_pullback_continuation": continuation,
        "p3_compression_breakout": compression,
    }
