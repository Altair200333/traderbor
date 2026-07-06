from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.data import CandleFrame


Side = Literal["long", "short"]
PatternId = Literal["P1", "P1H", "P2", "P3"]


class PatternHit(BaseModel):
    id: PatternId
    side: Side
    boundary_price: float | None = None
    detail: dict[str, float | int | bool | None] = Field(default_factory=dict)


def detect_patterns(frame: CandleFrame, ema20: list[float | None], ema50: list[float | None], atr: list[float | None], cfg: ScreenerConfig) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for side in ("long", "short"):
        hits.extend(_side_patterns(side, frame, ema20, ema50, atr, cfg))
    return hits


def patterns_for_side(hits: list[PatternHit], side: Side) -> list[PatternHit]:
    return [hit for hit in hits if hit.side == side]


def _side_patterns(side: Side, frame: CandleFrame, ema20: list[float | None], ema50: list[float | None], atr: list[float | None], cfg: ScreenerConfig) -> list[PatternHit]:
    t = frame.length - 1
    close = frame.close[t]
    result: list[PatternHit] = []
    if t >= cfg.p1_lookback:
        high_boundary = max(frame.high[t - cfg.p1_lookback : t])
        low_boundary = min(frame.low[t - cfg.p1_lookback : t])
        boundary = high_boundary if side == "long" else low_boundary
        hit = close > boundary if side == "long" else close < boundary
        if hit:
            result.append(PatternHit(id="P1", side=side, boundary_price=boundary, detail={"lookback": cfg.p1_lookback}))
        accepted = _recent_p1_hold(side, frame, cfg)
        if accepted is not None:
            result.append(accepted)
    if t >= cfg.p2_trend_window - 1:
        start = t - cfg.p2_trend_window + 1
        trend_count = 0
        trend_known = True
        for index in range(start, t + 1):
            slow = ema50[index]
            if slow is None:
                trend_known = False
                break
            if (frame.close[index] > slow) if side == "long" else (frame.close[index] < slow):
                trend_count += 1
        required = math.ceil(cfg.p2_trend_share * cfg.p2_trend_window)
        touch = False
        for index in range(max(0, t - cfg.p2_pullback_bars + 1), t + 1):
            fast = ema20[index]
            if fast is None:
                continue
            if side == "long" and frame.low[index] <= fast:
                touch = True
            if side == "short" and frame.high[index] >= fast:
                touch = True
        continuation = frame.close[t] > frame.high[t - 1] if side == "long" else frame.close[t] < frame.low[t - 1]
        if trend_known and trend_count >= required and touch and continuation:
            boundary = min(frame.low[max(0, t - cfg.p2_pullback_bars + 1) : t + 1]) if side == "long" else max(frame.high[max(0, t - cfg.p2_pullback_bars + 1) : t + 1])
            result.append(PatternHit(id="P2", side=side, boundary_price=boundary, detail={"trend_count": trend_count, "trend_required": required, "touch": touch}))
    if t >= cfg.p3_range_bars and t >= cfg.p3_atr_lookback_bars:
        current_atr = atr[t]
        previous_atr = atr[t - cfg.p3_atr_lookback_bars]
        if current_atr is not None and previous_atr is not None:
            range_high = max(frame.high[t - cfg.p3_range_bars : t])
            range_low = min(frame.low[t - cfg.p3_range_bars : t])
            boundary = range_high if side == "long" else range_low
            compressed = current_atr <= cfg.p3_atr_compression * previous_atr
            breakout = close > boundary if side == "long" else close < boundary
            if compressed and breakout:
                result.append(PatternHit(id="P3", side=side, boundary_price=boundary, detail={"atr_now": current_atr, "atr_then": previous_atr, "compression": cfg.p3_atr_compression}))
    return result


def _recent_p1_hold(side: Side, frame: CandleFrame, cfg: ScreenerConfig) -> PatternHit | None:
    t = frame.length - 1
    close = frame.close[t]
    max_age = min(cfg.p1_hold_bars, t - cfg.p1_lookback)
    for age in range(1, max_age + 1):
        trigger = t - age
        high_boundary = max(frame.high[trigger - cfg.p1_lookback : trigger])
        low_boundary = min(frame.low[trigger - cfg.p1_lookback : trigger])
        boundary = high_boundary if side == "long" else low_boundary
        trigger_close = frame.close[trigger]
        triggered = trigger_close > boundary if side == "long" else trigger_close < boundary
        held = close > boundary if side == "long" else close < boundary
        if triggered and held:
            return PatternHit(
                id="P1H",
                side=side,
                boundary_price=boundary,
                detail={"lookback": cfg.p1_lookback, "age_bars": age, "trigger_close": trigger_close},
            )
    return None
