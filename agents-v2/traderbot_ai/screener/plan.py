from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.data import CandleFrame
from traderbot_ai.screener.indicators import chande_kroll_stop, median_range_pct
from traderbot_ai.screener.patterns import PatternHit, Side


class PlanPrimitives(BaseModel):
    pattern_used: str
    boundary_price: float | None = None
    trigger_age_bars: int | None = None
    invalidation_price: float
    d_atr: float
    d_struct: float
    d_noise: float | None = None
    d_cksp: float | None = None
    d_final: float
    stop_feasible: bool
    tp_rr_default: float
    ref_entry: float


def noise_floor_stop_pct(close: float, atr_value: float | None, median_range: float | None, cfg: ScreenerConfig) -> float | None:
    """Minimum honest stop distance: below this the stop sits inside typical bar noise."""
    if close == 0:
        return None
    candidates = [cfg.stop_pct_min]
    if atr_value is not None:
        candidates.append(cfg.stop_atr_floor_mult * abs(atr_value) / abs(close))
    if median_range is not None:
        candidates.append(cfg.stop_noise_mult * median_range)
    return max(candidates)


def build_plan_primitives(side: Side, patterns: list[PatternHit], frame: CandleFrame, atr_value: float, cfg: ScreenerConfig) -> PlanPrimitives | None:
    chosen = _choose_pattern(patterns)
    if chosen is None:
        return None
    t = frame.length - 1
    close = frame.close[t]
    trigger_age_bars = _trigger_age_bars(chosen)
    if chosen.id == "P2":
        if side == "long":
            base = min(frame.low[max(0, t - cfg.p2_pullback_bars + 1) : t + 1])
            invalidation = base - cfg.struct_buffer_p2_atr * atr_value
        else:
            base = max(frame.high[max(0, t - cfg.p2_pullback_bars + 1) : t + 1])
            invalidation = base + cfg.struct_buffer_p2_atr * atr_value
    else:
        boundary = chosen.boundary_price
        if boundary is None:
            return None
        invalidation = boundary - cfg.struct_buffer_p1_p3_atr * atr_value if side == "long" else boundary + cfg.struct_buffer_p1_p3_atr * atr_value
    d_atr = cfg.stop_atr_mult_default * atr_value / close
    d_struct = abs(close - invalidation) / close
    d_noise = noise_floor_stop_pct(close, atr_value, median_range_pct(frame.high, frame.low, frame.close, cfg.stop_noise_median_window), cfg)
    d_final = max(value for value in (d_atr, d_struct, d_noise) if value is not None)
    return PlanPrimitives(
        pattern_used=chosen.id,
        boundary_price=chosen.boundary_price,
        trigger_age_bars=trigger_age_bars,
        invalidation_price=invalidation,
        d_atr=d_atr,
        d_struct=d_struct,
        d_noise=d_noise,
        d_cksp=_cksp_stop_pct(side, frame, close, cfg),
        d_final=d_final,
        stop_feasible=cfg.stop_pct_min <= d_final <= cfg.stop_pct_max,
        tp_rr_default=float(cfg.tp_rr_default.get(chosen.id, 2.0)),
        ref_entry=close,
    )


def _cksp_stop_pct(side: Side, frame: CandleFrame, close: float, cfg: ScreenerConfig) -> float | None:
    """Chande Kroll stop distance, recorded for offline stop-engine comparison (not enforced)."""
    if close == 0:
        return None
    long_stop, short_stop = chande_kroll_stop(frame.high, frame.low, frame.close, p=cfg.cksp_p, x=cfg.cksp_x, q=cfg.cksp_q)
    level = long_stop[-1] if side == "long" else short_stop[-1]
    if level is None:
        return None
    distance = (close - level) / abs(close) if side == "long" else (level - close) / abs(close)
    return max(0.0, distance)


def _choose_pattern(patterns: list[PatternHit]) -> PatternHit | None:
    priority: tuple[Literal["P2"], Literal["P1"], Literal["P1H"], Literal["P3"]] = ("P2", "P1", "P1H", "P3")
    for pattern_id in priority:
        for pattern in patterns:
            if pattern.id == pattern_id:
                return pattern
    return None


def _trigger_age_bars(pattern: PatternHit) -> int | None:
    if pattern.id == "P1H":
        value = pattern.detail.get("age_bars")
        return int(value) if value is not None else None
    if pattern.id in {"P1", "P3"}:
        return 0
    return None
