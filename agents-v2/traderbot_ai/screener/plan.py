from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.data import CandleFrame
from traderbot_ai.screener.patterns import PatternHit, Side


class PlanPrimitives(BaseModel):
    pattern_used: str
    boundary_price: float | None = None
    invalidation_price: float
    d_atr: float
    d_struct: float
    d_final: float
    stop_feasible: bool
    tp_rr_default: float
    ref_entry: float


def build_plan_primitives(side: Side, patterns: list[PatternHit], frame: CandleFrame, atr_value: float, cfg: ScreenerConfig) -> PlanPrimitives | None:
    chosen = _choose_pattern(patterns)
    if chosen is None:
        return None
    t = frame.length - 1
    close = frame.close[t]
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
    d_final = max(d_atr, d_struct)
    return PlanPrimitives(
        pattern_used=chosen.id,
        boundary_price=chosen.boundary_price,
        invalidation_price=invalidation,
        d_atr=d_atr,
        d_struct=d_struct,
        d_final=d_final,
        stop_feasible=cfg.stop_pct_min <= d_final <= cfg.stop_pct_max,
        tp_rr_default=float(cfg.tp_rr_default.get(chosen.id, 2.0)),
        ref_entry=close,
    )


def _choose_pattern(patterns: list[PatternHit]) -> PatternHit | None:
    priority: tuple[Literal["P2"], Literal["P1"], Literal["P3"]] = ("P2", "P1", "P3")
    for pattern_id in priority:
        for pattern in patterns:
            if pattern.id == pattern_id:
                return pattern
    return None
