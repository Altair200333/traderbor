from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.data import CandleFrame
from traderbot_ai.screener.indicators import ema, roc


class MaintenanceAction(BaseModel):
    action: Literal["close_position"]
    reason: Literal["max_hold", "impulse_break"]
    symbol: str
    side: str
    position_id: str | None = None


def check_max_hold(position: dict[str, Any], as_of_ms: int, cfg: ScreenerConfig) -> bool:
    opened_at = position.get("opened_at_ms")
    if opened_at is None:
        return False
    max_hold_ms = int(float(position.get("max_hold_hours") or cfg.max_hold_hours) * 60 * 60_000)
    return int(as_of_ms) - int(opened_at) >= max_hold_ms


def check_impulse_break(frame: CandleFrame, side: Literal["long", "short"], cfg: ScreenerConfig) -> bool:
    if frame.length < max(cfg.ema_fast + 2, 6):
        return False
    ema20 = ema(frame.close, cfg.ema_fast)
    roc4 = roc(frame.close, 4)
    t = frame.length - 1
    current_ema = ema20[t]
    previous_ema = ema20[t - 1]
    current_roc = roc4[t]
    if current_ema is None or previous_ema is None or current_roc is None:
        return False
    if side == "long":
        crossed = frame.close[t - 1] >= previous_ema and frame.close[t] < current_ema
        reversed_roc = current_roc <= -cfg.impulse_break_roc
    else:
        crossed = frame.close[t - 1] <= previous_ema and frame.close[t] > current_ema
        reversed_roc = current_roc >= cfg.impulse_break_roc
    return bool(crossed and reversed_roc)
