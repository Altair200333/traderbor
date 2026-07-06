from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.market import normalize_symbol
from traderbot_ai.screener.patterns import PatternHit, Side
from traderbot_ai.screener.state import TradingState


class GateResult(BaseModel):
    passed: bool | None
    value: float | str | None = None
    threshold: float | str | None = None
    reason: str | None = None


def evaluate_signal_gates(
    side: Side,
    *,
    close: float,
    roc_4h: float | None,
    roc_24h: float | None,
    roc_1h_last: float | None,
    vol_ratio: float | None,
    rsi: float | None,
    atr: float | None,
    atr_pct: float | None,
    ema20: float | None,
    ema50: float | None,
    btc_roc_4h: float | None,
    funding: float | None,
    patterns: list[PatternHit],
    cfg: ScreenerConfig,
) -> dict[str, GateResult]:
    return {
        "S1": _threshold("S1", roc_4h, cfg.roc_4h_long if side == "long" else cfg.roc_4h_short, ">=" if side == "long" else "<="),
        "S2": _threshold("S2", roc_24h, cfg.roc_24h_long if side == "long" else cfg.roc_24h_short, ">=" if side == "long" else "<="),
        "S3": _threshold("S3", vol_ratio, cfg.vol_ratio_min, ">="),
        "S4": _range("S4", rsi, cfg.rsi_long if side == "long" else cfg.rsi_short),
        "S5": _range("S5", atr_pct, (cfg.atr_pct_min, cfg.atr_pct_max)),
        "S6": _trend(side, close, ema50),
        "S7": _threshold("S7", btc_roc_4h, cfg.btc_roc_4h_block_long if side == "long" else cfg.btc_roc_4h_block_short, ">=" if side == "long" else "<="),
        "S8": _funding(side, funding, cfg),
        "S9a": _last_hour_share(roc_1h_last, roc_4h, cfg),
        "S9b": _extension(close, ema20, atr, cfg),
        "S9c": _breakout_distance(side, close, atr, patterns, cfg),
        "PAT": GateResult(passed=bool(patterns), value=",".join(pattern.id for pattern in patterns) if patterns else None, threshold="pattern"),
    }


def state_blocks(symbol: str, side: Side, state: TradingState, as_of_ms: int, cfg: ScreenerConfig) -> tuple[list[str], list[str]]:
    normalized = normalize_symbol(symbol)
    per_symbol = []
    global_blocks = scan_global_blocks(state, cfg)
    same_direction = sum(1 for position in state.open_positions if position.strategy_side == side)
    if same_direction >= cfg.max_same_direction:
        per_symbol.append("max_same_direction")
    if any(normalize_symbol(position.symbol) == normalized for position in state.open_positions):
        per_symbol.append("position_open")
    last_candidate = state.last_candidate_ts.get(normalized)
    if last_candidate is not None and int(as_of_ms) - int(last_candidate) < cfg.cooldown_candidate_ms:
        per_symbol.append("cooldown_candidate")
    last_stopout = state.last_stopout_ts.get(normalized)
    if last_stopout is not None and int(as_of_ms) - int(last_stopout) < cfg.cooldown_stopout_ms:
        per_symbol.append("cooldown_stopout")
    if state.consecutive_stopouts >= cfg.stopout_pause_count:
        global_blocks.append("stopout_pause")
    if state.daily_realized_pnl_pct <= cfg.daily_loss_limit_pct:
        global_blocks.append("daily_loss_limit")
    if state.weekly_realized_pnl_pct <= cfg.weekly_loss_limit_pct:
        global_blocks.append("weekly_loss_limit")
    return per_symbol, sorted(set(global_blocks))


def scan_global_blocks(state: TradingState, cfg: ScreenerConfig) -> list[str]:
    global_blocks = []
    if state.halt:
        global_blocks.append("halt_active")
    if state.trades_opened_today >= cfg.max_trades_per_day:
        global_blocks.append("max_trades_per_day")
    if len(state.open_positions) >= cfg.max_open_positions:
        global_blocks.append("max_positions")
    if state.consecutive_stopouts >= cfg.stopout_pause_count:
        global_blocks.append("stopout_pause")
    if state.daily_realized_pnl_pct <= cfg.daily_loss_limit_pct:
        global_blocks.append("daily_loss_limit")
    if state.weekly_realized_pnl_pct <= cfg.weekly_loss_limit_pct:
        global_blocks.append("weekly_loss_limit")
    return sorted(set(global_blocks))


def gates_pass(gates: dict[str, GateResult]) -> bool:
    for name, gate in gates.items():
        if name == "S8" and gate.passed is None:
            continue
        if gate.passed is not True:
            return False
    return True


def failed_gate_names(gates: dict[str, GateResult]) -> list[str]:
    return [name for name, gate in gates.items() if gate.passed is False]


def _threshold(name: str, value: float | None, threshold: float, op: Literal[">=", "<="]) -> GateResult:
    if value is None:
        return GateResult(passed=False, threshold=threshold, reason="missing")
    passed = value >= threshold if op == ">=" else value <= threshold
    return GateResult(passed=passed, value=value, threshold=threshold, reason=None if passed else f"{name}_threshold")


def _range(name: str, value: float | None, bounds: tuple[float, float]) -> GateResult:
    if value is None:
        return GateResult(passed=False, threshold=f"{bounds[0]}..{bounds[1]}", reason="missing")
    passed = bounds[0] <= value <= bounds[1]
    return GateResult(passed=passed, value=value, threshold=f"{bounds[0]}..{bounds[1]}", reason=None if passed else f"{name}_range")


def _trend(side: Side, close: float, ema50: float | None) -> GateResult:
    if ema50 is None:
        return GateResult(passed=False, reason="missing")
    passed = close > ema50 if side == "long" else close < ema50
    return GateResult(passed=passed, value=close, threshold=ema50, reason=None if passed else "S6_trend")


def _funding(side: Side, funding: float | None, cfg: ScreenerConfig) -> GateResult:
    if funding is None:
        return GateResult(passed=None, reason="funding_missing")
    threshold = cfg.funding_long_max if side == "long" else cfg.funding_short_min
    passed = funding <= threshold if side == "long" else funding >= threshold
    return GateResult(passed=passed, value=funding, threshold=threshold, reason=None if passed else "S8_funding")


def _last_hour_share(roc_1h_last: float | None, roc_4h: float | None, cfg: ScreenerConfig) -> GateResult:
    if roc_1h_last is None or roc_4h is None or roc_4h == 0:
        return GateResult(passed=False, reason="missing")
    value = abs(roc_1h_last) / abs(roc_4h)
    passed = value <= cfg.last_hour_share_max
    return GateResult(passed=passed, value=value, threshold=cfg.last_hour_share_max, reason=None if passed else "S9a_spike")


def _extension(close: float, ema20: float | None, atr: float | None, cfg: ScreenerConfig) -> GateResult:
    if ema20 is None or atr is None or atr == 0:
        return GateResult(passed=False, reason="missing")
    value = abs(close - ema20) / atr
    passed = value <= cfg.ext_atr_max
    return GateResult(passed=passed, value=value, threshold=cfg.ext_atr_max, reason=None if passed else "S9b_extension")


def _breakout_distance(side: Side, close: float, atr: float | None, patterns: list[PatternHit], cfg: ScreenerConfig) -> GateResult:
    breakout_patterns = [pattern for pattern in patterns if pattern.id in {"P1", "P3"} and pattern.boundary_price is not None]
    if not breakout_patterns:
        return GateResult(passed=True, reason="not_applicable")
    if atr is None or atr == 0:
        return GateResult(passed=False, reason="missing")
    distances = []
    for pattern in breakout_patterns:
        if pattern.boundary_price is None:
            continue
        boundary = float(pattern.boundary_price)
        distances.append((close - boundary) / atr if side == "long" else (boundary - close) / atr)
    value = max(distances)
    passed = value <= cfg.breakout_dist_atr_max
    return GateResult(passed=passed, value=value, threshold=cfg.breakout_dist_atr_max, reason=None if passed else "S9c_breakout_extension")
