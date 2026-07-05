from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from traderbot_ai.simulator.clock import (
    clear_file_simulation_clock_state,
    clear_process_simulation_clock_state,
    file_simulation_clock_ms,
    process_simulation_clock_ms,
    set_file_simulation_clock_state,
    set_process_simulation_clock_state,
    set_simulation_clock_state,
)
from traderbot_ai.simulator.entry import resolve_entry_fill
from traderbot_ai.simulator.market_cache import LocalMarketCache
from traderbot_ai.simulator.portfolio import SimulatedPortfolio
from traderbot_ai.tools.market import INTERVAL_MS, normalize_symbol, parse_time_ms


DecisionFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    start_ms: int
    end_ms: int
    decision_interval: str = "15m"
    execution_interval: str = "1m"
    balance_usdt: float = 1000.0
    fee_rate: float = 0.0


def build_config(
    symbol: str,
    start_time: str | int | float,
    end_time: str | int | float,
    decision_interval: str = "15m",
    execution_interval: str = "1m",
    balance_usdt: float = 1000.0,
    fee_rate: float = 0.0,
) -> BacktestConfig:
    start_ms = parse_time_ms(start_time)
    end_ms = parse_time_ms(end_time)
    if start_ms is None or end_ms is None:
        raise ValueError("start_time and end_time are required")
    if start_ms >= end_ms:
        raise ValueError("start_time must be before end_time")
    if decision_interval not in INTERVAL_MS:
        raise ValueError(f"unsupported decision_interval: {decision_interval}")
    if execution_interval not in INTERVAL_MS:
        raise ValueError(f"unsupported execution_interval: {execution_interval}")
    return BacktestConfig(
        symbol=normalize_symbol(symbol),
        start_ms=start_ms,
        end_ms=end_ms,
        decision_interval=decision_interval,
        execution_interval=execution_interval,
        balance_usdt=balance_usdt,
        fee_rate=fee_rate,
    )


def run_backtest(
    config: BacktestConfig,
    decide: DecisionFn,
    cache: LocalMarketCache | None = None,
    portfolio: SimulatedPortfolio | None = None,
) -> dict[str, Any]:
    cache = cache or LocalMarketCache()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if portfolio is None:
        temp_dir = tempfile.TemporaryDirectory()
        portfolio = SimulatedPortfolio(Path(temp_dir.name) / "portfolio.json", cache=cache)
    portfolio.reset(balance_usdt=config.balance_usdt, as_of=config.start_ms)
    step_ms = INTERVAL_MS[config.decision_interval]
    as_of_ms = config.start_ms
    steps = []
    previous_clock = process_simulation_clock_ms()
    previous_file_clock = file_simulation_clock_ms()

    try:
        while as_of_ms < config.end_ms:
            set_simulation_clock_state(as_of_ms)
            settlement = portfolio.settle(as_of=as_of_ms, interval=config.execution_interval, fee_rate=config.fee_rate)
            summary = portfolio.summary(as_of=as_of_ms, mark_interval=config.execution_interval)
            context = {
                "symbol": config.symbol,
                "as_of_ms": as_of_ms,
                "as_of_iso": _iso_ms(as_of_ms),
                "decision_interval": config.decision_interval,
                "execution_interval": config.execution_interval,
                "portfolio": summary,
            }
            decision = decide(context)
            order_result = None
            if decision.get("final_decision") in {"long", "short"}:
                try:
                    entry_fill = resolve_entry_fill(
                        cache=cache,
                        symbol=decision.get("symbol") or config.symbol,
                        as_of=as_of_ms,
                        interval=config.execution_interval,
                        requested_price=float(decision["price"]),
                        before=min(as_of_ms + step_ms, config.end_ms),
                    )
                    if entry_fill.opened_at_ms >= config.end_ms:
                        raise ValueError("entry fill is outside the backtest window")
                    order_result = portfolio.place_order(
                        final_decision=decision["final_decision"],
                        symbol=entry_fill.symbol,
                        price=entry_fill.fill_price,
                        stop_loss=float(decision["stop_loss"]),
                        take_profit=float(decision["take_profit"]),
                        amount=float(decision["amount"]),
                        opened_at=entry_fill.opened_at_ms,
                    )
                    order_result["entry_fill"] = entry_fill.to_dict()
                except Exception as error:
                    order_result = {"ok": False, "error": str(error)}
            steps.append(
                {
                    "as_of_ms": as_of_ms,
                    "as_of_iso": _iso_ms(as_of_ms),
                    "settlement": settlement,
                    "decision": decision,
                    "order_result": order_result,
                    "portfolio": portfolio.summary(as_of=as_of_ms, mark_interval=config.execution_interval),
                }
            )
            as_of_ms += step_ms

        set_simulation_clock_state(config.end_ms)
        final_settlement = portfolio.settle(
            as_of=config.end_ms,
            interval=config.execution_interval,
            fee_rate=config.fee_rate,
        )
        final_summary = portfolio.summary(as_of=config.end_ms, mark_interval=config.execution_interval)
        return {
            "config": {
                "symbol": config.symbol,
                "start_ms": config.start_ms,
                "end_ms": config.end_ms,
                "decision_interval": config.decision_interval,
                "execution_interval": config.execution_interval,
                "balance_usdt": config.balance_usdt,
                "fee_rate": config.fee_rate,
            },
            "steps": steps,
            "final_settlement": final_settlement,
            "final_portfolio": final_summary,
        }
    finally:
        if previous_file_clock is None:
            clear_file_simulation_clock_state()
        else:
            set_file_simulation_clock_state(previous_file_clock)
        if previous_clock is None:
            clear_process_simulation_clock_state()
        else:
            set_process_simulation_clock_state(previous_clock)
        if temp_dir is not None:
            temp_dir.cleanup()


def _iso_ms(timestamp_ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()
