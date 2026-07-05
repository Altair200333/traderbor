from __future__ import annotations

import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import BaseModel

from traderbot_ai.exchange import SimulatedExchange
from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs
from traderbot_ai.simulator.clock import (
    clear_file_simulation_clock_state,
    clear_process_simulation_clock_state,
    file_simulation_clock_ms,
    process_simulation_clock_ms,
    set_file_simulation_clock_state,
    set_process_simulation_clock_state,
    set_simulation_clock_state,
)
from traderbot_ai.simulator.market_cache import DEFAULT_CACHE_PATH, LocalMarketCache
from traderbot_ai.tools.market import INTERVAL_MS, normalize_symbol, parse_time_ms


ExchangeReplayDecisionFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ExchangeReplayConfig:
    symbols: tuple[str, ...]
    start_ms: int
    end_ms: int
    decision_interval: str = "4h"
    execution_interval: str = "1m"
    balances: dict[str, float] | None = None
    fee_rate: float = 0.0
    linear_leverage: float | None = None
    run_id: str = ""
    state_path: Path | None = None
    events_path: Path | None = None
    replay_path: Path | None = None


def build_exchange_replay_config(
    symbols: str | list[str] | tuple[str, ...],
    start_time: str | int | float,
    end_time: str | int | float,
    decision_interval: str = "4h",
    execution_interval: str = "1m",
    balance_usdt: float = 1000.0,
    balances: dict[str, float] | None = None,
    fee_rate: float = 0.0,
    linear_leverage: float | None = None,
    run_id: str | None = None,
    state_path: str | Path | None = None,
    events_path: str | Path | None = None,
    replay_path: str | Path | None = None,
) -> ExchangeReplayConfig:
    normalized_symbols = tuple(normalize_symbol(symbol) for symbol in _split_symbols(symbols))
    if not normalized_symbols:
        raise ValueError("at least one symbol is required")
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
    fee_rate_value = float(fee_rate)
    if not math.isfinite(fee_rate_value) or fee_rate_value < 0:
        raise ValueError("fee_rate must be non-negative")
    linear_leverage_value = None if linear_leverage is None else float(linear_leverage)
    if linear_leverage_value is not None and (not math.isfinite(linear_leverage_value) or linear_leverage_value < 1):
        raise ValueError("linear_leverage must be at least 1")

    ensure_runtime_dirs()
    replay_run_id = run_id or _default_run_id(normalized_symbols, start_ms, end_ms)
    safe_run_id = _safe_file_part(replay_run_id)
    return ExchangeReplayConfig(
        symbols=normalized_symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        decision_interval=decision_interval,
        execution_interval=execution_interval,
        balances={asset.upper(): float(amount) for asset, amount in (balances if balances is not None else {"USDT": balance_usdt}).items()},
        fee_rate=fee_rate_value,
        linear_leverage=linear_leverage_value,
        run_id=replay_run_id,
        state_path=Path(state_path) if state_path is not None else DATA_DIR / f"{safe_run_id}.exchange.json",
        events_path=Path(events_path) if events_path is not None else DATA_DIR / f"{safe_run_id}.exchange.events.jsonl",
        replay_path=Path(replay_path) if replay_path is not None else DATA_DIR / f"{safe_run_id}.replay.jsonl",
    )


def run_exchange_replay(
    config: ExchangeReplayConfig,
    decide: ExchangeReplayDecisionFn,
    cache: LocalMarketCache | None = None,
    exchange: SimulatedExchange | None = None,
) -> dict[str, Any]:
    if exchange is not None and Path(exchange.path).resolve() != Path(_required_path(config.state_path)).resolve():
        raise ValueError("exchange state path must match replay config state path")
    if exchange is not None and Path(exchange.events_path).resolve() != Path(_required_path(config.events_path)).resolve():
        raise ValueError("exchange events path must match replay config events path")
    if exchange is not None and cache is not None and Path(exchange.cache.path).resolve() != Path(cache.path).resolve():
        raise ValueError("exchange cache path must match replay cache path")
    cache = cache or (exchange.cache if exchange is not None else LocalMarketCache(os.getenv("TRADERBOT_MARKET_CACHE_PATH") or DEFAULT_CACHE_PATH))
    exchange = exchange or SimulatedExchange(path=_required_path(config.state_path), events_path=_required_path(config.events_path), cache=cache)
    replay_path = _required_path(config.replay_path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    if replay_path.exists():
        raise ValueError(f"replay_path already exists: {replay_path}")

    exchange.reset(config.balances if config.balances is not None else {"USDT": 1000.0}, as_of=config.start_ms)
    if config.linear_leverage is not None:
        for symbol in config.symbols:
            exchange.set_leverage("linear", symbol, str(config.linear_leverage), str(config.linear_leverage))

    _append_replay_event(replay_path, "replay_started", {"config": _config_dict(config)})
    previous_clock = process_simulation_clock_ms()
    previous_file_clock = file_simulation_clock_ms()
    steps = []
    step_ms = INTERVAL_MS[config.decision_interval]
    as_of_ms = config.start_ms

    try:
        with exchange_tool_environment(
            exchange.path,
            exchange.events_path,
            backend="simulated",
            fee_rate=config.fee_rate,
            cache_path=cache.path,
            execution_interval=config.execution_interval,
        ):
            while as_of_ms < config.end_ms:
                set_simulation_clock_state(as_of_ms)
                settlement = exchange.settle(as_of=as_of_ms, interval=config.execution_interval, fee_rate=config.fee_rate)
                wallet_before = exchange.wallet_summary(symbols=list(config.symbols), as_of=as_of_ms, mark_interval=config.execution_interval)
                context = {
                    "run_id": config.run_id,
                    "symbols": list(config.symbols),
                    "as_of_ms": as_of_ms,
                    "as_of_iso": _iso_ms(as_of_ms),
                    "next_as_of_ms": min(as_of_ms + step_ms, config.end_ms),
                    "decision_interval": config.decision_interval,
                    "execution_interval": config.execution_interval,
                    "fee_rate": config.fee_rate,
                    "wallet": wallet_before,
                    "settlement": settlement,
                    "exchange_state_path": str(exchange.path),
                    "exchange_events_path": str(exchange.events_path),
                    "replay_path": str(replay_path),
                }
                _append_replay_event(replay_path, "step_started", _step_header(context))
                exchange_event_cursor = _file_size(exchange.events_path)
                decision = _jsonable(decide(context))
                agent_exchange_events = _read_jsonl_since(exchange.events_path, exchange_event_cursor)
                wallet_after = exchange.wallet_summary(symbols=list(config.symbols), as_of=as_of_ms, mark_interval=config.execution_interval)
                step = {
                    "as_of_ms": as_of_ms,
                    "as_of_iso": _iso_ms(as_of_ms),
                    "settlement": _compact_settlement(settlement),
                    "wallet_before": _compact_wallet(wallet_before),
                    "decision": decision,
                    "agent_exchange_events": agent_exchange_events,
                    "wallet_after": _compact_wallet(wallet_after),
                }
                steps.append(step)
                _append_replay_event(replay_path, "step_completed", step)
                as_of_ms += step_ms

            set_simulation_clock_state(config.end_ms)
            final_settlement = exchange.settle(as_of=config.end_ms, interval=config.execution_interval, fee_rate=config.fee_rate)
            final_wallet = exchange.wallet_summary(symbols=list(config.symbols), as_of=config.end_ms, mark_interval=config.execution_interval)
            result = {
                "ok": True,
                "config": _config_dict(config),
                "steps": steps,
                "final_settlement": _compact_settlement(final_settlement),
                "final_wallet": final_wallet,
                "exchange_state_path": str(exchange.path),
                "exchange_events_path": str(exchange.events_path),
                "replay_path": str(replay_path),
            }
            _append_replay_event(replay_path, "replay_completed", {"final_wallet": _compact_wallet(final_wallet)})
            return result
    except BaseException as error:
        _append_replay_event(replay_path, "replay_failed", {"error": str(error), "error_type": type(error).__name__})
        raise
    finally:
        if previous_file_clock is None:
            clear_file_simulation_clock_state()
        else:
            set_file_simulation_clock_state(previous_file_clock)
        if previous_clock is None:
            clear_process_simulation_clock_state()
        else:
            set_process_simulation_clock_state(previous_clock)


@contextmanager
def exchange_tool_environment(
    state_path: str | Path,
    events_path: str | Path,
    backend: str | None = None,
    fee_rate: float | None = None,
    cache_path: str | Path | None = None,
    execution_interval: str | None = None,
) -> Iterator[None]:
    old_backend = os.environ.get("TRADERBOT_EXCHANGE_BACKEND")
    old_state = os.environ.get("TRADERBOT_EXCHANGE_STATE_PATH")
    old_events = os.environ.get("TRADERBOT_EXCHANGE_EVENTS_PATH")
    old_fee_rate = os.environ.get("TRADERBOT_EXCHANGE_FEE_RATE")
    old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
    old_execution_interval = os.environ.get("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL")
    if backend is not None:
        os.environ["TRADERBOT_EXCHANGE_BACKEND"] = backend
    else:
        os.environ.pop("TRADERBOT_EXCHANGE_BACKEND", None)
    os.environ["TRADERBOT_EXCHANGE_STATE_PATH"] = str(state_path)
    os.environ["TRADERBOT_EXCHANGE_EVENTS_PATH"] = str(events_path)
    if fee_rate is not None:
        os.environ["TRADERBOT_EXCHANGE_FEE_RATE"] = str(float(fee_rate))
    else:
        os.environ.pop("TRADERBOT_EXCHANGE_FEE_RATE", None)
    if cache_path is not None:
        os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache_path)
    else:
        os.environ.pop("TRADERBOT_MARKET_CACHE_PATH", None)
    if execution_interval is not None:
        os.environ["TRADERBOT_EXCHANGE_EXECUTION_INTERVAL"] = execution_interval
    else:
        os.environ.pop("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", None)
    try:
        yield
    finally:
        _restore_env("TRADERBOT_EXCHANGE_BACKEND", old_backend)
        _restore_env("TRADERBOT_EXCHANGE_STATE_PATH", old_state)
        _restore_env("TRADERBOT_EXCHANGE_EVENTS_PATH", old_events)
        _restore_env("TRADERBOT_EXCHANGE_FEE_RATE", old_fee_rate)
        _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)
        _restore_env("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", old_execution_interval)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _required_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("replay path is required")
    return Path(path)


def _split_symbols(symbols: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(symbols, str):
        values = symbols.split(",")
    else:
        values = list(symbols)
    return [str(item).strip() for item in values if str(item).strip()]


def _default_run_id(symbols: tuple[str, ...], start_ms: int, end_ms: int) -> str:
    symbol_part = "-".join(symbol.lower() for symbol in symbols[:4])
    return f"exchange-replay-{symbol_part}-{start_ms}-{end_ms}-{uuid.uuid4().hex[:8]}"


def _safe_file_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return cleaned or f"exchange-replay-{uuid.uuid4().hex[:8]}"


def _append_replay_event(path: Path, event_type: str, payload: dict[str, Any]) -> None:
    event = {
        "event_id": f"replay-{uuid.uuid4().hex}",
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": _jsonable(payload),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _read_jsonl_since(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for line in handle.read().splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line.decode("utf-8")))
            except json.JSONDecodeError:
                continue
    return events


def _config_dict(config: ExchangeReplayConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "symbols": list(config.symbols),
        "start_ms": config.start_ms,
        "end_ms": config.end_ms,
        "decision_interval": config.decision_interval,
        "execution_interval": config.execution_interval,
        "balances": config.balances,
        "fee_rate": config.fee_rate,
        "linear_leverage": config.linear_leverage,
        "state_path": str(config.state_path),
        "events_path": str(config.events_path),
        "replay_path": str(config.replay_path),
    }


def _step_header(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": context["run_id"],
        "symbols": context["symbols"],
        "as_of_ms": context["as_of_ms"],
        "as_of_iso": context["as_of_iso"],
        "next_as_of_ms": context["next_as_of_ms"],
    }


def _compact_settlement(settlement: dict[str, Any]) -> dict[str, Any]:
    return {
        "settled_until_ms": settlement.get("settled_until_ms"),
        "filled_orders": settlement.get("filled_orders", []),
        "closed_positions": settlement.get("closed_positions", []),
    }


def _compact_wallet(wallet: dict[str, Any]) -> dict[str, Any]:
    return {
        "as_of_ms": wallet.get("as_of_ms"),
        "totals": wallet.get("totals", {}),
        "balances": wallet.get("balances", []),
        "open_orders": wallet.get("open_orders", []),
        "open_positions": wallet.get("open_positions", []),
        "leverage": wallet.get("leverage", {}),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _iso_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()
