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
from traderbot_ai.screener import ScreenerConfig, ScreenerStateStore, TradingState, scan as run_screener
from traderbot_ai.screener.artifacts import write_scan_artifacts
from traderbot_ai.screener.data import load_closed, validate_frame
from traderbot_ai.screener.maintenance import MaintenanceAction, check_impulse_break, check_max_hold
from traderbot_ai.screener.render import to_markdown_table
from traderbot_ai.screener.state import state_from_wallet_and_events
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
    screener_mode: str = "off"
    entry_policy: str = "next_open"
    retest_pullback: float = 0.4
    retest_ttl_min: int = 120


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
    screener_mode: str = "off",
    entry_policy: str = "next_open",
    retest_pullback: float = 0.4,
    retest_ttl_min: int = 120,
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
    if screener_mode not in {"off", "deterministic", "legacy-self-screen"}:
        raise ValueError("screener_mode must be off, deterministic, or legacy-self-screen")
    if screener_mode == "deterministic" and decision_interval != "1h":
        raise ValueError("deterministic screener replay requires decision_interval=1h")
    if screener_mode == "deterministic":
        _validate_deterministic_replay_timing(start_ms, end_ms, decision_interval)
    if entry_policy not in {"next_open", "limit_retest"}:
        raise ValueError("entry_policy must be next_open or limit_retest")
    if entry_policy == "limit_retest" and screener_mode != "deterministic":
        raise ValueError("limit_retest entry policy requires screener_mode=deterministic")
    retest_pullback_value = float(retest_pullback)
    if not (0.0 < retest_pullback_value < 1.0):
        raise ValueError("retest_pullback must be between 0 and 1")
    retest_ttl_value = int(retest_ttl_min)
    if retest_ttl_value < 1:
        raise ValueError("retest_ttl_min must be at least 1 minute")
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
        screener_mode=screener_mode,
        entry_policy=entry_policy,
        retest_pullback=retest_pullback_value,
        retest_ttl_min=retest_ttl_value,
    )


def run_exchange_replay(
    config: ExchangeReplayConfig,
    decide: ExchangeReplayDecisionFn,
    cache: LocalMarketCache | None = None,
    exchange: SimulatedExchange | None = None,
) -> dict[str, Any]:
    if config.screener_mode == "deterministic" and config.decision_interval != "1h":
        raise ValueError("deterministic screener replay requires decision_interval=1h")
    if config.screener_mode == "deterministic":
        _validate_deterministic_replay_timing(config.start_ms, config.end_ms, config.decision_interval)
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
    screener_cfg = ScreenerConfig()
    screener_state = ScreenerStateStore()
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
            entry_policy=config.entry_policy if config.entry_policy != "next_open" else None,
            retest_pullback=config.retest_pullback,
            retest_ttl_min=config.retest_ttl_min,
        ):
            while as_of_ms < config.end_ms:
                set_simulation_clock_state(as_of_ms)
                settlement = exchange.settle(as_of=as_of_ms, interval=config.execution_interval, fee_rate=config.fee_rate)
                wallet_before = exchange.wallet_summary(symbols=list(config.symbols), as_of=as_of_ms, mark_interval=config.execution_interval)
                maintenance_cursor = _file_size(exchange.events_path)
                maintenance_actions: list[dict[str, Any]] = []
                maintenance_warnings: list[str] = []
                maintenance_exchange_events: list[dict[str, Any]] = []
                if config.screener_mode == "deterministic":
                    maintenance_actions, maintenance_warnings = _apply_runner_maintenance(exchange, cache, wallet_before, as_of_ms, config.fee_rate, screener_cfg)
                    maintenance_exchange_events = _read_jsonl_since(exchange.events_path, maintenance_cursor)
                    if maintenance_actions:
                        wallet_before = exchange.wallet_summary(symbols=list(config.symbols), as_of=as_of_ms, mark_interval=config.execution_interval)
                scan_context = _build_deterministic_scan_context(config, cache, wallet_before, as_of_ms, screener_cfg, screener_state) if config.screener_mode == "deterministic" else {}
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
                    "maintenance_actions": maintenance_actions,
                    "maintenance_warnings": maintenance_warnings,
                    **scan_context,
                    "exchange_state_path": str(exchange.path),
                    "exchange_events_path": str(exchange.events_path),
                    "replay_path": str(replay_path),
                }
                _append_replay_event(replay_path, "step_started", _step_header(context))
                exchange_event_cursor = _file_size(exchange.events_path)
                if config.screener_mode == "deterministic" and not context.get("screener_candidates"):
                    decision = _auto_hold_decision(context)
                else:
                    if config.screener_mode == "deterministic":
                        with _deterministic_candidate_allowlist_env(context):
                            decision = _jsonable(decide(context))
                    else:
                        decision = _jsonable(decide(context))
                agent_exchange_events = _read_jsonl_since(exchange.events_path, exchange_event_cursor)
                _validate_decision_exchange_consistency(
                    decision,
                    agent_exchange_events,
                    allow_hold_position_closes=config.screener_mode != "deterministic",
                    allowed_entry_candidates=_allowed_entry_candidates(context) if config.screener_mode == "deterministic" else None,
                    strict_entry_events=config.screener_mode == "deterministic",
                    expected_entry_policy=config.entry_policy,
                )
                wallet_after = exchange.wallet_summary(symbols=list(config.symbols), as_of=as_of_ms, mark_interval=config.execution_interval)
                step = {
                    "as_of_ms": as_of_ms,
                    "as_of_iso": _iso_ms(as_of_ms),
                    "settlement": _compact_settlement(settlement),
                    "maintenance_actions": maintenance_actions,
                    "maintenance_warnings": maintenance_warnings,
                    "maintenance_exchange_events": maintenance_exchange_events,
                    "scan": _compact_scan_context(context),
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
    entry_policy: str | None = None,
    retest_pullback: float | None = None,
    retest_ttl_min: int | None = None,
) -> Iterator[None]:
    old_backend = os.environ.get("TRADERBOT_EXCHANGE_BACKEND")
    old_state = os.environ.get("TRADERBOT_EXCHANGE_STATE_PATH")
    old_events = os.environ.get("TRADERBOT_EXCHANGE_EVENTS_PATH")
    old_fee_rate = os.environ.get("TRADERBOT_EXCHANGE_FEE_RATE")
    old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
    old_execution_interval = os.environ.get("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL")
    old_entry_policy = os.environ.get("TRADERBOT_ENTRY_POLICY")
    old_retest_pullback = os.environ.get("TRADERBOT_RETEST_PULLBACK")
    old_retest_ttl = os.environ.get("TRADERBOT_RETEST_TTL_MIN")
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
    if entry_policy is not None:
        os.environ["TRADERBOT_ENTRY_POLICY"] = entry_policy
    else:
        os.environ.pop("TRADERBOT_ENTRY_POLICY", None)
    if retest_pullback is not None:
        os.environ["TRADERBOT_RETEST_PULLBACK"] = str(float(retest_pullback))
    else:
        os.environ.pop("TRADERBOT_RETEST_PULLBACK", None)
    if retest_ttl_min is not None:
        os.environ["TRADERBOT_RETEST_TTL_MIN"] = str(int(retest_ttl_min))
    else:
        os.environ.pop("TRADERBOT_RETEST_TTL_MIN", None)
    try:
        yield
    finally:
        _restore_env("TRADERBOT_EXCHANGE_BACKEND", old_backend)
        _restore_env("TRADERBOT_EXCHANGE_STATE_PATH", old_state)
        _restore_env("TRADERBOT_EXCHANGE_EVENTS_PATH", old_events)
        _restore_env("TRADERBOT_EXCHANGE_FEE_RATE", old_fee_rate)
        _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)
        _restore_env("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", old_execution_interval)
        _restore_env("TRADERBOT_ENTRY_POLICY", old_entry_policy)
        _restore_env("TRADERBOT_RETEST_PULLBACK", old_retest_pullback)
        _restore_env("TRADERBOT_RETEST_TTL_MIN", old_retest_ttl)


def _build_deterministic_scan_context(
    config: ExchangeReplayConfig,
    cache: LocalMarketCache,
    wallet: dict[str, Any],
    as_of_ms: int,
    screener_cfg: ScreenerConfig,
    screener_state: ScreenerStateStore,
) -> dict[str, Any]:
    state = state_from_wallet_and_events(
        wallet,
        _read_jsonl_since(_required_path(config.events_path), 0),
        as_of_ms,
        last_candidate_ts=screener_state.last_candidate_ts,
        last_candidate_quality=screener_state.last_candidate_quality,
    )
    result = run_screener(
        symbols=list(config.symbols),
        as_of_ms=as_of_ms,
        cfg=screener_cfg,
        state=state,
        cache_path=cache.path,
    )
    artifacts = write_scan_artifacts(result, config.run_id)
    rows = [row.model_dump(mode="json") for row in result.symbols]
    screener_state.update_from_scan_rows(rows, as_of_ms)
    candidate_primitives = [
        {
            "symbol": row.symbol,
            "side": row.candidate,
            "quality": row.candidate_quality or "hard",
            "failed_gates": row.failed_gates,
            "marginal_reasons": row.marginal_reasons,
            "plan": None if row.plan is None else row.plan.model_dump(mode="json"),
            "score": row.candidate_score,
            "signal_candidate_before_state": row.signal_candidate_before_state,
        }
        for row in result.symbols
        if row.candidate in {"long", "short"}
    ]
    # Runner-internal primitives keep the reference plan for logs/validation;
    # the agent-facing view carries scanner facts only.
    candidate_view = [
        {
            "symbol": item["symbol"],
            "side": item["side"],
            "quality": item["quality"],
            "failed_gates": item["failed_gates"],
            "marginal_reasons": item["marginal_reasons"],
        }
        for item in candidate_primitives
    ]
    return {
        "screener_mode": "deterministic",
        "scan_markdown": to_markdown_table(result),
        "scan_artifact_path": artifacts["artifact_path"],
        "scan_hash": artifacts["sha256"],
        "candidate_primitives": candidate_primitives,
        "candidate_view": candidate_view,
        "screener_candidates": [item["symbol"] for item in candidate_primitives],
        "max_entry_drift_pct": screener_cfg.max_entry_drift_pct,
        "entry_policy": config.entry_policy,
        "retest_pullback": config.retest_pullback,
        "retest_ttl_min": config.retest_ttl_min,
        "global_blocks": result.global_blocks,
        "data_warnings": result.data_warnings,
    }


def _apply_runner_maintenance(
    exchange: SimulatedExchange,
    cache: LocalMarketCache,
    wallet: dict[str, Any],
    as_of_ms: int,
    fee_rate: float,
    screener_cfg: ScreenerConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    actions = []
    warnings = []
    for position in wallet.get("open_positions") or []:
        side = "long" if position.get("side") in {"Buy", "long"} else "short"
        reason = None
        if check_max_hold(position, as_of_ms, screener_cfg):
            reason = "max_hold"
        else:
            try:
                frame = load_closed(cache.path, position.get("symbol"), INTERVAL_MS["1h"], as_of_ms, screener_cfg.min_1h_bars)
                issue = validate_frame(frame, as_of_ms, screener_cfg.min_1h_bars)
                if issue is not None:
                    warnings.append(f"{position.get('symbol')}:impulse_break_unchecked:{issue.reason}")
                elif check_impulse_break(frame, side, screener_cfg):
                    reason = "impulse_break"
            except Exception as error:
                warnings.append(f"{position.get('symbol')}:impulse_break_unchecked:{error}")
                reason = None
        if reason is None:
            continue
        action = MaintenanceAction(
            action="close_position",
            reason=reason,
            symbol=position.get("symbol"),
            side=side,
            position_id=position.get("position_id"),
        )
        closed = exchange.close_position(
            position_id=position.get("position_id"),
            symbol=position.get("symbol"),
            as_of=as_of_ms,
            mark_interval="1m",
            fee_rate=fee_rate,
        )
        actions.append({**action.model_dump(mode="json"), "closed_position": closed.get("closed_position")})
    return actions, warnings


def _auto_hold_decision(context: dict[str, Any]) -> dict[str, Any]:
    reason = "auto hold: deterministic screener produced no candidates"
    if context.get("global_blocks"):
        reason += f"; global blocks: {', '.join(context['global_blocks'])}"
    if context.get("data_warnings"):
        reason += f"; data warnings: {', '.join(context['data_warnings'][:5])}"
    if context.get("maintenance_warnings"):
        reason += f"; maintenance warnings: {', '.join(context['maintenance_warnings'][:5])}"
    return {
        "final_decision": "hold",
        "symbol": context["symbols"][0],
        "timeframe": context["decision_interval"],
        "thesis": "No deterministic screener candidate.",
        "price": None,
        "stop_loss": None,
        "take_profit": None,
        "amount": 0.0,
        "confidence": 0.0,
        "risk_summary": reason,
        "tool_summary": ["deterministic_screener:auto_hold"],
        "scan_hash": context.get("scan_hash"),
        "scan_artifact_path": context.get("scan_artifact_path"),
    }


def _validate_deterministic_replay_timing(start_ms: int, end_ms: int, decision_interval: str) -> None:
    step_ms = INTERVAL_MS[decision_interval]
    if int(start_ms) % step_ms != 0 or int(end_ms) % step_ms != 0:
        raise ValueError("deterministic screener replay requires start_time and end_time aligned to closed 1h boundaries")


def _compact_scan_context(context: dict[str, Any]) -> dict[str, Any] | None:
    if context.get("screener_mode") != "deterministic":
        return None
    return {
        "screener_mode": context.get("screener_mode"),
        "scan_artifact_path": context.get("scan_artifact_path"),
        "scan_hash": context.get("scan_hash"),
        "candidates": context.get("screener_candidates", []),
        "candidate_primitives": context.get("candidate_primitives", []),
        "global_blocks": context.get("global_blocks", []),
        "data_warnings": context.get("data_warnings", []),
    }


def _allowed_entry_candidates(context: dict[str, Any]) -> set[tuple[str, str]]:
    allowed = set()
    for item in context.get("candidate_primitives") or []:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        side = item.get("side")
        if side not in {"long", "short"}:
            continue
        try:
            allowed.add((normalize_symbol(str(symbol)), side))
        except Exception:
            continue
    return allowed


@contextmanager
def _deterministic_candidate_allowlist_env(context: dict[str, Any]) -> Iterator[None]:
    old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
    old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
    max_drift = context.get("max_entry_drift_pct")
    payload = []
    for item in context.get("candidate_primitives") or []:
        if not (isinstance(item, dict) and item.get("side") in {"long", "short"}):
            continue
        entry = {"symbol": item.get("symbol"), "side": item.get("side")}
        plan = item.get("plan") if isinstance(item.get("plan"), dict) else {}
        if plan.get("ref_entry") is not None and max_drift is not None:
            entry["ref_price"] = plan.get("ref_entry")
            entry["max_drift_pct"] = max_drift
        payload.append(entry)
    os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
    os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    try:
        yield
    finally:
        _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
        _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _required_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("replay path is required")
    return Path(path)


def _validate_decision_exchange_consistency(
    decision: Any,
    agent_exchange_events: list[dict[str, Any]],
    *,
    allow_hold_position_closes: bool = True,
    allowed_entry_candidates: set[tuple[str, str]] | None = None,
    strict_entry_events: bool = False,
    expected_entry_policy: str | None = None,
) -> None:
    if not isinstance(decision, dict):
        return
    final_decision = str(decision.get("final_decision", "")).lower()
    event_types = [str(event.get("type", "")) for event in agent_exchange_events]
    place_order_events = [event for event in agent_exchange_events if str(event.get("type", "")) == "place_order"]
    if final_decision in {"long", "short"}:
        if allowed_entry_candidates is not None:
            try:
                candidate_key = (normalize_symbol(str(decision.get("symbol") or "")), final_decision)
            except Exception:
                candidate_key = ("", final_decision)
            if candidate_key not in allowed_entry_candidates:
                raise RuntimeError(f"{final_decision} decision is not in deterministic screener candidates")
        if len(place_order_events) != 1:
            raise RuntimeError(f"{final_decision} decision did not produce a place_order exchange event")
        if not _place_order_event_matches_decision(place_order_events[0], decision, final_decision, expected_entry_policy=expected_entry_policy):
            raise RuntimeError(f"{final_decision} decision does not match the place_order exchange event (expected entry policy: {expected_entry_policy or 'next_open'})")
        if strict_entry_events:
            disallowed = []
            for event in agent_exchange_events:
                event_type = str(event.get("type", ""))
                if event_type == "place_order":
                    continue
                if event_type == "set_leverage" and _set_leverage_event_matches_decision(event, decision):
                    continue
                disallowed.append(event_type)
            if disallowed:
                raise RuntimeError(f"{final_decision} decision produced disallowed exchange events: {', '.join(disallowed)}")
    if final_decision == "hold":
        allowed = {"position_closed"} if allow_hold_position_closes else set()
        disallowed = [event_type for event_type in event_types if event_type not in allowed]
        if disallowed:
            raise RuntimeError(f"hold decision produced disallowed exchange events: {', '.join(disallowed)}")


def _place_order_event_matches_decision(event: dict[str, Any], decision: dict[str, Any], final_decision: str, *, expected_entry_policy: str | None = None) -> bool:
    raw_payload = event.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    if str(payload.get("category", "")).lower() != "linear":
        return False
    if not _symbols_match(payload.get("symbol"), decision.get("symbol")):
        return False
    expected_side = "Buy" if final_decision == "long" else "Sell"
    if str(payload.get("side")) != expected_side:
        return False
    amount = _optional_float(decision.get("amount"))
    if amount is None or amount <= 0:
        return False
    payload_policy = str(payload.get("entry_policy") or "")
    if expected_entry_policy == "limit_retest" and payload_policy != "limit_retest":
        # configured retest entry must not silently fall back to a market fill
        return False
    if payload_policy == "limit_retest":
        # runner-owned retest entry: the accepted order is a pending limit, no position yet
        if str(payload.get("orderType", "")).lower() != "limit":
            return False
        if str(payload.get("status", "")).lower() != "new":
            return False
        if payload.get("expires_at_ms") is None:
            return False
        qty = _optional_float(payload.get("qty"))
        ref_price = _optional_float(payload.get("entry_ref_price"))
        notional = qty * ref_price if qty is not None and ref_price is not None else None
    else:
        if str(payload.get("orderType", "")).lower() != "market":
            return False
        if str(payload.get("status", "")).lower() != "filled":
            return False
        raw_position = payload.get("position")
        position: dict[str, Any] = raw_position if isinstance(raw_position, dict) else {}
        if not position:
            return False
        if str(position.get("category", "")).lower() != "linear":
            return False
        notional = _optional_float(payload.get("notional_usdt"))
        if notional is None:
            notional = _optional_float(position.get("notional_usdt"))
        if notional is None:
            qty = _optional_float(payload.get("qty"))
            price = _optional_float(payload.get("price"))
            if qty is not None and price is not None:
                notional = qty * price
    if notional is None or not math.isclose(notional, amount, rel_tol=0.02, abs_tol=1e-6):
        return False
    for decision_key, payload_key in (("stop_loss", "stopLoss"), ("take_profit", "takeProfit")):
        decision_value = _optional_float(decision.get(decision_key))
        payload_value = _optional_float(payload.get(payload_key))
        if decision_value is None or payload_value is None:
            return False
        if not math.isclose(payload_value, decision_value, rel_tol=1e-6, abs_tol=1e-8):
            return False
    return True


def _set_leverage_event_matches_decision(event: dict[str, Any], decision: dict[str, Any]) -> bool:
    raw_payload = event.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    if str(payload.get("category", "")).lower() != "linear":
        return False
    return _symbols_match(payload.get("symbol"), decision.get("symbol"))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _symbols_match(left: Any, right: Any) -> bool:
    try:
        return normalize_symbol(str(left or "")) == normalize_symbol(str(right or ""))
    except ValueError:
        return False


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
        "screener_mode": config.screener_mode,
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
