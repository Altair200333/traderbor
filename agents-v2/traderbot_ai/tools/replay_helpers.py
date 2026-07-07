from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from traderbot_ai.screener.artifacts import write_scan_artifacts
from traderbot_ai.screener.market import normalize_symbol
from traderbot_ai.screener.render import to_markdown_table
from traderbot_ai.screener.screener import agent_row_view
from traderbot_ai.screener.screener import get_setup_digest as get_setup_digest_impl
from traderbot_ai.screener.screener import scan as scan_screener
from traderbot_ai.screener.state import state_from_wallet_and_events
from traderbot_ai.simulator.clock import active_simulation_clock_ms, guarded_simulation_as_of
from traderbot_ai.simulator.market_cache import DEFAULT_CACHE_PATH
from traderbot_ai.tools.exchange import _agent_tool_response, get_wallet_impl
from traderbot_ai.tools.market import parse_time_ms


DETERMINISTIC_DEEP_DIVE_ANCHORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DETERMINISTIC_DEEP_DIVE_LIMITS = {"1h": 170, "4h": 60}


def scan_momentum_universe(symbols: list[str] | str, as_of: str | int | float, decision_interval: str = "4h") -> dict[str, Any]:
    """Compact deterministic coarse scan for replay momentum candidates."""
    try:
        as_of_ms = guarded_simulation_as_of(as_of)
        if as_of_ms is None:
            return {"ok": False, "error": "as_of is required for scan_momentum_universe"}
        wallet = _agent_tool_response(get_wallet_impl(symbols=_symbols_csv(symbols), as_of=as_of_ms, mark_interval="1m"))
        state = state_from_wallet_and_events(wallet, _read_exchange_events(), as_of_ms)
        result = scan_screener(
            symbols=symbols,
            as_of_ms=as_of_ms,
            state=state,
            cache_path=os.getenv("TRADERBOT_MARKET_CACHE_PATH") or DEFAULT_CACHE_PATH,
        )
        artifact = _maybe_write_scan_artifact(result)
        rows = [agent_row_view(row) for row in result.symbols]
        fresh_symbol_count = sum(1 for row in result.symbols if row.status == "ok")
        return {
            "ok": fresh_symbol_count > 0,
            "error": None if fresh_symbol_count > 0 else "no fresh 1h candles at as_of",
            "as_of_ms": result.as_of_ms,
            "decision_interval": decision_interval,
            "source": "local_cache",
            "fresh_symbol_count": fresh_symbol_count,
            "candidates": [row for row in rows if row.get("candidate") in {"long", "short"}],
            "rejected": [row for row in rows if row.get("candidate") not in {"long", "short"}],
            "global_blocks": result.global_blocks,
            "data_warnings": result.data_warnings,
            "btc_roc_4h": result.btc_roc_4h,
            "markdown": to_markdown_table(result),
            "scan_hash": artifact.get("sha256"),
            "scan_artifact_path": artifact.get("artifact_path"),
        }
    except Exception as error:
        return {"ok": False, "error": str(error)}


def get_candidate_detail(symbol: str, side: Literal["long", "short"], as_of: str | int | float) -> dict[str, Any]:
    """Compact deterministic 1h detail for one shortlisted momentum candidate."""
    return get_setup_digest(symbol=symbol, side=side, as_of=as_of)


def get_setup_digest(symbol: str, side: Literal["long", "short"], as_of: str | int | float) -> dict[str, Any]:
    """Compact deterministic setup digest for one shortlisted momentum candidate."""
    try:
        allow_error = _deterministic_candidate_error(symbol, side)
        if allow_error is not None:
            return allow_error
        exact = _deterministic_exact_as_of(as_of, "get_setup_digest")
        if exact.get("ok") is not True:
            return exact
        as_of_ms = int(exact["as_of_ms"])
        if as_of_ms is None:
            return {"ok": False, "error": "as_of is required for get_setup_digest"}
        wallet = _agent_tool_response(get_wallet_impl(symbols=symbol, as_of=as_of_ms, mark_interval="1m"))
        state = state_from_wallet_and_events(wallet, _read_exchange_events(), as_of_ms)
        return get_setup_digest_impl(
            symbol=symbol,
            side=side,
            as_of_ms=as_of_ms,
            state=state,
            cache_path=os.getenv("TRADERBOT_MARKET_CACHE_PATH") or DEFAULT_CACHE_PATH,
        )
    except Exception as error:
        return {"ok": False, "symbol": symbol, "side": side, "error": str(error)}


def get_wallet_compact(symbols: str = "", as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
    """Get compact wallet state for token-efficient replay context checks."""
    wallet = _agent_tool_response(get_wallet_impl(symbols=symbols, as_of=as_of, mark_interval=mark_interval))
    totals = wallet.get("totals") or {}
    return {
        "ok": wallet.get("ok", True),
        "as_of_ms": wallet.get("as_of_ms"),
        "equity_usdt": totals.get("equity_usdt"),
        "free_usdt": totals.get("free_usdt"),
        "locked_usdt": totals.get("locked_usdt"),
        "valuation_complete": totals.get("valuation_complete"),
        "open_order_count": len(wallet.get("open_orders") or []),
        "open_position_count": len(wallet.get("open_positions") or []),
        "positions": wallet.get("open_positions") or [],
        "runtime_overrides": wallet.get("runtime_overrides"),
        "error": wallet.get("error"),
    }


def get_open_positions(symbols: str = "", as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
    """Get open positions only."""
    wallet = _agent_tool_response(get_wallet_impl(symbols=symbols, as_of=as_of, mark_interval=mark_interval))
    return {
        "ok": wallet.get("ok", True),
        "as_of_ms": wallet.get("as_of_ms"),
        "positions": wallet.get("open_positions") or [],
        "error": wallet.get("error"),
    }


def get_recent_trade_events(as_of: str | int | float, lookback_hours: int = 168, limit: int = 200) -> dict[str, Any]:
    """Read recent simulator exchange events for risk reconstruction."""
    try:
        as_of_ms = guarded_simulation_as_of(as_of)
    except Exception as error:
        return {"ok": False, "error": str(error)}
    if as_of_ms is None:
        return {"ok": False, "error": "as_of is required for get_recent_trade_events"}
    path_value = os.getenv("TRADERBOT_EXCHANGE_EVENTS_PATH")
    if not path_value:
        return {"ok": False, "error": "TRADERBOT_EXCHANGE_EVENTS_PATH is not configured"}
    start_ms = as_of_ms - max(1, int(lookback_hours)) * 60 * 60_000
    events = []
    path = Path(path_value)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_ms_int = _event_replay_time_ms(record)
            if event_ms_int is not None and start_ms <= event_ms_int <= as_of_ms:
                events.append(record)
    return {"ok": True, "as_of_ms": as_of_ms, "lookback_hours": lookback_hours, "events": events[-max(1, int(limit)) :]}


def _event_replay_time_ms(record: dict[str, Any]) -> int | None:
    for container in (record, record.get("payload") if isinstance(record.get("payload"), dict) else None):
        if not isinstance(container, dict):
            continue
        for key in ("as_of_ms", "timestamp_ms", "ts_ms", "exit_time_ms", "settled_until_ms", "created_at_ms", "opened_at_ms"):
            value = container.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
    return None


def _read_exchange_events() -> list[dict[str, Any]]:
    path_value = os.getenv("TRADERBOT_EXCHANGE_EVENTS_PATH")
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _deterministic_candidate_error(symbol: str, side: str) -> dict[str, Any] | None:
    if os.getenv("TRADERBOT_SCREENER_MODE") != "deterministic":
        return None
    raw = _deterministic_candidates_json()
    if not raw:
        return {
            "ok": False,
            "symbol": symbol,
            "side": side,
            "screener_mode": "deterministic",
            "error": "deterministic candidate allowlist is not configured",
        }
    try:
        requested = (normalize_symbol(symbol), side)
        allowed = {
            (normalize_symbol(str(item.get("symbol") or "")), str(item.get("side") or ""))
            for item in json.loads(raw)
            if isinstance(item, dict)
        }
    except Exception as error:
        return {
            "ok": False,
            "symbol": symbol,
            "side": side,
            "screener_mode": "deterministic",
            "error": f"invalid deterministic candidate allowlist: {error}",
        }
    if requested in allowed:
        return None
    return {
        "ok": False,
        "symbol": symbol,
        "side": side,
        "screener_mode": "deterministic",
        "error": "setup digest is only available for runner-provided deterministic candidates",
        "allowed_candidates": [{"symbol": item[0], "side": item[1]} for item in sorted(allowed)],
    }


def _deterministic_candidates_json() -> str | None:
    """Candidate payload: inline env var for small sets, file indirection for hot bars
    (Windows command lines cap out near 8k chars; codex passes env via -c arguments)."""
    raw = os.getenv("TRADERBOT_DETERMINISTIC_CANDIDATES")
    if raw:
        return raw
    path = os.getenv("TRADERBOT_DETERMINISTIC_CANDIDATES_PATH")
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return None


def deterministic_entry_drift_error(symbol: str, side: str, entry_price: float | None) -> dict[str, Any] | None:
    """Reject entries whose price drifted adversely from the scan reference (anti-chase clamp)."""
    if os.getenv("TRADERBOT_SCREENER_MODE") != "deterministic":
        return None
    if entry_price is None or float(entry_price) <= 0:
        return None
    raw = _deterministic_candidates_json()
    if not raw:
        return None
    try:
        items = json.loads(raw)
    except Exception:
        return None
    if not isinstance(items, list):
        return None
    try:
        normalized = normalize_symbol(symbol)
    except Exception:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            item_symbol = normalize_symbol(str(item.get("symbol") or ""))
        except Exception:
            continue
        if item_symbol != normalized or str(item.get("side") or "") != side:
            continue
        ref_price = item.get("ref_price")
        max_drift = item.get("max_drift_pct")
        if ref_price is None or max_drift is None:
            return None
        ref = float(ref_price)
        if ref <= 0:
            return None
        drift = (float(entry_price) - ref) / ref
        adverse = drift > float(max_drift) if side == "long" else drift < -float(max_drift)
        if not adverse:
            return None
        return {
            "ok": False,
            "symbol": normalized,
            "side": side,
            "screener_mode": "deterministic",
            "error": f"entry price drifted {drift:+.4%} from scan reference {ref:.8g}; max adverse drift is {float(max_drift):.2%}",
            "ref_price": ref,
            "entry_price": float(entry_price),
            "entry_drift_pct": drift,
        }
    return None


def deterministic_stop_noise_error(symbol: str, side: str, entry_price: float | None, stop_loss: float | None) -> dict[str, Any] | None:
    """Reject entries whose stop sits inside the scan's measured noise floor (guaranteed noise stop-out)."""
    if os.getenv("TRADERBOT_SCREENER_MODE") != "deterministic":
        return None
    if entry_price is None or float(entry_price) <= 0:
        return None
    if stop_loss is None or float(stop_loss) <= 0:
        return None
    raw = _deterministic_candidates_json()
    if not raw:
        return None
    try:
        items = json.loads(raw)
    except Exception:
        return None
    if not isinstance(items, list):
        return None
    try:
        normalized = normalize_symbol(symbol)
    except Exception:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            item_symbol = normalize_symbol(str(item.get("symbol") or ""))
        except Exception:
            continue
        if item_symbol != normalized or str(item.get("side") or "") != side:
            continue
        floor = item.get("noise_floor_pct")
        if floor is None or float(floor) <= 0:
            return None
        entry = float(entry_price)
        distance = (entry - float(stop_loss)) / entry if side == "long" else (float(stop_loss) - entry) / entry
        if distance <= 0:
            # inverted geometry is rejected by risk validation, not here
            return None
        if distance >= float(floor):
            return None
        return {
            "ok": False,
            "symbol": normalized,
            "side": side,
            "screener_mode": "deterministic",
            "error": f"stop distance {distance:.4%} is inside the measured noise floor {float(floor):.4%}; widen the stop beyond noise or hold",
            "stop_distance_pct": distance,
            "noise_floor_pct": float(floor),
        }
    return None


def deterministic_candle_deep_dive_request(
    symbol: str,
    interval: str,
    as_of: str | int | float | None,
    limit: int,
    start_time: str | int | float | None,
    end_time: str | int | float | None,
) -> dict[str, Any] | None:
    if os.getenv("TRADERBOT_SCREENER_MODE") != "deterministic":
        return None
    normalized_interval = str(interval or "").strip().lower()
    allowed_limit = DETERMINISTIC_DEEP_DIVE_LIMITS.get(normalized_interval)
    if as_of is None or str(as_of).strip() == "":
        return _deterministic_candle_error(symbol, normalized_interval, "deterministic get_candles requires exact as_of")
    exact = _deterministic_exact_as_of(as_of, "deterministic get_candles")
    if exact.get("ok") is not True:
        return _deterministic_candle_error(symbol, normalized_interval, str(exact["error"]))
    if start_time is not None or end_time is not None:
        return _deterministic_candle_error(symbol, normalized_interval, "deterministic get_candles does not allow start_time or end_time")
    if allowed_limit is None:
        return _deterministic_candle_error(symbol, normalized_interval, "deterministic get_candles only allows 1h or 4h intervals")
    try:
        requested_limit = int(limit)
    except Exception:
        return _deterministic_candle_error(symbol, normalized_interval, "deterministic get_candles limit must be an integer")
    if requested_limit < 1:
        return _deterministic_candle_error(symbol, normalized_interval, "deterministic get_candles limit must be positive")
    if requested_limit > allowed_limit:
        return _deterministic_candle_error(
            symbol,
            normalized_interval,
            f"deterministic get_candles limit for {normalized_interval} must be <= {allowed_limit}",
        )
    try:
        normalized_symbol = normalize_symbol(symbol)
    except Exception as error:
        return _deterministic_candle_error(symbol, normalized_interval, f"invalid symbol: {error}")
    allowed_symbols = _deterministic_deep_dive_symbols()
    if normalized_symbol not in allowed_symbols:
        return {
            "ok": False,
            "symbol": normalized_symbol,
            "interval": normalized_interval,
            "screener_mode": "deterministic",
            "error": "deterministic get_candles is only available for runner candidates and BTC/ETH/SOL context anchors",
            "allowed_symbols": sorted(allowed_symbols),
        }
    return {
        "ok": True,
        "symbol": normalized_symbol,
        "interval": normalized_interval,
        "limit": requested_limit,
        "allowed_symbols": sorted(allowed_symbols),
    }


def _deterministic_exact_as_of(as_of: str | int | float | None, tool_name: str) -> dict[str, Any]:
    as_of_ms = parse_time_ms(as_of)
    if as_of_ms is None:
        return {"ok": False, "error": f"{tool_name} requires exact as_of"}
    active = active_simulation_clock_ms()
    if active is None:
        return {"ok": True, "as_of_ms": as_of_ms}
    if as_of_ms > active:
        return {"ok": False, "error": f"as_of {as_of_ms} exceeds simulation clock {active}"}
    if as_of_ms != active:
        return {"ok": False, "error": f"{tool_name} requires exact as_of equal to simulation clock {active}; got {as_of_ms}"}
    return {"ok": True, "as_of_ms": as_of_ms}


def _deterministic_deep_dive_symbols() -> set[str]:
    allowed = {normalize_symbol(symbol) for symbol in DETERMINISTIC_DEEP_DIVE_ANCHORS}
    raw = _deterministic_candidates_json()
    if not raw:
        return allowed
    try:
        items = json.loads(raw)
    except Exception:
        return allowed
    if not isinstance(items, list):
        return allowed
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        if symbol is None:
            continue
        try:
            allowed.add(normalize_symbol(str(symbol)))
        except Exception:
            continue
    return allowed


def _deterministic_candle_error(symbol: str, interval: str, message: str) -> dict[str, Any]:
    try:
        normalized_symbol = normalize_symbol(symbol)
    except Exception:
        normalized_symbol = str(symbol)
    return {
        "ok": False,
        "symbol": normalized_symbol,
        "interval": interval,
        "screener_mode": "deterministic",
        "error": message,
    }


def _symbols_csv(symbols: list[str] | str) -> str:
    if isinstance(symbols, str):
        return symbols
    return ",".join(str(symbol) for symbol in symbols)


def _maybe_write_scan_artifact(result: Any) -> dict[str, str]:
    run_id = os.getenv("TRADERBOT_RUN_ID") or os.getenv("TRADERBOT_MCP_RUN_ID")
    if not run_id:
        return {}
    try:
        return write_scan_artifacts(result, run_id)
    except Exception:
        return {}
