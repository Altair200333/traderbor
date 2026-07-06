from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from traderbot_ai.mcp.momentum import get_candidate_detail_impl, scan_momentum_universe_impl
from traderbot_ai.tools.exchange import _agent_tool_response, get_wallet_impl
from traderbot_ai.tools.market import parse_time_ms


def scan_momentum_universe(symbols: list[str] | str, as_of: str | int | float, decision_interval: str = "4h") -> dict[str, Any]:
    """Compact deterministic coarse scan for replay momentum candidates."""
    return scan_momentum_universe_impl(symbols=symbols, as_of=as_of, decision_interval=decision_interval)


def get_candidate_detail(symbol: str, side: Literal["long", "short"], as_of: str | int | float) -> dict[str, Any]:
    """Compact deterministic 1h detail for one shortlisted momentum candidate."""
    return get_candidate_detail_impl(symbol=symbol, side=side, as_of=as_of)


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
    as_of_ms = parse_time_ms(as_of)
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
