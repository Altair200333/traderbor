from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from traderbot_ai.decision.serialization import jsonable
from traderbot_ai.tools import replay_helpers
from traderbot_ai.tools.exchange import (
    _agent_tool_response,
    cancel_order_impl,
    close_position_impl,
    get_wallet_impl,
    place_order_impl,
    set_leverage_impl,
)
from traderbot_ai.tools.risk import _calculate_position_size_impl, _validate_order_impl
from traderbot_ai.tools.simulator import (
    get_cached_candles_impl,
    get_cached_price_impl,
    get_market_cache_status_impl,
    simulate_order_exit_impl,
)


def get_candles(
    symbol: str,
    interval: str = "1m",
    as_of: str | int | float | None = None,
    limit: int = 120,
    start_time: str | int | float | None = None,
    end_time: str | int | float | None = None,
) -> dict[str, Any]:
    """Read closed candles from the configured local cache."""
    if _deterministic_screener_mode():
        return _deterministic_reject("get_candles", "raw candles are disabled; use runner scan table and get_setup_digest for listed candidates")
    return get_cached_candles_impl(
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        lookback=limit,
        start_time=start_time,
        end_time=end_time,
    )


def get_current_price(symbol: str, interval: str = "1m", as_of: str | int | float | None = None) -> dict[str, Any]:
    """Read the latest cached close price at or before as_of."""
    return get_cached_price_impl(symbol=symbol, interval=interval, as_of=as_of)


def get_market_cache_status() -> dict[str, Any]:
    """Inspect configured local market cache coverage."""
    return get_market_cache_status_impl()


def get_cache_status_compact() -> dict[str, Any]:
    """Inspect local market cache coverage without verbose fields."""
    status = get_market_cache_status_impl()
    return {
        "ok": status.get("ok"),
        "cache_path": status.get("cache_path"),
        "markets": status.get("markets", []),
        "error": status.get("error"),
    }


def scan_momentum_universe(symbols: list[str] | str, as_of: str | int | float, decision_interval: str = "4h") -> dict[str, Any]:
    """Compact deterministic coarse scan for replay momentum candidates."""
    if _deterministic_screener_mode():
        return _deterministic_reject("scan_momentum_universe", "broad scan already ran in the replay runner")
    return replay_helpers.scan_momentum_universe(symbols=symbols, as_of=as_of, decision_interval=decision_interval)


def get_candidate_detail(symbol: str, side: Literal["long", "short"], as_of: str | int | float) -> dict[str, Any]:
    """Compact deterministic 1h detail for one shortlisted momentum candidate."""
    return replay_helpers.get_candidate_detail(symbol=symbol, side=side, as_of=as_of)


def get_setup_digest(symbol: str, side: Literal["long", "short"], as_of: str | int | float) -> dict[str, Any]:
    """Canonical deterministic setup digest for one shortlisted momentum candidate."""
    return replay_helpers.get_setup_digest(symbol=symbol, side=side, as_of=as_of)


def get_recent_trade_events(as_of: str | int | float, lookback_hours: int = 168, limit: int = 200) -> dict[str, Any]:
    """Read recent simulator exchange events for risk reconstruction."""
    return replay_helpers.get_recent_trade_events(as_of=as_of, lookback_hours=lookback_hours, limit=limit)


def simulate_order_exit(
    final_decision: str,
    symbol: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    opened_at: str | int | float,
    scan_until: str | int | float,
    interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict[str, Any]:
    """Resolve TP/SL order exit against cached candles using the existing execution engine."""
    return simulate_order_exit_impl(
        final_decision=final_decision,
        symbol=symbol,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        amount=amount,
        opened_at=opened_at,
        scan_until=scan_until,
        interval=interval,
        fee_rate=fee_rate,
    )


def validate_order(
    final_decision: Literal["long", "short", "hold"],
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    balance_usdt: float,
    max_position_fraction: float = 1.0,
    max_loss_fraction: float = 0.0075,
    min_stop_distance_pct: float = 0.01,
    max_stop_distance_pct: float = 0.04,
    min_reward_risk: float = 1.5,
    round_trip_fee_fraction: float = 0.0,
    expected_funding_fraction: float = 0.0,
    expected_slippage_fraction: float = 0.0,
) -> dict[str, Any]:
    """Validate order geometry and risk with the existing risk implementation."""
    return _validate_order_impl(
        final_decision=final_decision,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        amount=amount,
        balance_usdt=balance_usdt,
        max_position_fraction=max_position_fraction,
        max_loss_fraction=max_loss_fraction,
        min_stop_distance_pct=min_stop_distance_pct,
        max_stop_distance_pct=max_stop_distance_pct,
        min_reward_risk=min_reward_risk,
        round_trip_fee_fraction=round_trip_fee_fraction,
        expected_funding_fraction=expected_funding_fraction,
        expected_slippage_fraction=expected_slippage_fraction,
    )


def calculate_position_size(
    balance_usdt: float,
    price: float,
    stop_loss: float,
    risk_fraction: float = 0.0075,
    daily_risk_budget_remaining: float | None = None,
    size_pct_of_daily_budget: float = 35.0,
    max_notional_abs: float | None = None,
    liquidity_cap: float | None = None,
    max_position_fraction: float = 1.0,
) -> dict[str, Any]:
    """Calculate notional size with the existing risk implementation."""
    return _calculate_position_size_impl(
        balance_usdt=balance_usdt,
        price=price,
        stop_loss=stop_loss,
        risk_fraction=risk_fraction,
        daily_risk_budget_remaining=daily_risk_budget_remaining,
        size_pct_of_daily_budget=size_pct_of_daily_budget,
        max_notional_abs=max_notional_abs,
        liquidity_cap=liquidity_cap,
        max_position_fraction=max_position_fraction,
    )


def get_wallet(symbols: str = "", as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
    """Get balances, orders, positions, and USDT conversion from the configured exchange backend."""
    return _agent_tool_response(get_wallet_impl(symbols=symbols, as_of=as_of, mark_interval=mark_interval))


def get_wallet_compact(symbols: str = "", as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
    """Get compact wallet state for token-efficient replay context checks."""
    return replay_helpers.get_wallet_compact(symbols=symbols, as_of=as_of, mark_interval=mark_interval)


def get_open_positions(symbols: str = "", as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
    """Get open positions only."""
    return replay_helpers.get_open_positions(symbols=symbols, as_of=as_of, mark_interval=mark_interval)


def set_leverage(category: str, symbol: str, buyLeverage: str, sellLeverage: str) -> dict[str, Any]:
    """Set simulator leverage through the existing exchange implementation."""
    inputs = {
        "category": category,
        "symbol": symbol,
        "buyLeverage": buyLeverage,
        "sellLeverage": sellLeverage,
    }
    result = _agent_tool_response(set_leverage_impl(**inputs))
    _audit_write("set_leverage", inputs, result)
    return result


def place_order(
    category: str,
    symbol: str,
    side: str,
    orderType: str,
    qty: float,
    price: float | None = None,
    timeInForce: str = "GTC",
    takeProfit: float | None = None,
    stopLoss: float | None = None,
    orderLinkId: str | None = None,
    leverage: float | None = None,
    reduceOnly: bool = False,
    positionIdx: int = 0,
    tpslMode: str = "Full",
    marketUnit: str | None = None,
    fee_rate: float = 0.0,
    as_of: str | int | float | None = None,
    mark_interval: str = "1m",
) -> dict[str, Any]:
    """Place a simulator exchange order. For linear orders, qty is base-asset quantity, not USDT notional; leave marketUnit null."""
    inputs = {
        "category": category,
        "symbol": symbol,
        "side": side,
        "orderType": orderType,
        "qty": qty,
        "price": price,
        "timeInForce": timeInForce,
        "takeProfit": takeProfit,
        "stopLoss": stopLoss,
        "orderLinkId": orderLinkId,
        "leverage": leverage,
        "reduceOnly": reduceOnly,
        "positionIdx": positionIdx,
        "tpslMode": tpslMode,
        "marketUnit": marketUnit,
        "fee_rate": fee_rate,
        "as_of": as_of,
        "mark_interval": mark_interval,
    }
    preflight = _require_write_time("place_order", as_of) or _require_order_link_id(orderLinkId)
    if preflight is not None:
        _audit_write("place_order", inputs, preflight)
        return preflight
    preflight = _deterministic_place_order_error(inputs)
    if preflight is not None:
        _audit_write("place_order", inputs, preflight)
        return preflight
    risk_validation = _validate_place_order_risk(inputs)
    if risk_validation is not None and risk_validation.get("ok") is not True:
        result = {"ok": False, "error": "risk validation failed before place_order", "risk_validation": risk_validation}
        _audit_write("place_order", inputs, result)
        return result
    result = _agent_tool_response(place_order_impl(**inputs))
    if risk_validation is not None:
        result["risk_validation"] = risk_validation
    _audit_write("place_order", inputs, result)
    return result


def cancel_order(
    order_id: str | None = None,
    orderLinkId: str | None = None,
    category: str | None = None,
    symbol: str | None = None,
    as_of: str | int | float | None = None,
    interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict[str, Any]:
    """Cancel a simulator exchange order through the existing exchange implementation."""
    inputs = {
        "order_id": order_id,
        "orderLinkId": orderLinkId,
        "category": category,
        "symbol": symbol,
        "as_of": as_of,
        "interval": interval,
        "fee_rate": fee_rate,
    }
    if _deterministic_screener_mode():
        result = _deterministic_reject("cancel_order", "runner-owned deterministic mode does not allow provider cancels")
        _audit_write("cancel_order", inputs, result)
        return result
    preflight = _require_write_time("cancel_order", as_of)
    if preflight is not None:
        _audit_write("cancel_order", inputs, preflight)
        return preflight
    result = _agent_tool_response(cancel_order_impl(**inputs))
    _audit_write("cancel_order", inputs, result)
    return result


def close_position(
    position_id: str | None = None,
    category: str | None = None,
    symbol: str | None = None,
    side: str | None = None,
    price: float | None = None,
    as_of: str | int | float | None = None,
    mark_interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict[str, Any]:
    """Close a simulator exchange position through the existing exchange implementation."""
    inputs = {
        "position_id": position_id,
        "category": category,
        "symbol": symbol,
        "side": side,
        "price": price,
        "as_of": as_of,
        "mark_interval": mark_interval,
        "fee_rate": fee_rate,
    }
    if _deterministic_screener_mode():
        result = _deterministic_reject("close_position", "runner-owned deterministic mode does not allow provider maintenance closes")
        _audit_write("close_position", inputs, result)
        return result
    preflight = _require_write_time("close_position", as_of)
    if preflight is not None:
        _audit_write("close_position", inputs, preflight)
        return preflight
    result = _agent_tool_response(close_position_impl(**inputs))
    _audit_write("close_position", inputs, result)
    return result


def _require_write_time(tool_name: str, as_of: str | int | float | None) -> dict[str, Any] | None:
    if as_of is None or str(as_of).strip() == "":
        return {"ok": False, "error": f"{tool_name} requires as_of in exchange replay MCP mode"}
    return None


def _deterministic_screener_mode() -> bool:
    return os.getenv("TRADERBOT_SCREENER_MODE") == "deterministic"


def _deterministic_reject(tool_name: str, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool_name,
        "screener_mode": "deterministic",
        "error": reason,
    }


def _require_order_link_id(order_link_id: str | None) -> dict[str, Any] | None:
    if order_link_id is None or str(order_link_id).strip() == "":
        return {"ok": False, "error": "place_order requires idempotent orderLinkId in exchange replay MCP mode"}
    return None


def _deterministic_place_order_error(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if not _deterministic_screener_mode():
        return None
    if bool(inputs.get("reduceOnly")):
        return _deterministic_reject("place_order", "runner-owned deterministic mode does not allow reduce-only provider orders")
    if str(inputs.get("category") or "").lower() != "linear":
        return _deterministic_reject("place_order", "runner-owned deterministic mode only allows linear finalist entries")
    side = str(inputs.get("side") or "").lower()
    candidate_side = "long" if side in {"buy", "long"} else "short" if side in {"sell", "short"} else ""
    if not candidate_side:
        return _deterministic_reject("place_order", f"unsupported deterministic entry side: {inputs.get('side')}")
    allow_error = replay_helpers._deterministic_candidate_error(str(inputs.get("symbol") or ""), candidate_side)
    if allow_error is None:
        return None
    result = dict(allow_error)
    result["tool"] = "place_order"
    result["error"] = "place_order is only available for runner-provided deterministic candidates"
    return result


def _validate_place_order_risk(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if str(inputs.get("category")).lower() != "linear":
        return None
    side = str(inputs.get("side") or "")
    final_decision = "long" if side.lower() in {"buy", "long"} else "short" if side.lower() in {"sell", "short"} else None
    if final_decision is None:
        return {"ok": False, "errors": [f"unsupported side for risk validation: {side}"], "warnings": []}
    take_profit = inputs.get("takeProfit")
    stop_loss = inputs.get("stopLoss")
    if take_profit is None or stop_loss is None:
        return {"ok": False, "errors": ["linear replay place_order requires takeProfit and stopLoss"], "warnings": []}
    entry_price = _entry_price_for_order(inputs)
    if entry_price is None or entry_price <= 0:
        return {"ok": False, "errors": ["could not resolve entry price for risk validation"], "warnings": []}
    wallet = get_wallet_impl(symbols=str(inputs.get("symbol") or ""), as_of=inputs.get("as_of"), mark_interval=str(inputs.get("mark_interval") or "1m"))
    if wallet.get("ok") is not True:
        return {"ok": False, "errors": [f"could not read wallet for risk validation: {wallet.get('error')}"], "warnings": []}
    totals = wallet.get("totals") or {}
    balance_usdt = float(totals.get("equity_usdt") or 0.0)
    amount = float(inputs.get("qty") or 0.0) * entry_price
    fee_rate = _replay_fee_rate(float(inputs.get("fee_rate") or 0.0))
    return _validate_order_impl(
        final_decision=final_decision,
        price=entry_price,
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        amount=amount,
        balance_usdt=balance_usdt,
        max_position_fraction=0.20,
        max_loss_fraction=0.0075,
        round_trip_fee_fraction=fee_rate,
    )


def _entry_price_for_order(inputs: dict[str, Any]) -> float | None:
    price = inputs.get("price")
    if price is not None:
        return float(price)
    if str(inputs.get("orderType") or "").lower() != "market":
        return None
    result = get_cached_price_impl(
        symbol=str(inputs.get("symbol") or ""),
        interval=str(inputs.get("mark_interval") or "1m"),
        as_of=inputs.get("as_of"),
    )
    if result.get("ok") is not True:
        return None
    return float(result["price"])


def _replay_fee_rate(fee_rate: float) -> float:
    override = os.getenv("TRADERBOT_EXCHANGE_FEE_RATE")
    if override is not None and override.strip() != "":
        return float(override)
    return fee_rate


def _event_replay_time_ms(record: dict[str, Any]) -> int | None:
    return replay_helpers._event_replay_time_ms(record)


def _audit_write(tool_name: str, inputs: dict[str, Any], result: dict[str, Any]) -> None:
    audit_path = os.getenv("TRADERBOT_MCP_AUDIT_PATH")
    if not audit_path:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": os.getenv("TRADERBOT_MCP_RUN_ID"),
        "step_id": os.getenv("TRADERBOT_MCP_STEP_ID"),
        "tool": tool_name,
        "input": jsonable(inputs),
        "result": jsonable(result),
    }
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
