from __future__ import annotations

import json
import math
import os
from typing import Any

from agents import function_tool

from traderbot_ai.exchange import CancelOrderError, ExchangeBackend, SimulatedExchangeBackend, create_exchange_backend
from traderbot_ai.simulator.clock import guarded_simulation_as_of
from traderbot_ai.tools.market import INTERVAL_MS


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _error(message: str, **data) -> dict:
    return {"ok": False, "error": message, **data}


def _agent_tool_response(result: dict[str, Any]) -> dict[str, Any]:
    response = dict(result)
    state = response.pop("state", None)
    if state is None and _looks_like_state(response):
        state = {
            key: response.pop(key)
            for key in ("as_of_ms", "balances", "orders", "positions", "closed_positions")
            if key in response
        }
    if isinstance(state, dict):
        response["state_summary"] = _state_summary(state)
    overrides = _runtime_overrides()
    if overrides:
        response["runtime_overrides"] = overrides
    return response


def _looks_like_state(value: dict[str, Any]) -> bool:
    return "balances" in value and ("orders" in value or "positions" in value or "closed_positions" in value)


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "as_of_ms": state.get("as_of_ms"),
        "balances": state.get("balances", {}),
        "open_order_count": len(state.get("orders") or []),
        "open_position_count": len(state.get("positions") or []),
        "closed_position_count": len(state.get("closed_positions") or []),
    }


def _runtime_overrides() -> dict[str, Any]:
    result: dict[str, Any] = {}
    fee_rate = os.getenv("TRADERBOT_EXCHANGE_FEE_RATE")
    if fee_rate is not None and fee_rate.strip() != "":
        result["fee_rate"] = {"source": "TRADERBOT_EXCHANGE_FEE_RATE", "value": fee_rate}
    interval = os.getenv("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL")
    if interval is not None and interval.strip() != "":
        result["execution_interval"] = {"source": "TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", "value": interval}
    return result


def _exchange() -> ExchangeBackend:
    return create_exchange_backend()


def _simulated_exchange() -> SimulatedExchangeBackend:
    backend = create_exchange_backend()
    if getattr(backend, "backend_kind", None) != "simulated" or not isinstance(backend, SimulatedExchangeBackend):
        raise ValueError("exchange backend does not support simulator-only operation")
    return backend


def _effective_fee_rate(fee_rate: float) -> float:
    replay_fee_rate = os.getenv("TRADERBOT_EXCHANGE_FEE_RATE")
    if replay_fee_rate is not None and replay_fee_rate.strip() != "":
        value = float(replay_fee_rate)
        if not math.isfinite(value) or value < 0:
            raise ValueError("TRADERBOT_EXCHANGE_FEE_RATE must be non-negative")
        return value
    value = float(fee_rate)
    if not math.isfinite(value) or value < 0:
        raise ValueError("fee_rate must be non-negative")
    return value


def _effective_interval(interval: str) -> str:
    replay_interval = os.getenv("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL")
    if replay_interval is not None and replay_interval.strip() != "":
        if replay_interval not in INTERVAL_MS:
            raise ValueError("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL is unsupported")
        return replay_interval
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")
    return interval


def reset_exchange_impl(balances_json: str = '{"USDT": 1000}', as_of: str | int | float | None = None) -> dict[str, Any]:
    try:
        balances = json.loads(balances_json)
        if not isinstance(balances, dict):
            return _error("balances_json must be an object like {\"USDT\": 1000}")
        return _ok(**_simulated_exchange().reset({str(key): float(value) for key, value in balances.items()}, as_of=guarded_simulation_as_of(as_of)))
    except Exception as error:
        return _error(str(error))


def get_wallet_impl(symbols: str = "", as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
    try:
        wallet = _exchange().wallet_summary(symbols=symbols, as_of=guarded_simulation_as_of(as_of), mark_interval=_effective_interval(mark_interval))
        return wallet if wallet.get("ok") is True else _ok(**wallet)
    except Exception as error:
        return _error(str(error))


def set_leverage_impl(category: str, symbol: str, buyLeverage: str, sellLeverage: str) -> dict[str, Any]:
    try:
        return _ok(**_exchange().set_leverage(category=category, symbol=symbol, buyLeverage=buyLeverage, sellLeverage=sellLeverage))
    except Exception as error:
        return _error(str(error), category=category, symbol=symbol)


def place_order_impl(
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
    try:
        return _ok(
            **_exchange().place_order(
                category=category,
                symbol=symbol,
                side=side,
                orderType=orderType,
                qty=qty,
                price=price,
                timeInForce=timeInForce,
                takeProfit=takeProfit,
                stopLoss=stopLoss,
                orderLinkId=orderLinkId,
                leverage=leverage,
                reduceOnly=reduceOnly,
                positionIdx=positionIdx,
                tpslMode=tpslMode,
                marketUnit=marketUnit,
                fee_rate=_effective_fee_rate(fee_rate),
                as_of=guarded_simulation_as_of(as_of),
                mark_interval=_effective_interval(mark_interval),
            )
        )
    except Exception as error:
        return _error(str(error), category=category, symbol=symbol, side=side)


def cancel_order_impl(
    order_id: str | None = None,
    orderLinkId: str | None = None,
    category: str | None = None,
    symbol: str | None = None,
    as_of: str | int | float | None = None,
    interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict[str, Any]:
    try:
        return _ok(
            **_exchange().cancel_order(
                order_id=order_id,
                orderLinkId=orderLinkId,
                category=category,
                symbol=symbol,
                as_of=guarded_simulation_as_of(as_of),
                interval=_effective_interval(interval),
                fee_rate=_effective_fee_rate(fee_rate),
            )
        )
    except CancelOrderError as error:
        return _error(str(error), order_id=order_id, orderLinkId=orderLinkId, filled_orders_before_cancel=error.filled_orders, state=error.state)
    except Exception as error:
        return _error(str(error), order_id=order_id, orderLinkId=orderLinkId)


def settle_exchange_impl(as_of: str | int | float, interval: str = "1m", fee_rate: float = 0.0) -> dict[str, Any]:
    try:
        return _ok(**_simulated_exchange().settle(as_of=guarded_simulation_as_of(as_of), interval=_effective_interval(interval), fee_rate=_effective_fee_rate(fee_rate)))
    except Exception as error:
        return _error(str(error), as_of=as_of, interval=interval)


def close_position_impl(
    position_id: str | None = None,
    category: str | None = None,
    symbol: str | None = None,
    side: str | None = None,
    price: float | None = None,
    as_of: str | int | float | None = None,
    mark_interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict[str, Any]:
    try:
        return _ok(
            **_exchange().close_position(
                position_id=position_id,
                category=category,
                symbol=symbol,
                side=side,
                price=price,
                as_of=guarded_simulation_as_of(as_of),
                mark_interval=_effective_interval(mark_interval),
                fee_rate=_effective_fee_rate(fee_rate),
            )
        )
    except Exception as error:
        return _error(str(error), position_id=position_id)


@function_tool
def reset_exchange(balances_json: str = '{"USDT": 1000}', as_of: str | None = None) -> dict:
    """Reset the simulated exchange wallet. balances_json example: {"USDT": 1000, "BTC": 0.1}."""
    return _agent_tool_response(reset_exchange_impl(balances_json=balances_json, as_of=as_of))


@function_tool
def get_wallet(symbols: str = "", as_of: str | None = None, mark_interval: str = "1m") -> dict:
    """Get balances, USDT conversion, totals, open orders, and open positions."""
    return _agent_tool_response(get_wallet_impl(symbols=symbols, as_of=as_of, mark_interval=mark_interval))


@function_tool
def set_leverage(category: str, symbol: str, buyLeverage: str, sellLeverage: str) -> dict:
    """Set Bybit-like leverage for the simulator. One-way mode requires buyLeverage == sellLeverage."""
    return _agent_tool_response(
        set_leverage_impl(category=category, symbol=symbol, buyLeverage=buyLeverage, sellLeverage=sellLeverage)
    )


@function_tool
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
    as_of: str | None = None,
    mark_interval: str = "1m",
) -> dict:
    """Place a spot or linear simulator order with Bybit-like parameters. as_of is required; spot Market Buy defaults to quoteCoin qty."""
    return _agent_tool_response(
        place_order_impl(
            category=category,
            symbol=symbol,
            side=side,
            orderType=orderType,
            qty=qty,
            price=price,
            timeInForce=timeInForce,
            takeProfit=takeProfit,
            stopLoss=stopLoss,
            orderLinkId=orderLinkId,
            leverage=leverage,
            reduceOnly=reduceOnly,
            positionIdx=positionIdx,
            tpslMode=tpslMode,
            marketUnit=marketUnit,
            fee_rate=fee_rate,
            as_of=as_of,
            mark_interval=mark_interval,
        )
    )


@function_tool
def cancel_order(
    order_id: str | None = None,
    orderLinkId: str | None = None,
    category: str | None = None,
    symbol: str | None = None,
    as_of: str | None = None,
    interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict:
    """Cancel a simulated open order at as_of after first filling any eligible limit orders."""
    return _agent_tool_response(
        cancel_order_impl(
            order_id=order_id,
            orderLinkId=orderLinkId,
            category=category,
            symbol=symbol,
            as_of=as_of,
            interval=interval,
            fee_rate=fee_rate,
        )
    )


@function_tool
def settle_exchange(as_of: str, interval: str = "1m", fee_rate: float = 0.0) -> dict:
    """Fill eligible limit orders and settle simulated linear positions whose TP/SL fired by as_of."""
    return _agent_tool_response(settle_exchange_impl(as_of=as_of, interval=interval, fee_rate=fee_rate))


@function_tool
def close_position(
    position_id: str | None = None,
    category: str | None = None,
    symbol: str | None = None,
    side: str | None = None,
    price: float | None = None,
    as_of: str | None = None,
    mark_interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict:
    """Close a simulated linear position at the cached mark price for as_of. Optional price must match that mark."""
    return _agent_tool_response(
        close_position_impl(
            position_id=position_id,
            category=category,
            symbol=symbol,
            side=side,
            price=price,
            as_of=as_of,
            mark_interval=mark_interval,
            fee_rate=fee_rate,
        )
    )
