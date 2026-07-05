from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class CancelOrderError(ValueError):
    def __init__(self, message: str, filled_orders: list[dict[str, Any]], state: dict[str, Any]) -> None:
        super().__init__(message)
        self.filled_orders = filled_orders
        self.state = state


@runtime_checkable
class ExchangeBackend(Protocol):
    def wallet_summary(
        self,
        symbols: str | list[str] | None = None,
        as_of: str | int | float | None = None,
        mark_interval: str = "1m",
    ) -> dict[str, Any]:
        ...

    def set_leverage(self, category: str, symbol: str, buyLeverage: str | float, sellLeverage: str | float) -> dict[str, Any]:
        ...

    def place_order(
        self,
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
        ...

    def cancel_order(
        self,
        order_id: str | None = None,
        orderLinkId: str | None = None,
        category: str | None = None,
        symbol: str | None = None,
        as_of: str | int | float | None = None,
        interval: str = "1m",
        fee_rate: float = 0.0,
    ) -> dict[str, Any]:
        ...

    def close_position(
        self,
        position_id: str | None = None,
        category: str | None = None,
        symbol: str | None = None,
        side: str | None = None,
        price: float | None = None,
        as_of: str | int | float | None = None,
        mark_interval: str = "1m",
        fee_rate: float = 0.0,
    ) -> dict[str, Any]:
        ...


@runtime_checkable
class SimulatedExchangeBackend(ExchangeBackend, Protocol):
    backend_kind: str
    path: Any
    events_path: Any

    def reset(self, balances: dict[str, float] | None = None, as_of: str | int | float | None = None) -> dict[str, Any]:
        ...

    def settle(self, as_of: str | int | float, interval: str = "1m", fee_rate: float = 0.0) -> dict[str, Any]:
        ...
