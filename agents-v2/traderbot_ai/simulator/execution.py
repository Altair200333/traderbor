from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.tools.market import INTERVAL_MS, normalize_symbol, parse_time_ms


OrderKind = Literal["long", "short"]
ExitReason = Literal["TP", "SL"]


@dataclass(frozen=True)
class Order:
    kind: OrderKind
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    amount: float
    opened_at_ms: int
    id: str | None = None

    @classmethod
    def from_values(
        cls,
        kind: str,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        amount: float,
        opened_at: str | int | float,
        id: str | None = None,
    ) -> "Order":
        opened_at_ms = parse_time_ms(opened_at)
        if opened_at_ms is None:
            raise ValueError("opened_at is required")
        return cls(
            kind=_parse_kind(kind),
            symbol=normalize_symbol(symbol),
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            amount=float(amount),
            opened_at_ms=opened_at_ms,
            id=id,
        )


@dataclass(frozen=True)
class ExitResult:
    status: Literal["open", "closed"]
    order: Order
    exit_reason: ExitReason | None = None
    exit_price: float | None = None
    exit_time_ms: int | None = None
    pnl: float = 0.0
    fees: float = 0.0
    cash_returned: float = 0.0
    ambiguous: bool = False
    resolution_interval: str | None = None
    source_interval: str | None = None
    candle: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "order": {
                "id": self.order.id,
                "kind": self.order.kind,
                "symbol": self.order.symbol,
                "entry_price": self.order.entry_price,
                "stop_loss": self.order.stop_loss,
                "take_profit": self.order.take_profit,
                "amount": self.order.amount,
                "opened_at_ms": self.order.opened_at_ms,
            },
            "exit_reason": self.exit_reason,
            "exit_price": self.exit_price,
            "exit_time_ms": self.exit_time_ms,
            "pnl": self.pnl,
            "fees": self.fees,
            "cash_returned": self.cash_returned,
            "ambiguous": self.ambiguous,
            "resolution_interval": self.resolution_interval,
            "source_interval": self.source_interval,
            "candle": self.candle,
        }


class ExecutionEngine:
    def __init__(self, cache: LocalMarketCache | None = None) -> None:
        self.cache = cache

    def validate_order(self, order: Order) -> None:
        if order.kind not in {"long", "short"}:
            raise ValueError(f"unsupported order kind: {order.kind}")
        for name, value in (
            ("entry_price", order.entry_price),
            ("stop_loss", order.stop_loss),
            ("take_profit", order.take_profit),
            ("amount", order.amount),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if order.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if order.amount <= 0:
            raise ValueError("amount must be positive")
        if order.kind == "long" and not (order.stop_loss < order.entry_price < order.take_profit):
            raise ValueError("long requires stop_loss < entry_price < take_profit")
        if order.kind == "short" and not (order.take_profit < order.entry_price < order.stop_loss):
            raise ValueError("short requires take_profit < entry_price < stop_loss")

    def resolve_order(
        self,
        order: Order,
        interval: str = "1m",
        scan_until: str | int | float | None = None,
        candles: list[Candle] | None = None,
        fee_rate: float = 0.0,
    ) -> ExitResult:
        self.validate_order(order)
        if fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")

        scan_until_ms = parse_time_ms(scan_until)
        if candles is None:
            if scan_until_ms is None:
                raise ValueError("scan_until is required when resolving against cached candles")
            if self.cache is None:
                raise ValueError("cache is required when candles are not provided")
            candles = self.cache.get_candles(
                symbol=order.symbol,
                interval=interval,
                start_ms=order.opened_at_ms,
                end_ms=scan_until_ms + 1,
                as_of_ms=scan_until_ms,
            )

        for candle in candles:
            if candle.open_time < order.opened_at_ms:
                continue
            if scan_until_ms is not None and candle.open_time > scan_until_ms:
                break
            if scan_until_ms is not None and candle.close_time > scan_until_ms:
                break

            gap_result = self._gap_exit(order, candle)
            if gap_result:
                reason, price = gap_result
                return self._closed(order, candle, reason, price, fee_rate, interval, interval, False)

            tp_hit, sl_hit = self._range_hits(order, candle)
            if tp_hit and sl_hit:
                agg_trade = self._resolve_with_agg_trades(order, candle, interval, fee_rate)
                if agg_trade is not None:
                    return agg_trade
                finer = self._resolve_with_finer_interval(order, candle, interval, fee_rate)
                if finer is not None:
                    return finer
                return self._closed(order, candle, "SL", order.stop_loss, fee_rate, interval, interval, True)
            if tp_hit:
                return self._closed(order, candle, "TP", order.take_profit, fee_rate, interval, interval, False)
            if sl_hit:
                return self._closed(order, candle, "SL", order.stop_loss, fee_rate, interval, interval, False)

        return ExitResult(status="open", order=order)

    def _gap_exit(self, order: Order, candle: Candle) -> tuple[ExitReason, float] | None:
        if order.kind == "long":
            if candle.open <= order.stop_loss:
                return "SL", candle.open
            if candle.open >= order.take_profit:
                return "TP", candle.open
        if order.kind == "short":
            if candle.open >= order.stop_loss:
                return "SL", candle.open
            if candle.open <= order.take_profit:
                return "TP", candle.open
        return None

    def _range_hits(self, order: Order, candle: Candle) -> tuple[bool, bool]:
        if order.kind == "long":
            return candle.high >= order.take_profit, candle.low <= order.stop_loss
        return candle.low <= order.take_profit, candle.high >= order.stop_loss

    def _resolve_with_finer_interval(
        self,
        order: Order,
        candle: Candle,
        interval: str,
        fee_rate: float,
    ) -> ExitResult | None:
        if self.cache is None:
            return None
        parent_ms = INTERVAL_MS.get(interval)
        if parent_ms is None:
            return None
        finer_intervals = [
            item
            for item in self.cache.available_intervals(order.symbol)
            if item in INTERVAL_MS and INTERVAL_MS[item] < parent_ms
        ]
        for finer_interval in finer_intervals:
            finer_ms = INTERVAL_MS[finer_interval]
            finer_candles = self.cache.get_candles(
                symbol=order.symbol,
                interval=finer_interval,
                start_ms=candle.open_time,
                end_ms=candle.close_time + 1,
                as_of_ms=candle.close_time,
            )
            if not _has_contiguous_coverage(finer_candles, candle.open_time, candle.close_time + 1, finer_ms):
                continue
            result = self.resolve_order(
                order=order,
                interval=finer_interval,
                candles=finer_candles,
                fee_rate=fee_rate,
            )
            if result.status == "closed":
                return ExitResult(
                    status=result.status,
                    order=result.order,
                    exit_reason=result.exit_reason,
                    exit_price=result.exit_price,
                    exit_time_ms=result.exit_time_ms,
                    pnl=result.pnl,
                    fees=result.fees,
                    cash_returned=result.cash_returned,
                    ambiguous=result.ambiguous,
                    resolution_interval=result.resolution_interval or finer_interval,
                    source_interval=interval,
                    candle=result.candle,
                )
        return None

    def _resolve_with_agg_trades(
        self,
        order: Order,
        candle: Candle,
        interval: str,
        fee_rate: float,
    ) -> ExitResult | None:
        if self.cache is None:
            return None
        start_ms = candle.open_time
        end_ms = candle.close_time + 1
        if not self.cache.has_agg_trade_coverage(order.symbol, start_ms, end_ms):
            return None
        for trade in self.cache.get_agg_trades(order.symbol, start_ms, end_ms):
            hit = self._trade_hit(order, trade.price)
            if hit is None:
                continue
            reason, price = hit
            return self._closed_result(
                order=order,
                reason=reason,
                exit_price=price,
                exit_time_ms=trade.trade_time,
                fee_rate=fee_rate,
                resolution_interval="aggTrades",
                source_interval=interval,
                ambiguous=False,
                source={"source": "agg_trade", "trade": trade.detailed()},
            )
        return None

    def _trade_hit(self, order: Order, price: float) -> tuple[ExitReason, float] | None:
        if order.kind == "long":
            if price >= order.take_profit:
                return "TP", order.take_profit
            if price <= order.stop_loss:
                return "SL", order.stop_loss
        else:
            if price <= order.take_profit:
                return "TP", order.take_profit
            if price >= order.stop_loss:
                return "SL", order.stop_loss
        return None

    def _closed(
        self,
        order: Order,
        candle: Candle,
        reason: ExitReason,
        exit_price: float,
        fee_rate: float,
        resolution_interval: str,
        source_interval: str,
        ambiguous: bool,
    ) -> ExitResult:
        return self._closed_result(
            order=order,
            reason=reason,
            exit_price=exit_price,
            exit_time_ms=candle.open_time,
            fee_rate=fee_rate,
            resolution_interval=resolution_interval,
            source_interval=source_interval,
            ambiguous=ambiguous,
            source=candle.detailed(),
        )

    def _closed_result(
        self,
        order: Order,
        reason: ExitReason,
        exit_price: float,
        exit_time_ms: int,
        fee_rate: float,
        resolution_interval: str,
        source_interval: str,
        ambiguous: bool,
        source: dict[str, Any],
    ) -> ExitResult:
        units = order.amount / order.entry_price
        if order.kind == "long":
            gross_pnl = (exit_price - order.entry_price) * units
        else:
            gross_pnl = (order.entry_price - exit_price) * units
        exit_notional = units * exit_price
        fees = fee_rate * (order.amount + exit_notional)
        raw_pnl = gross_pnl - fees
        cash_returned = max(0.0, order.amount + raw_pnl)
        pnl = cash_returned - order.amount
        return ExitResult(
            status="closed",
            order=order,
            exit_reason=reason,
            exit_price=exit_price,
            exit_time_ms=exit_time_ms,
            pnl=pnl,
            fees=fees,
            cash_returned=cash_returned,
            ambiguous=ambiguous,
            resolution_interval=resolution_interval,
            source_interval=source_interval,
            candle=source,
        )


def _parse_kind(value: str) -> OrderKind:
    if value == "long":
        return "long"
    if value == "short":
        return "short"
    raise ValueError(f"unsupported order kind: {value}")


def _has_contiguous_coverage(candles: list[Candle], start_ms: int, end_ms: int, interval_ms: int) -> bool:
    if not candles:
        return False
    expected = start_ms
    for candle in candles:
        if candle.open_time != expected:
            return False
        expected += interval_ms
        if expected >= end_ms:
            return True
    return False
