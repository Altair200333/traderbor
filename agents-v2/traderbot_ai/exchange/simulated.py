from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from traderbot_ai.exchange.interface import CancelOrderError
from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs
from traderbot_ai.simulator.execution import ExecutionEngine, Order
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache, candle_freshness
from traderbot_ai.tools.market import INTERVAL_MS, normalize_symbol, parse_time_ms


Category = Literal["spot", "linear"]
Side = Literal["Buy", "Sell"]
OrderType = Literal["Market", "Limit"]
MarketUnit = Literal["baseCoin", "quoteCoin"]

DEFAULT_EXCHANGE_STATE_PATH = DATA_DIR / "exchange_simulator.json"
DEFAULT_EXCHANGE_EVENTS_PATH = DATA_DIR / "exchange_events.jsonl"
MAX_SIM_LEVERAGE = 100.0
TIME_IN_FORCE_VALUES = {"GTC", "IOC", "FOK", "PostOnly", "RPI"}
_STATE_LOCK = RLock()


@dataclass(frozen=True)
class Mark:
    symbol: str
    asset: str
    price: float
    timestamp_ms: int


class SimulatedExchange:
    backend_kind = "simulated"

    def __init__(
        self,
        path: str | Path = DEFAULT_EXCHANGE_STATE_PATH,
        events_path: str | Path = DEFAULT_EXCHANGE_EVENTS_PATH,
        cache: LocalMarketCache | None = None,
    ) -> None:
        self.path = Path(path)
        self.events_path = Path(events_path)
        self.cache = cache or LocalMarketCache()
        ensure_runtime_dirs()

    def _default_state(self, balances: dict[str, float], as_of: str | int | float | None) -> dict[str, Any]:
        return {
            "schema_version": "exchange-sim/v1",
            "balances": {
                asset.upper(): {"free": _non_negative(amount, asset), "locked": 0.0}
                for asset, amount in balances.items()
                if float(amount) != 0.0
            },
            "orders": [],
            "positions": [],
            "closed_positions": [],
            "leverage": {},
            "updated_at": _now_iso(),
            "as_of_ms": parse_time_ms(as_of),
        }

    def reset(self, balances: dict[str, float] | None = None, as_of: str | int | float | None = None) -> dict[str, Any]:
        with _STATE_LOCK:
            initial = {"USDT": 1000.0} if balances is None else balances
            state = self._default_state(initial, as_of)
            self._save(state)
            self._reset_events()
            self._log("exchange_reset", {"balances": state["balances"], "as_of_ms": state["as_of_ms"]})
            return state

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            state = self._default_state({"USDT": 1000.0}, as_of=None)
            self._save(state)
            return state
        state = json.loads(self.path.read_text(encoding="utf-8"))
        state.setdefault("orders", [])
        state.setdefault("positions", [])
        state.setdefault("closed_positions", [])
        state.setdefault("leverage", {})
        return state

    def wallet_summary(
        self,
        symbols: str | list[str] | None = None,
        as_of: str | int | float | None = None,
        mark_interval: str = "1m",
    ) -> dict[str, Any]:
        mark_interval = _parse_interval(mark_interval)
        with _STATE_LOCK:
            state = self.load()
            as_of_ms = parse_time_ms(as_of)
            if as_of_ms is not None:
                self._ensure_not_reading_past(state, as_of_ms)
            requested_symbols = _split_symbols(symbols)
            assets = set(state.get("balances", {}).keys())
            for symbol in requested_symbols:
                assets.add(_base_asset(symbol))

            marks = {mark.asset: mark for mark in self._marks_for_assets(assets, requested_symbols, as_of_ms, mark_interval)}
            balances = []
            free_usdt = 0.0
            locked_usdt = 0.0
            unpriced_assets = []
            for asset in sorted(assets):
                balance = state.get("balances", {}).get(asset, {"free": 0.0, "locked": 0.0})
                free = float(balance.get("free", 0.0))
                locked = float(balance.get("locked", 0.0))
                rate = 1.0 if asset == "USDT" else marks.get(asset).price if asset in marks else None
                free_value = free * rate if rate is not None else None
                locked_value = locked * rate if rate is not None else None
                if free_value is not None:
                    free_usdt += free_value
                if locked_value is not None:
                    locked_usdt += locked_value
                if rate is None and (free or locked):
                    unpriced_assets.append(asset)
                balances.append(
                    {
                        "asset": asset,
                        "free": free,
                        "locked": locked,
                        "total": free + locked,
                        "usdt_rate": rate,
                        "free_usdt": free_value,
                        "locked_usdt": locked_value,
                        "total_usdt": None if rate is None else (free + locked) * rate,
                        "mark_time_ms": None if asset == "USDT" or asset not in marks else marks[asset].timestamp_ms,
                    }
                )

            positions = [self._position_summary(item, as_of_ms, mark_interval) for item in state.get("positions", [])]
            unrealized = sum(float(item.get("unrealized_pnl_usdt") or 0.0) for item in positions)
            unpriced_positions = [item["position_id"] for item in positions if item.get("mark_price") is None]
            total_wallet_usdt = free_usdt + locked_usdt
            return {
                "ok": True,
                "mode": "simulator",
                "as_of_ms": as_of_ms,
                "mark_interval": mark_interval,
                "balances": balances,
                "totals": {
                    "free_usdt": free_usdt,
                    "locked_usdt": locked_usdt,
                    "wallet_usdt": total_wallet_usdt,
                    "unrealized_pnl_usdt": unrealized,
                    "equity_usdt": total_wallet_usdt + unrealized,
                    "valuation_complete": not unpriced_assets and not unpriced_positions,
                    "unpriced_assets": unpriced_assets,
                    "unpriced_positions": unpriced_positions,
                },
                "open_orders": state.get("orders", []),
                "open_positions": positions,
                "leverage": state.get("leverage", {}),
                "state_path": str(self.path),
                "events_path": str(self.events_path),
            }

    def set_leverage(self, category: str, symbol: str, buyLeverage: str | float, sellLeverage: str | float) -> dict[str, Any]:
        if category != "linear":
            raise ValueError("simulator leverage is supported for linear category only")
        normalized = normalize_symbol(symbol)
        buy = _parse_leverage(buyLeverage)
        sell = _parse_leverage(sellLeverage)
        if buy != sell:
            raise ValueError("simulator uses one-way mode; buyLeverage must equal sellLeverage")
        with _STATE_LOCK:
            state = self.load()
            key = _leverage_key(category, normalized)
            state.setdefault("leverage", {})[key] = {
                "category": category,
                "symbol": normalized,
                "buyLeverage": buy,
                "sellLeverage": sell,
            }
            state["updated_at"] = _now_iso()
            self._save(state)
            self._log("set_leverage", state["leverage"][key])
            return {"leverage": state["leverage"][key], "state": state}

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
        expiresAtMs: int | float | None = None,
        entryPolicy: str | None = None,
        entryRefPrice: float | None = None,
    ) -> dict[str, Any]:
        mark_interval = _parse_interval(mark_interval)
        category = _parse_category(category)
        side = _parse_side(side)
        order_type = _parse_order_type(orderType)
        time_in_force = _parse_time_in_force(timeInForce)
        if order_type == "Limit" and time_in_force != "GTC":
            raise ValueError("simulator exchange currently supports GTC timeInForce for Limit orders only")
        if order_type == "Market" and time_in_force not in {"GTC", "IOC"}:
            raise ValueError("simulator exchange currently supports GTC or IOC timeInForce for Market orders only")
        market_unit = _parse_market_unit(marketUnit)
        if reduceOnly:
            raise ValueError("reduceOnly is not implemented in simulator exchange yet")
        if positionIdx != 0:
            raise ValueError("simulator exchange currently supports one-way positionIdx=0 only")
        if tpslMode != "Full":
            raise ValueError("simulator exchange currently supports tpslMode=Full only")
        qty = _positive(qty, "qty")
        take_profit = _optional_positive(takeProfit, "takeProfit")
        stop_loss = _optional_positive(stopLoss, "stopLoss")
        fee_rate = _non_negative(fee_rate, "fee_rate")
        as_of_ms = parse_time_ms(as_of)
        if as_of_ms is None:
            raise ValueError("as_of is required")
        expires_at_ms = None if expiresAtMs is None else int(expiresAtMs)
        if expires_at_ms is not None:
            if order_type != "Limit" or category != "linear":
                raise ValueError("expiresAtMs is only supported for linear Limit orders")
            if expires_at_ms <= as_of_ms:
                raise ValueError("expiresAtMs must be after as_of")
        entry_ref_price = _optional_positive(entryRefPrice, "entryRefPrice")
        normalized = normalize_symbol(symbol)
        with _STATE_LOCK:
            state = self.load()
            self._ensure_monotonic_as_of(state, as_of_ms)
            advanced = self._advance_state_to(state, as_of_ms, mark_interval, fee_rate=fee_rate)
            self._validate_order_link_id(state, orderLinkId)
            mark_price = self._order_price(normalized, order_type, price, as_of_ms, mark_interval)
            if category == "spot":
                if take_profit is not None or stop_loss is not None:
                    raise ValueError("spot TP/SL is not implemented in simulator exchange")
                result = self._place_spot_order(
                    state,
                    normalized,
                    side,
                    order_type,
                    qty,
                    mark_price,
                    time_in_force,
                    orderLinkId,
                    market_unit,
                    fee_rate,
                    as_of_ms,
                )
            else:
                if market_unit is not None:
                    raise ValueError("marketUnit is only valid for spot Market orders")
                result = self._place_linear_order(
                    state,
                    normalized,
                    side,
                    order_type,
                    qty,
                    mark_price,
                    time_in_force,
                    take_profit,
                    stop_loss,
                    orderLinkId,
                    leverage,
                    fee_rate,
                    as_of_ms,
                    expires_at_ms=expires_at_ms,
                    entry_policy=entryPolicy,
                    entry_ref_price=entry_ref_price,
                )
            state["updated_at"] = _now_iso()
            state["as_of_ms"] = as_of_ms
            self._save(state)
            self._log_advanced_events(advanced)
            self._log("place_order", result)
            return {"order": result, "advanced_before_order": advanced, "state": state}

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
        interval = _parse_interval(interval)
        if not order_id and not orderLinkId:
            raise ValueError("order_id or orderLinkId is required")
        expected_category = _parse_category(category) if category is not None else None
        expected_symbol = normalize_symbol(symbol) if symbol is not None else None
        as_of_ms = parse_time_ms(as_of)
        if as_of_ms is None:
            raise ValueError("as_of is required")
        fee_rate = _non_negative(fee_rate, "fee_rate")
        with _STATE_LOCK:
            state = self.load()
            self._ensure_monotonic_as_of(state, as_of_ms)
            advanced = self._advance_state_to(state, as_of_ms, interval, fee_rate)
            filled_orders = advanced["filled_orders"]
            remaining = []
            matches = []
            for order in state.get("orders", []):
                if _cancel_order_matches(order, order_id, orderLinkId, expected_category, expected_symbol):
                    matches.append(order)
                else:
                    remaining.append(order)
            if not matches:
                target_fills = [
                    order
                    for order in filled_orders
                    if _cancel_order_matches(order, order_id, orderLinkId, expected_category, expected_symbol)
                ]
                if advanced["filled_orders"] or advanced["closed_positions"]:
                    state["updated_at"] = _now_iso()
                    state["as_of_ms"] = as_of_ms
                    self._save(state)
                    self._log_advanced_events(advanced)
                if target_fills:
                    raise CancelOrderError("open order was already filled before cancel as_of", target_fills, state)
                raise ValueError("open order not found")
            if len(matches) > 1:
                raise ValueError("multiple open orders matched; orderLinkId must be unique")
            cancelled = matches[0]
            self._release_order_lock(state, cancelled)
            cancelled = {**cancelled, "status": "Cancelled", "cancelled_at": _now_iso()}
            state["orders"] = remaining
            state["updated_at"] = _now_iso()
            state["as_of_ms"] = as_of_ms
            self._save(state)
            self._log_advanced_events(advanced)
            self._log("cancel_order", cancelled)
            return {
                "cancelled_order": cancelled,
                "filled_orders_before_cancel": advanced["filled_orders"],
                "closed_positions_before_cancel": advanced["closed_positions"],
                "state": state,
            }

    def settle(self, as_of: str | int | float, interval: str = "1m", fee_rate: float = 0.0) -> dict[str, Any]:
        interval = _parse_interval(interval)
        as_of_ms = parse_time_ms(as_of)
        if as_of_ms is None:
            raise ValueError("as_of is required")
        fee_rate = _non_negative(fee_rate, "fee_rate")
        with _STATE_LOCK:
            state = self.load()
            self._ensure_monotonic_as_of(state, as_of_ms)
            advanced = self._advance_state_to(state, as_of_ms, interval, fee_rate)
            state["updated_at"] = _now_iso()
            state["as_of_ms"] = as_of_ms
            self._save(state)
            self._log_advanced_events(advanced)
            return {
                "settled_until_ms": as_of_ms,
                "filled_orders": advanced["filled_orders"],
                "closed_positions": advanced["closed_positions"],
                "expired_orders": advanced["expired_orders"],
                "state": state,
            }

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
        mark_interval = _parse_interval(mark_interval)
        fee_rate = _non_negative(fee_rate, "fee_rate")
        expected_category = _parse_category(category) if category is not None else None
        expected_symbol = normalize_symbol(symbol) if symbol is not None else None
        expected_side = _parse_side(side) if side is not None else None
        if position_id is None and expected_symbol is None:
            raise ValueError("position_id or symbol is required")
        as_of_ms = parse_time_ms(as_of)
        if as_of_ms is None:
            raise ValueError("as_of is required")
        with _STATE_LOCK:
            state = self.load()
            self._ensure_monotonic_as_of(state, as_of_ms)
            advanced = self._advance_state_to(state, as_of_ms, mark_interval, fee_rate)
            remaining = []
            matches = []
            for position in state.get("positions", []):
                if _position_matches_close(position, position_id, expected_category, expected_symbol, expected_side):
                    matches.append(position)
                else:
                    remaining.append(position)
            if len(matches) > 1:
                self._save_advanced_if_any(state, as_of_ms, advanced)
                raise ValueError("multiple open positions matched; provide position_id or side")
            target = matches[0] if matches else None
            if target is None:
                advanced_targets = [
                    position
                    for position in advanced["closed_positions"]
                    if _position_matches_close(position, position_id, expected_category, expected_symbol, expected_side)
                ]
                if len(advanced_targets) > 1:
                    self._save_advanced_if_any(state, as_of_ms, advanced)
                    raise ValueError("multiple closed positions matched; provide position_id or side")
                if advanced_targets:
                    self._save_advanced_if_any(state, as_of_ms, advanced)
                    return {"closed_position": advanced_targets[0], "triggered_before_manual_close": True, "state": state}
                self._save_advanced_if_any(state, as_of_ms, advanced)
                raise ValueError("open position not found")
            if as_of_ms < int(target["opened_at_ms"]):
                self._save_advanced_if_any(state, as_of_ms, advanced)
                raise ValueError("as_of must be at or after position opened_at_ms")
            close_fee_rate = _position_close_fee_rate(target, fee_rate)
            try:
                mark_price = self._order_price(target["symbol"], "Market", None, as_of_ms, mark_interval)
                if price is not None:
                    requested_price = _positive(price, "price")
                    if not _nearly_equal(requested_price, mark_price):
                        raise ValueError("manual close price must match the cached mark price at as_of")
            except Exception:
                self._save_advanced_if_any(state, as_of_ms, advanced)
                raise
            exit_price = mark_price
            exit_time_ms = as_of_ms
            exit_notional = float(target["qty"]) * exit_price
            entry_notional = float(target["qty"]) * float(target["entry_price"])
            fees = close_fee_rate * (entry_notional + exit_notional)
            closed = self._close_position_record(state, target, exit_price, exit_time_ms, "Manual", fees, "manual", "manual", False)
            state["positions"] = remaining
            state.setdefault("closed_positions", []).append(closed)
            state["updated_at"] = _now_iso()
            state["as_of_ms"] = exit_time_ms
            self._save(state)
            self._log_advanced_events(advanced)
            self._log("position_closed", closed)
            return {"closed_position": closed, "state": state}

    def _advance_state_to(self, state: dict[str, Any], as_of_ms: int, interval: str, fee_rate: float) -> dict[str, list[dict[str, Any]]]:
        filled_orders, expired_orders = self._fill_limit_orders(state, as_of_ms, interval, fee_rate)
        remaining = []
        closed = []
        for position in list(state.get("positions", [])):
            if position.get("status") != "Open":
                remaining.append(position)
                continue
            closed_position = self._triggered_position_close(state, position, as_of_ms, interval, fee_rate)
            if closed_position is None:
                remaining.append(position)
                continue
            closed.append(closed_position)
        state["positions"] = remaining
        state.setdefault("closed_positions", []).extend(closed)
        return {"filled_orders": filled_orders, "closed_positions": closed, "expired_orders": expired_orders}

    def _log_advanced_events(self, advanced: dict[str, list[dict[str, Any]]]) -> None:
        self._log_filled_orders(advanced.get("filled_orders", []))
        for expired_order in advanced.get("expired_orders", []):
            self._log("order_expired", expired_order)
        for closed_position in advanced.get("closed_positions", []):
            self._log("position_closed", closed_position)

    def _save_advanced_if_any(self, state: dict[str, Any], as_of_ms: int, advanced: dict[str, list[dict[str, Any]]]) -> None:
        if not advanced.get("filled_orders") and not advanced.get("closed_positions") and not advanced.get("expired_orders"):
            return
        state["updated_at"] = _now_iso()
        state["as_of_ms"] = as_of_ms
        self._save(state)
        self._log_advanced_events(advanced)

    def _log_filled_orders(self, filled_orders: list[dict[str, Any]]) -> None:
        for filled_order in filled_orders:
            self._log("order_filled", filled_order)

    def _place_spot_order(
        self,
        state: dict[str, Any],
        symbol: str,
        side: Side,
        order_type: OrderType,
        qty: float,
        price: float,
        time_in_force: str,
        order_link_id: str | None,
        market_unit: MarketUnit | None,
        fee_rate: float,
        as_of_ms: int | None,
    ) -> dict[str, Any]:
        if market_unit is not None and order_type != "Market":
            raise ValueError("marketUnit is only valid for spot Market orders")
        base = _base_asset(symbol)
        base_qty = qty
        quote_qty = qty * price
        if order_type == "Market":
            market_unit = market_unit or ("quoteCoin" if side == "Buy" else "baseCoin")
            if market_unit == "quoteCoin":
                quote_qty = qty
                base_qty = quote_qty / price
            else:
                base_qty = qty
                quote_qty = base_qty * price
        fee = _spot_fee(side, base, base_qty, quote_qty, fee_rate)
        record = self._order_record("Filled" if order_type == "Market" else "New", "spot", symbol, side, order_type, qty, price, time_in_force, order_link_id, as_of_ms)
        record.update({**fee, "base_qty": base_qty, "quote_qty": quote_qty, "fee_rate": fee_rate})
        if market_unit is not None:
            record["marketUnit"] = market_unit
        if order_type == "Market":
            if side == "Buy":
                self._debit_free(state, "USDT", quote_qty)
                self._balance(state, base)["free"] += float(record["net_base_qty"])
            else:
                self._debit_free(state, base, base_qty)
                self._balance(state, "USDT")["free"] += float(record["net_quote_qty"])
            return record
        if side == "Buy":
            self._lock_balance(state, "USDT", quote_qty)
            record["locked_asset"] = "USDT"
            record["locked_amount"] = quote_qty
        else:
            self._lock_balance(state, base, base_qty)
            record["locked_asset"] = base
            record["locked_amount"] = base_qty
        state.setdefault("orders", []).append(record)
        return record

    def _place_linear_order(
        self,
        state: dict[str, Any],
        symbol: str,
        side: Side,
        order_type: OrderType,
        qty: float,
        price: float,
        time_in_force: str,
        take_profit: float | None,
        stop_loss: float | None,
        order_link_id: str | None,
        leverage: float | None,
        fee_rate: float,
        as_of_ms: int | None,
        expires_at_ms: int | None = None,
        entry_policy: str | None = None,
        entry_ref_price: float | None = None,
    ) -> dict[str, Any]:
        leverage_value = _parse_leverage(leverage) if leverage is not None else self._leverage_for(state, "linear", symbol, side)
        self._validate_tpsl(side, price, take_profit, stop_loss)
        margin = qty * price / leverage_value
        self._lock_balance(state, "USDT", margin)
        record = self._order_record("Filled" if order_type == "Market" else "New", "linear", symbol, side, order_type, qty, price, time_in_force, order_link_id, as_of_ms)
        record.update(
            {
                "leverage": leverage_value,
                "fee_rate": fee_rate,
                "margin_usdt": margin,
                "takeProfit": take_profit,
                "stopLoss": stop_loss,
                "locked_asset": "USDT",
                "locked_amount": margin,
            }
        )
        if expires_at_ms is not None:
            record["expires_at_ms"] = expires_at_ms
            record["expires_at"] = _iso(expires_at_ms)
        if entry_policy is not None:
            record["entry_policy"] = entry_policy
        if entry_ref_price is not None:
            record["entry_ref_price"] = entry_ref_price
        if order_type == "Market":
            position = self._position_from_order(record, opened_at_ms=record["created_at_ms"])
            state.setdefault("positions", []).append(position)
            return {**record, "position": position}
        state.setdefault("orders", []).append(record)
        return record

    def _fill_limit_orders(self, state: dict[str, Any], as_of_ms: int, interval: str, fee_rate: float = 0.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        remaining = []
        filled = []
        expired = []
        for order in state.get("orders", []):
            if order.get("orderType") != "Limit":
                remaining.append(order)
                continue
            candle = self._limit_fill_candle(order, as_of_ms, interval)
            if candle is not None:
                fill = self._fill_limit_order(state, order, candle, fee_rate)
                filled.append(fill)
                continue
            expires_at_ms = order.get("expires_at_ms")
            if expires_at_ms is not None and as_of_ms >= int(expires_at_ms):
                self._release_order_lock(state, order)
                expired.append({**order, "status": "Expired", "expired_at_ms": int(expires_at_ms), "expired_at": _iso(int(expires_at_ms))})
                continue
            remaining.append(order)
        state["orders"] = remaining
        return filled, expired

    def _limit_fill_candle(self, order: dict[str, Any], as_of_ms: int, interval: str) -> Candle | None:
        candles = self.cache.get_candles(
            symbol=order["symbol"],
            interval=interval,
            start_ms=int(order["created_at_ms"]),
            end_ms=as_of_ms + 1,
            as_of_ms=as_of_ms,
        )
        side = _parse_side(order["side"])
        limit_price = float(order["price"])
        expires_at_ms = order.get("expires_at_ms")
        for candle in candles:
            if expires_at_ms is not None and candle.open_time >= int(expires_at_ms):
                return None
            if side == "Buy" and candle.low <= limit_price:
                return candle
            if side == "Sell" and candle.high >= limit_price:
                return candle
        return None

    def _fill_limit_order(self, state: dict[str, Any], order: dict[str, Any], candle: Candle, fee_rate: float) -> dict[str, Any]:
        filled_at_ms = int(candle.open_time)
        fill_price = (
            self._linear_limit_fill_price(state, order, candle)
            if order["category"] == "linear"
            else self._limit_fill_price(order, candle)
        )
        base_qty = float(order.get("base_qty") or order["qty"])
        quote_qty = base_qty * fill_price
        fill_fee_rate = _order_fill_fee_rate(order, fee_rate)
        spot_fee = _spot_fee(order["side"], _base_asset(order["symbol"]), base_qty, quote_qty, fill_fee_rate) if order["category"] == "spot" else {}
        fill = {
            **order,
            "limit_price": float(order["price"]),
            "price": fill_price,
            "status": "Filled",
            "filled_at_ms": filled_at_ms,
            "filled_at": _iso(filled_at_ms),
            "fill_price": fill_price,
            "fill_candle": candle.compact(),
            "base_qty": base_qty,
            "quote_qty": quote_qty,
            "fee_rate": fill_fee_rate,
            **spot_fee,
        }
        if order["category"] == "spot":
            self._settle_spot_limit_fill(state, fill)
            return fill
        self._settle_linear_limit_margin(state, fill)
        position_opened_at_ms = int(candle.close_time) + 1
        fill["position_opened_at_ms"] = position_opened_at_ms
        fill["fill_time_model"] = "position_opens_after_fill_candle_close"
        position = self._position_from_order(fill, opened_at_ms=position_opened_at_ms)
        state.setdefault("positions", []).append(position)
        fill["position"] = position
        return fill

    def _limit_fill_price(self, order: dict[str, Any], candle: Candle) -> float:
        limit_price = float(order["price"])
        side = _parse_side(order["side"])
        if side == "Buy" and candle.open <= limit_price:
            return float(candle.open)
        if side == "Sell" and candle.open >= limit_price:
            return float(candle.open)
        return limit_price

    def _linear_limit_fill_price(self, state: dict[str, Any], order: dict[str, Any], candle: Candle) -> float:
        limit_price = float(order["price"])
        candidate = self._limit_fill_price(order, candle)
        if _nearly_equal(candidate, limit_price):
            return limit_price
        side = _parse_side(order["side"])
        try:
            self._validate_tpsl(side, candidate, order.get("takeProfit"), order.get("stopLoss"))
        except ValueError:
            return limit_price
        current_margin = float(order["margin_usdt"])
        required_margin = float(order["qty"]) * candidate / float(order["leverage"])
        if required_margin > current_margin:
            free_usdt = self._balance(state, "USDT")["free"]
            if free_usdt + 1e-12 < required_margin - current_margin:
                return limit_price
        return candidate

    def _settle_spot_limit_fill(self, state: dict[str, Any], order: dict[str, Any]) -> None:
        locked_asset = order.get("locked_asset")
        locked_amount = float(order.get("locked_amount") or 0.0)
        if locked_asset and locked_amount:
            balance = self._balance(state, locked_asset)
            if balance["locked"] + 1e-12 < locked_amount:
                raise ValueError(f"locked {locked_asset} is below order lock")
            balance["locked"] -= locked_amount
        base = _base_asset(order["symbol"])
        if order["side"] == "Buy":
            refund = locked_amount - float(order["quote_qty"])
            if refund > 0:
                self._balance(state, "USDT")["free"] += refund
            self._balance(state, base)["free"] += float(order["net_base_qty"])
        else:
            self._balance(state, "USDT")["free"] += float(order["net_quote_qty"])

    def _settle_linear_limit_margin(self, state: dict[str, Any], order: dict[str, Any]) -> None:
        current_margin = float(order["margin_usdt"])
        required_margin = float(order["qty"]) * float(order["price"]) / float(order["leverage"])
        usdt = self._balance(state, "USDT")
        if required_margin > current_margin:
            self._debit_free(state, "USDT", required_margin - current_margin)
            usdt["locked"] += required_margin - current_margin
        elif required_margin < current_margin:
            release = current_margin - required_margin
            if usdt["locked"] + 1e-12 < release:
                raise ValueError("locked USDT is below limit order margin release")
            usdt["locked"] -= release
            usdt["free"] += release
        order["margin_usdt"] = required_margin
        order["locked_amount"] = required_margin

    def _position_from_order(self, order: dict[str, Any], opened_at_ms: int) -> dict[str, Any]:
        return {
            "position_id": order["order_id"],
            "orderLinkId": order.get("orderLinkId"),
            "category": "linear",
            "symbol": order["symbol"],
            "side": order["side"],
            "qty": float(order["qty"]),
            "entry_price": float(order["price"]),
            "notional_usdt": float(order["qty"]) * float(order["price"]),
            "margin_usdt": float(order["margin_usdt"]),
            "leverage": float(order["leverage"]),
            "fee_rate": float(order.get("fee_rate") or 0.0),
            "takeProfit": order.get("takeProfit"),
            "stopLoss": order.get("stopLoss"),
            "opened_at_ms": opened_at_ms,
            "opened_at": _iso(opened_at_ms),
            "status": "Open",
        }

    def _execution_order_for_position(self, position: dict[str, Any]) -> Order | None:
        entry = float(position["entry_price"])
        take_profit = position.get("takeProfit")
        stop_loss = position.get("stopLoss")
        liquidation = self._liquidation_price(position)
        if position["side"] == "Buy":
            synthetic_stop = max(liquidation, 0.0 if stop_loss is None else float(stop_loss))
            synthetic_take = entry * 1_000_000_000.0 if take_profit is None else float(take_profit)
            kind = "long"
        else:
            synthetic_take = 0.0 if take_profit is None else float(take_profit)
            synthetic_stop = min(liquidation, entry * 1_000_000_000.0 if stop_loss is None else float(stop_loss))
            kind = "short"
        return Order.from_values(
            kind=kind,
            symbol=position["symbol"],
            entry_price=entry,
            stop_loss=synthetic_stop,
            take_profit=synthetic_take,
            amount=float(position["qty"]) * entry,
            opened_at=int(position["opened_at_ms"]),
            id=position["position_id"],
        )

    def _triggered_position_close(
        self,
        state: dict[str, Any],
        position: dict[str, Any],
        as_of_ms: int,
        interval: str,
        fee_rate: float,
    ) -> dict[str, Any] | None:
        order = self._execution_order_for_position(position)
        if order is None:
            return None
        result = ExecutionEngine(self.cache).resolve_order(order, interval=interval, scan_until=as_of_ms, fee_rate=_position_close_fee_rate(position, fee_rate))
        if result.status != "closed" or result.exit_price is None or result.exit_time_ms is None:
            return None
        exit_price = float(result.exit_price)
        exit_reason = str(result.exit_reason)
        if self._is_liquidation_exit(position, exit_price):
            exit_reason = "Liquidation"
        return self._close_position_record(
            state=state,
            position=position,
            exit_price=exit_price,
            exit_time_ms=int(result.exit_time_ms),
            exit_reason=exit_reason,
            fees=float(result.fees),
            resolution_interval=result.resolution_interval,
            source_interval=result.source_interval,
            ambiguous=result.ambiguous,
        )

    def _liquidation_price(self, position: dict[str, Any]) -> float:
        entry = float(position["entry_price"])
        qty = float(position["qty"])
        margin = float(position["margin_usdt"])
        distance = margin / qty
        if position["side"] == "Buy":
            return max(0.0, entry - distance)
        return entry + distance

    def _is_liquidation_exit(self, position: dict[str, Any], exit_price: float) -> bool:
        liquidation = self._liquidation_price(position)
        if position["side"] == "Buy":
            return exit_price <= liquidation or _nearly_equal(exit_price, liquidation)
        return exit_price >= liquidation or _nearly_equal(exit_price, liquidation)

    def _close_position_record(
        self,
        state: dict[str, Any],
        position: dict[str, Any],
        exit_price: float,
        exit_time_ms: int,
        exit_reason: str,
        fees: float,
        resolution_interval: str | None,
        source_interval: str | None,
        ambiguous: bool,
    ) -> dict[str, Any]:
        margin = float(position["margin_usdt"])
        qty = float(position["qty"])
        entry = float(position["entry_price"])
        gross_pnl = _position_pnl(position["side"], qty, entry, exit_price)
        raw_return = margin + gross_pnl - fees
        cash_returned = max(0.0, raw_return)
        usdt = self._balance(state, "USDT")
        if usdt["locked"] + 1e-12 < margin:
            raise ValueError("locked USDT is below position margin")
        usdt["locked"] -= margin
        if abs(usdt["locked"]) < 1e-12:
            usdt["locked"] = 0.0
        usdt["free"] += cash_returned
        closed = {
            **position,
            "status": "Closed",
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "exit_time_ms": exit_time_ms,
            "exit_time": _iso(exit_time_ms),
            "gross_pnl_usdt": gross_pnl,
            "fees_usdt": fees,
            "cash_returned_usdt": cash_returned,
            "realized_pnl_usdt": cash_returned - margin,
            "resolution_interval": resolution_interval,
            "source_interval": source_interval,
            "ambiguous": ambiguous,
        }
        return closed

    def _order_price(self, symbol: str, order_type: OrderType, price: float | None, as_of_ms: int | None, mark_interval: str) -> float:
        if order_type == "Limit":
            if price is None:
                raise ValueError("price is required for Limit orders")
            return _positive(price, "price")
        candle = self._latest_mark_candle(symbol=symbol, interval=mark_interval, as_of_ms=as_of_ms)
        if candle is None:
            raise ValueError(f"no cached mark price for {symbol} {mark_interval}")
        return float(candle.close)

    def _position_summary(self, position: dict[str, Any], as_of_ms: int | None, mark_interval: str) -> dict[str, Any]:
        candle = self._latest_mark_candle(position["symbol"], mark_interval, as_of_ms=as_of_ms)
        mark = float(candle.close) if candle is not None else None
        gross = None if mark is None else _position_pnl(position["side"], float(position["qty"]), float(position["entry_price"]), mark)
        capped = None if gross is None else max(-float(position["margin_usdt"]), gross)
        return {
            **position,
            "mark_price": mark,
            "mark_time_ms": None if candle is None else candle.close_time,
            "unrealized_gross_pnl_usdt": gross,
            "unrealized_pnl_usdt": capped,
            "liquidation_price": self._liquidation_price(position),
        }

    def _marks_for_assets(self, assets: set[str], symbols: list[str], as_of_ms: int | None, mark_interval: str) -> list[Mark]:
        result = []
        symbol_by_asset = {_base_asset(symbol): normalize_symbol(symbol) for symbol in symbols}
        for asset in sorted(assets):
            if asset == "USDT":
                continue
            symbol = symbol_by_asset.get(asset, f"{asset}USDT")
            candle = self._latest_mark_candle(symbol=symbol, interval=mark_interval, as_of_ms=as_of_ms)
            if candle is not None:
                result.append(Mark(symbol=candle.symbol, asset=asset, price=float(candle.close), timestamp_ms=int(candle.close_time)))
        return result

    def _latest_mark_candle(self, symbol: str, interval: str, as_of_ms: int | None) -> Candle | None:
        candle = self.cache.latest_candle(symbol=symbol, interval=interval, as_of_ms=as_of_ms)
        if candle is None:
            return candle
        if not candle_freshness(candle, interval, as_of_ms)["fresh"]:
            return None
        return candle

    def _release_order_lock(self, state: dict[str, Any], order: dict[str, Any]) -> None:
        asset = order.get("locked_asset")
        amount = float(order.get("locked_amount") or 0.0)
        if asset and amount:
            balance = self._balance(state, asset)
            if balance["locked"] + 1e-12 < amount:
                raise ValueError(f"locked {asset} is below order lock")
            balance["locked"] -= amount
            balance["free"] += amount
            if abs(balance["locked"]) < 1e-12:
                balance["locked"] = 0.0

    def _lock_balance(self, state: dict[str, Any], asset: str, amount: float) -> None:
        self._debit_free(state, asset, amount)
        self._balance(state, asset)["locked"] += amount

    def _debit_free(self, state: dict[str, Any], asset: str, amount: float) -> None:
        balance = self._balance(state, asset)
        if balance["free"] + 1e-12 < amount:
            raise ValueError(f"insufficient free {asset}: need {amount}, have {balance['free']}")
        balance["free"] -= amount
        if abs(balance["free"]) < 1e-12:
            balance["free"] = 0.0

    def _balance(self, state: dict[str, Any], asset: str) -> dict[str, float]:
        return state.setdefault("balances", {}).setdefault(asset.upper(), {"free": 0.0, "locked": 0.0})

    def _ensure_monotonic_as_of(self, state: dict[str, Any], as_of_ms: int) -> None:
        previous = state.get("as_of_ms")
        if previous is not None and int(as_of_ms) < int(previous):
            raise ValueError(f"as_of cannot move backwards: {as_of_ms} < {previous}")

    def _ensure_not_reading_past(self, state: dict[str, Any], as_of_ms: int) -> None:
        current = state.get("as_of_ms")
        if current is not None and int(as_of_ms) < int(current):
            raise ValueError(f"wallet as_of is before current simulator state: {as_of_ms} < {current}")

    def _leverage_for(self, state: dict[str, Any], category: str, symbol: str, side: Side) -> float:
        item = state.get("leverage", {}).get(_leverage_key(category, symbol))
        if item is None:
            return 1.0
        return float(item["buyLeverage"] if side == "Buy" else item["sellLeverage"])

    def _validate_tpsl(self, side: Side, price: float, take_profit: float | None, stop_loss: float | None) -> None:
        if side == "Buy":
            if take_profit is not None and take_profit <= price:
                raise ValueError("Buy linear order requires takeProfit > price")
            if stop_loss is not None and stop_loss >= price:
                raise ValueError("Buy linear order requires stopLoss < price")
        if side == "Sell":
            if take_profit is not None and take_profit >= price:
                raise ValueError("Sell linear order requires takeProfit < price")
            if stop_loss is not None and stop_loss <= price:
                raise ValueError("Sell linear order requires stopLoss > price")

    def _validate_order_link_id(self, state: dict[str, Any], order_link_id: str | None) -> None:
        if not order_link_id:
            return
        for collection in ("orders", "positions"):
            for item in state.get(collection, []):
                if item.get("orderLinkId") == order_link_id:
                    raise ValueError("orderLinkId must be unique")

    def _order_record(
        self,
        status: str,
        category: Category,
        symbol: str,
        side: Side,
        order_type: OrderType,
        qty: float,
        price: float,
        time_in_force: str,
        order_link_id: str | None,
        as_of_ms: int | None,
    ) -> dict[str, Any]:
        created_ms = as_of_ms if as_of_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
        return {
            "order_id": _id("ord"),
            "orderLinkId": order_link_id,
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "price": price,
            "timeInForce": time_in_force,
            "status": status,
            "created_at_ms": created_ms,
            "created_at": _iso(created_ms),
        }

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

    def _reset_events(self) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")

    def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        event = {"event_id": _id("evt"), "type": event_type, "timestamp": _now_iso(), "payload": payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _parse_category(value: str) -> Category:
    if value == "spot":
        return "spot"
    if value == "linear":
        return "linear"
    raise ValueError("category must be spot or linear")


def _parse_side(value: str) -> Side:
    if value in {"Buy", "buy", "long"}:
        return "Buy"
    if value in {"Sell", "sell", "short"}:
        return "Sell"
    raise ValueError("side must be Buy or Sell")


def _parse_order_type(value: str) -> OrderType:
    if value in {"Market", "market"}:
        return "Market"
    if value in {"Limit", "limit"}:
        return "Limit"
    raise ValueError("orderType must be Market or Limit")


def _parse_time_in_force(value: str) -> str:
    aliases = {"gtc": "GTC", "ioc": "IOC", "fok": "FOK", "postonly": "PostOnly", "rpi": "RPI"}
    normalized = aliases.get(str(value).strip().lower(), value)
    if normalized not in TIME_IN_FORCE_VALUES:
        raise ValueError(f"timeInForce must be one of {', '.join(sorted(TIME_IN_FORCE_VALUES))}")
    return normalized


def _parse_market_unit(value: str | None) -> MarketUnit | None:
    if value is None or value == "":
        return None
    if value in {"baseCoin", "quoteCoin"}:
        return value
    raise ValueError("marketUnit must be baseCoin or quoteCoin")


def _parse_interval(value: str) -> str:
    if value not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {value}")
    return value


def _cancel_order_matches(
    order: dict[str, Any],
    order_id: str | None,
    order_link_id: str | None,
    category: str | None,
    symbol: str | None,
) -> bool:
    id_matches = (
        order.get("order_id") == order_id
        if order_id
        else bool(order_link_id and order.get("orderLinkId") == order_link_id)
    )
    if not id_matches:
        return False
    if category is not None and order.get("category") != category:
        return False
    if symbol is not None and order.get("symbol") != symbol:
        return False
    return True


def _position_matches_close(
    position: dict[str, Any],
    position_id: str | None,
    category: str | None,
    symbol: str | None,
    side: str | None,
) -> bool:
    if position_id is not None and position.get("position_id") != position_id:
        return False
    if position_id is None and symbol is not None and position.get("symbol") != symbol:
        return False
    if category is not None and position.get("category") != category:
        return False
    if symbol is not None and position.get("symbol") != symbol:
        return False
    if side is not None and position.get("side") != side:
        return False
    return position_id is not None or symbol is not None


def _parse_leverage(value: str | float) -> float:
    leverage = _positive(value, "leverage")
    if leverage < 1 or leverage > MAX_SIM_LEVERAGE:
        raise ValueError(f"leverage must be between 1 and {MAX_SIM_LEVERAGE}")
    return leverage


def _optional_positive(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _positive(value, name)


def _positive(value: str | float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: str | float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _spot_fee(side: str, base_asset: str, base_qty: float, quote_qty: float, fee_rate: float) -> dict[str, Any]:
    fee_rate = _non_negative(fee_rate, "fee_rate")
    if fee_rate >= 1:
        raise ValueError("spot fee_rate must be less than 1")
    if side == "Buy":
        fee_amount = base_qty * fee_rate
        return {
            "fee_rate": fee_rate,
            "fee_asset": base_asset,
            "fee_amount": fee_amount,
            "net_base_qty": base_qty - fee_amount,
            "net_quote_qty": quote_qty,
        }
    fee_amount = quote_qty * fee_rate
    return {
        "fee_rate": fee_rate,
        "fee_asset": "USDT",
        "fee_amount": fee_amount,
        "net_base_qty": base_qty,
        "net_quote_qty": quote_qty - fee_amount,
    }


def _order_fill_fee_rate(order: dict[str, Any], fee_rate: float) -> float:
    requested_fee_rate = _non_negative(fee_rate, "fee_rate")
    if requested_fee_rate:
        return requested_fee_rate
    stored_fee_rate = order.get("fee_rate")
    if stored_fee_rate is None:
        return 0.0
    return _non_negative(float(stored_fee_rate), "order fee_rate")


def _position_close_fee_rate(position: dict[str, Any], fee_rate: float) -> float:
    requested_fee_rate = _non_negative(fee_rate, "fee_rate")
    if requested_fee_rate:
        return requested_fee_rate
    stored_fee_rate = position.get("fee_rate")
    if stored_fee_rate is None:
        return 0.0
    return _non_negative(float(stored_fee_rate), "position fee_rate")


def _nearly_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _position_pnl(side: str, qty: float, entry: float, mark: float) -> float:
    if side == "Buy":
        return (mark - entry) * qty
    return (entry - mark) * qty


def _split_symbols(symbols: str | list[str] | None) -> list[str]:
    if symbols is None:
        return []
    if isinstance(symbols, list):
        values = symbols
    else:
        values = [item.strip() for item in str(symbols).split(",")]
    return [normalize_symbol(item) for item in values if item]


def _base_asset(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    return normalized[:-4] if normalized.endswith("USDT") else normalized


def _leverage_key(category: str, symbol: str) -> str:
    return f"{category}:{normalize_symbol(symbol)}"


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
