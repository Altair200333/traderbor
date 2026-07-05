from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs
from traderbot_ai.simulator.execution import ExecutionEngine, Order
from traderbot_ai.simulator.market_cache import LocalMarketCache
from traderbot_ai.tools.market import normalize_symbol, parse_time_ms


DEFAULT_SIM_PORTFOLIO_PATH = DATA_DIR / "simulator_portfolio.json"


@dataclass(frozen=True)
class MarkPrice:
    price: float
    timestamp_ms: int


class SimulatedPortfolio:
    def __init__(self, path: str | Path = DEFAULT_SIM_PORTFOLIO_PATH, cache: LocalMarketCache | None = None) -> None:
        self.path = Path(path)
        self.cache = cache or LocalMarketCache()
        ensure_runtime_dirs()

    def reset(self, balance_usdt: float = 1000.0, as_of: str | int | float | None = None) -> dict[str, Any]:
        if balance_usdt <= 0:
            raise ValueError("balance_usdt must be positive")
        as_of_ms = parse_time_ms(as_of)
        state = {
            "schema_version": "sim-portfolio/v1",
            "balance": {"USDT": float(balance_usdt)},
            "open_orders": [],
            "closed_trades": [],
            "last_settled_ms": as_of_ms,
            "updated_at": _now_iso(),
        }
        self._save(state)
        return state

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self.reset()
        return json.loads(self.path.read_text(encoding="utf-8"))

    def place_order(
        self,
        final_decision: str,
        symbol: str,
        price: float,
        stop_loss: float,
        take_profit: float,
        amount: float,
        opened_at: str | int | float,
    ) -> dict[str, Any]:
        state = self.load()
        order = Order.from_values(
            kind=final_decision,
            symbol=symbol,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            amount=amount,
            opened_at=opened_at,
            id=_order_id(opened_at),
        )
        ExecutionEngine().validate_order(order)

        balance = float(state.setdefault("balance", {}).setdefault("USDT", 0.0))
        if order.amount > balance:
            raise ValueError(f"amount {order.amount} exceeds simulated balance {balance}")

        order_record = _order_to_record(order)
        order_record["status"] = "open"
        order_record["opened_at_iso"] = _iso(order.opened_at_ms)
        state["balance"]["USDT"] = balance - order.amount
        state.setdefault("open_orders", []).append(order_record)
        state["updated_at"] = _now_iso()
        self._save(state)
        return {"order": order_record, "state": state}

    def settle(
        self,
        as_of: str | int | float,
        interval: str = "1m",
        fee_rate: float = 0.0,
    ) -> dict[str, Any]:
        as_of_ms = parse_time_ms(as_of)
        if as_of_ms is None:
            raise ValueError("as_of is required")

        state = self.load()
        engine = ExecutionEngine(self.cache)
        balance = float(state.setdefault("balance", {}).setdefault("USDT", 0.0))
        remaining = []
        closed = []

        for item in state.get("open_orders", []):
            order = _order_from_record(item)
            result = engine.resolve_order(order, interval=interval, scan_until=as_of_ms, fee_rate=fee_rate)
            if result.status == "closed":
                trade = result.to_dict()
                trade["closed_at_iso"] = _iso(result.exit_time_ms) if result.exit_time_ms is not None else None
                trade["opened_order"] = item
                balance += result.cash_returned
                closed.append(trade)
            else:
                remaining.append(item)

        state["balance"]["USDT"] = balance
        state["open_orders"] = remaining
        state.setdefault("closed_trades", []).extend(closed)
        state["last_settled_ms"] = as_of_ms
        state["updated_at"] = _now_iso()
        self._save(state)
        return {"settled_until_ms": as_of_ms, "closed_trades": closed, "state": state}

    def summary(self, as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
        state = self.load()
        as_of_ms = parse_time_ms(as_of)
        cash = float(state.get("balance", {}).get("USDT", 0.0))
        marks = []
        open_equity = 0.0
        for item in state.get("open_orders", []):
            order = _order_from_record(item)
            mark = self._mark_price(order.symbol, mark_interval, as_of_ms)
            if mark is None:
                marks.append({"order_id": order.id, "symbol": order.symbol, "error": "no cached mark"})
                open_equity += order.amount
                continue
            units = order.amount / order.entry_price
            unrealized = (mark.price - order.entry_price) * units if order.kind == "long" else (order.entry_price - mark.price) * units
            equity = max(0.0, order.amount + unrealized)
            unrealized = equity - order.amount
            open_equity += equity
            marks.append(
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "mark_price": mark.price,
                    "mark_time_ms": mark.timestamp_ms,
                    "unrealized_pnl": unrealized,
                    "equity": equity,
                }
            )
        return {
            "balance": state.get("balance", {}),
            "cash_usdt": cash,
            "open_order_count": len(state.get("open_orders", [])),
            "closed_trade_count": len(state.get("closed_trades", [])),
            "open_equity_usdt": open_equity,
            "total_equity_usdt": cash + open_equity,
            "marks": marks,
            "state": state,
        }

    def _mark_price(self, symbol: str, interval: str, as_of_ms: int | None) -> MarkPrice | None:
        candle = self.cache.latest_candle(symbol=symbol, interval=interval, as_of_ms=as_of_ms)
        if candle is None:
            return None
        return MarkPrice(price=candle.close, timestamp_ms=candle.close_time)

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def _order_to_record(order: Order) -> dict[str, Any]:
    return {
        "id": order.id,
        "kind": order.kind,
        "symbol": order.symbol,
        "price": order.entry_price,
        "stop_loss": order.stop_loss,
        "take_profit": order.take_profit,
        "amount": order.amount,
        "opened_at_ms": order.opened_at_ms,
    }


def _order_from_record(data: dict[str, Any]) -> Order:
    return Order.from_values(
        kind=data["kind"],
        symbol=data["symbol"],
        entry_price=float(data["price"]),
        stop_loss=float(data["stop_loss"]),
        take_profit=float(data["take_profit"]),
        amount=float(data["amount"]),
        opened_at=int(data["opened_at_ms"]),
        id=data.get("id"),
    )


def _order_id(opened_at: str | int | float) -> str:
    opened_at_ms = parse_time_ms(opened_at)
    return f"sim-{opened_at_ms}-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
