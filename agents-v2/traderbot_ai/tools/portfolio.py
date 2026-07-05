from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from agents import function_tool

from traderbot_ai.config import load_settings
from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs
from traderbot_ai.simulator.execution import ExecutionEngine, Order


PORTFOLIO_PATH = DATA_DIR / "portfolio.json"


def _default_portfolio() -> dict:
    settings = load_settings()
    return {"balance": {"USDT": settings.paper_balance_usdt}, "positions": [], "orders": []}


def _load_portfolio() -> dict:
    ensure_runtime_dirs()
    if not PORTFOLIO_PATH.exists():
        PORTFOLIO_PATH.write_text(json.dumps(_default_portfolio(), indent=2), encoding="utf-8")
    return json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))


def _save_portfolio(data: dict) -> None:
    ensure_runtime_dirs()
    PORTFOLIO_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_paper_order_request(
    final_decision: str,
    symbol: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    balance: float,
) -> Order:
    checked_order = Order.from_values(
        kind=final_decision,
        symbol=symbol,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        amount=amount,
        opened_at=datetime.now(timezone.utc).isoformat(),
    )
    ExecutionEngine().validate_order(checked_order)
    if amount > balance:
        raise ValueError(f"amount {amount} exceeds paper balance {balance}")
    max_position = balance * 0.30
    if amount > max_position:
        raise ValueError(f"amount {amount} exceeds max paper position {max_position}")
    units = checked_order.amount / checked_order.entry_price
    stop_distance = abs(checked_order.entry_price - checked_order.stop_loss)
    estimated_loss = units * stop_distance
    max_loss = balance * 0.02
    if estimated_loss > max_loss:
        raise ValueError(f"estimated loss {estimated_loss} exceeds max paper loss {max_loss}")
    return checked_order


@function_tool
def get_current_balance() -> dict:
    """Get the local paper-trading balance and open positions."""
    data = _load_portfolio()
    return {"balance": data.get("balance", {}), "positions": data.get("positions", []), "orders": data.get("orders", [])}


@function_tool
def reset_paper_portfolio(balance_usdt: float = 1000.0) -> dict:
    """Reset the local paper-trading portfolio."""
    data = {"balance": {"USDT": balance_usdt}, "positions": [], "orders": []}
    _save_portfolio(data)
    return data


@function_tool
def paper_place_order(
    final_decision: Literal["long", "short"],
    symbol: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    validation_ok: bool,
) -> dict:
    """Place a paper order only after risk validation has passed."""
    if not validation_ok:
        return {"ok": False, "error": "validation_ok must be true before placing a paper order"}

    data = _load_portfolio()
    balance = data.setdefault("balance", {}).setdefault("USDT", 0.0)
    try:
        validate_paper_order_request(final_decision, symbol, price, stop_loss, take_profit, amount, balance)
    except Exception as error:
        return {"ok": False, "error": str(error)}

    order = {
        "id": f"paper-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "kind": final_decision,
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "amount": amount,
        "status": "open",
    }
    data["balance"]["USDT"] = balance - amount
    data.setdefault("orders", []).append(order)
    _save_portfolio(data)
    return {"ok": True, "order": order, "balance": data["balance"]}
