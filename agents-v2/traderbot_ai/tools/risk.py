from __future__ import annotations

from typing import Literal

from agents import function_tool


@function_tool
def validate_order(
    final_decision: Literal["long", "short", "hold"],
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    balance_usdt: float,
    max_position_fraction: float = 0.30,
    max_loss_fraction: float = 0.02,
) -> dict:
    """Validate basic futures order geometry and risk sizing."""
    errors: list[str] = []
    warnings: list[str] = []

    if final_decision == "hold":
        return {
            "ok": True,
            "errors": [],
            "warnings": ["No order requested."],
            "estimated_loss_usdt": 0.0,
            "max_loss_fraction": 0.0,
        }

    if price <= 0:
        errors.append("price must be positive")
    if amount < 0:
        errors.append("amount must be non-negative")
    if balance_usdt <= 0:
        errors.append("balance_usdt must be positive")

    max_position = balance_usdt * max_position_fraction
    if amount > max_position:
        errors.append(f"amount {amount} exceeds max position {max_position}")

    if final_decision == "long":
        if not (stop_loss < price < take_profit):
            errors.append("long requires stop_loss < price < take_profit")
        per_unit_loss = max(0.0, price - stop_loss)
    elif final_decision == "short":
        if not (take_profit < price < stop_loss):
            errors.append("short requires take_profit < price < stop_loss")
        per_unit_loss = max(0.0, stop_loss - price)
    else:
        errors.append(f"unsupported final_decision: {final_decision}")
        per_unit_loss = 0.0

    units = amount / price if price > 0 else 0.0
    estimated_loss = units * per_unit_loss
    allowed_loss = balance_usdt * max_loss_fraction
    if estimated_loss > allowed_loss:
        errors.append(f"estimated loss {estimated_loss} exceeds allowed loss {allowed_loss}")
    if estimated_loss == 0 and final_decision != "hold":
        warnings.append("estimated loss is zero; check stop_loss and price")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "estimated_loss_usdt": estimated_loss,
        "max_loss_fraction": (estimated_loss / balance_usdt) if balance_usdt > 0 else 0.0,
    }


@function_tool
def calculate_position_size(
    balance_usdt: float,
    price: float,
    stop_loss: float,
    risk_fraction: float = 0.01,
    max_position_fraction: float = 0.30,
) -> dict:
    """Calculate a USDT position size from risk budget and stop distance."""
    if balance_usdt <= 0 or price <= 0:
        return {"amount": 0.0, "error": "balance_usdt and price must be positive"}
    stop_distance = abs(price - stop_loss)
    if stop_distance <= 0:
        return {"amount": 0.0, "error": "stop_loss must differ from price"}

    risk_budget = balance_usdt * max(0.0, min(risk_fraction, 1.0))
    units = risk_budget / stop_distance
    risk_based_amount = units * price
    max_position_amount = balance_usdt * max(0.0, min(max_position_fraction, 1.0))
    amount = min(risk_based_amount, max_position_amount)
    return {
        "amount": amount,
        "risk_budget": risk_budget,
        "risk_based_amount": risk_based_amount,
        "max_position_amount": max_position_amount,
    }
