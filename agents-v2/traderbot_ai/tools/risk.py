from __future__ import annotations

from typing import Literal

from agents import function_tool


def _validate_order_impl(
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
) -> dict:
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

    max_position = balance_usdt * max(0.0, max_position_fraction)
    if max_position > 0 and amount > max_position:
        errors.append(f"amount {amount} exceeds max position {max_position}")

    stop_distance = 0.0
    reward_distance = 0.0
    if final_decision == "long":
        if not (stop_loss < price < take_profit):
            errors.append("long requires stop_loss < price < take_profit")
        stop_distance = max(0.0, price - stop_loss)
        reward_distance = max(0.0, take_profit - price)
    elif final_decision == "short":
        if not (take_profit < price < stop_loss):
            errors.append("short requires take_profit < price < stop_loss")
        stop_distance = max(0.0, stop_loss - price)
        reward_distance = max(0.0, price - take_profit)
    else:
        errors.append(f"unsupported final_decision: {final_decision}")

    if price > 0 and final_decision in ("long", "short"):
        stop_distance_pct = stop_distance / price
        reward_distance_pct = reward_distance / price
        if stop_distance_pct < min_stop_distance_pct:
            errors.append(f"stop distance {stop_distance_pct} is below minimum {min_stop_distance_pct}")
        if stop_distance_pct > max_stop_distance_pct:
            errors.append(f"stop distance {stop_distance_pct} exceeds maximum {max_stop_distance_pct}")
        reward_risk = (reward_distance / stop_distance) if stop_distance > 0 else 0.0
        if reward_risk < min_reward_risk:
            errors.append(f"reward:risk {reward_risk} is below minimum {min_reward_risk}")
        required_target = (
            4 * max(0.0, round_trip_fee_fraction)
            + max(0.0, expected_funding_fraction)
            + max(0.0, expected_slippage_fraction)
        )
        if required_target > 0 and reward_distance_pct < required_target:
            errors.append(f"target distance {reward_distance_pct} is below fee/funding/slippage gate {required_target}")

    units = amount / price if price > 0 else 0.0
    estimated_loss = units * stop_distance
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
) -> dict:
    """Validate futures order geometry, stop bounds, reward:risk, fee gate, and risk sizing."""
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


def _calculate_position_size_impl(
    balance_usdt: float,
    price: float,
    stop_loss: float,
    risk_fraction: float = 0.0075,
    daily_risk_budget_remaining: float | None = None,
    size_pct_of_daily_budget: float = 35.0,
    max_notional_abs: float | None = None,
    liquidity_cap: float | None = None,
    max_position_fraction: float = 1.0,
) -> dict:
    if balance_usdt <= 0 or price <= 0:
        return {"amount": 0.0, "error": "balance_usdt and price must be positive"}
    stop_distance = abs(price - stop_loss)
    if stop_distance <= 0:
        return {"amount": 0.0, "error": "stop_loss must differ from price"}

    risk_budget = balance_usdt * max(0.0, min(risk_fraction, 1.0))
    if daily_risk_budget_remaining is not None:
        daily_size_pct = max(15.0, min(size_pct_of_daily_budget, 50.0))
        daily_share = max(0.0, daily_risk_budget_remaining) * daily_size_pct / 100.0
        risk_budget = min(risk_budget, daily_share)
    units = risk_budget / stop_distance
    risk_based_amount = units * price
    limits = [risk_based_amount]
    if max_notional_abs is not None and max_notional_abs > 0:
        limits.append(max_notional_abs)
    if liquidity_cap is not None and liquidity_cap > 0:
        limits.append(liquidity_cap)
    if max_position_fraction > 0:
        limits.append(balance_usdt * max_position_fraction)
    amount = min(limits)
    return {
        "amount": amount,
        "risk_budget": risk_budget,
        "risk_based_amount": risk_based_amount,
        "max_notional_abs": max_notional_abs,
        "liquidity_cap": liquidity_cap,
        "max_position_amount": balance_usdt * max_position_fraction if max_position_fraction > 0 else None,
    }


@function_tool
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
) -> dict:
    """Calculate a USDT position size from stop risk, daily risk share, and optional notional/liquidity caps."""
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
