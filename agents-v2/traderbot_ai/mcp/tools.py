from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from traderbot_ai.decision.serialization import jsonable
from traderbot_ai.tools.analysis import compute_indicators_impl, run_analysis_code_impl
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
        preflight = replay_helpers.deterministic_candle_deep_dive_request(symbol, interval, as_of, limit, start_time, end_time)
        if preflight is not None and preflight.get("ok") is not True:
            preflight["tool"] = "get_candles"
            return preflight
        if preflight is not None:
            symbol = str(preflight["symbol"])
            interval = str(preflight["interval"])
            limit = int(preflight["limit"])
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


def compute_indicators(
    symbol: str,
    interval: str = "1h",
    as_of: str | int | float | None = None,
    limit: int = 170,
    indicators: str = "strategy",
    source: Literal["auto", "cache", "live"] = "auto",
    tail: int = 5,
    render_chart: bool = False,
) -> dict[str, Any]:
    """Load guarded candles and compute technical indicators plus optional chart artifact."""
    return compute_indicators_impl(
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        limit=limit,
        indicators=indicators,
        source=source,
        tail=tail,
        render_chart=render_chart,
    )


def run_analysis_code(
    code: str,
    symbol: str,
    interval: str = "1h",
    as_of: str | int | float | None = None,
    limit: int = 170,
    source: Literal["auto", "cache", "live"] = "auto",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run scratch Python on guarded candle data loaded by the tool."""
    return run_analysis_code_impl(
        code=code,
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        limit=limit,
        source=source,
        timeout_seconds=timeout_seconds,
    )


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
    entry_policy_info = _apply_limit_retest_entry_policy(inputs)
    result = _agent_tool_response(place_order_impl(**inputs))
    if risk_validation is not None:
        result["risk_validation"] = risk_validation
    if entry_policy_info is not None:
        result["entry_policy"] = entry_policy_info
    _audit_write("place_order", inputs, result)
    return result


def _apply_limit_retest_entry_policy(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Runner-owned retest entry: rewrite an accepted deterministic Market entry into a pending limit with a TTL.

    The agent's plan stays intact (same qty, stop, take-profit); only the entry waits for a pullback of
    pullback*stop_distance from the reference price and is skipped when price never pulls back before the TTL.
    """
    if os.getenv("TRADERBOT_SCREENER_MODE") != "deterministic":
        return None
    if os.getenv("TRADERBOT_ENTRY_POLICY") != "limit_retest":
        return None
    if str(inputs.get("category") or "").lower() != "linear":
        return None
    if str(inputs.get("orderType") or "").lower() != "market":
        return None
    if inputs.get("reduceOnly"):
        return None
    side = str(inputs.get("side") or "").lower()
    if side not in {"buy", "sell"}:
        return None
    stop_loss = inputs.get("stopLoss")
    entry_ref = _entry_price_for_order(inputs)
    if stop_loss is None or entry_ref is None or entry_ref <= 0:
        return None
    from traderbot_ai.tools.market import parse_time_ms

    as_of_ms = parse_time_ms(inputs.get("as_of"))
    if as_of_ms is None:
        return None
    stop_pct = abs(entry_ref - float(stop_loss)) / entry_ref
    if stop_pct <= 0:
        return None
    pullback = float(os.getenv("TRADERBOT_RETEST_PULLBACK") or 0.4)
    ttl_min = int(os.getenv("TRADERBOT_RETEST_TTL_MIN") or 120)
    offset = pullback * stop_pct
    limit_price = entry_ref * (1.0 - offset) if side == "buy" else entry_ref * (1.0 + offset)
    expires_at_ms = as_of_ms + ttl_min * 60_000
    inputs["orderType"] = "Limit"
    inputs["price"] = limit_price
    inputs["timeInForce"] = "GTC"
    inputs["expiresAtMs"] = expires_at_ms
    inputs["entryPolicy"] = "limit_retest"
    inputs["entryRefPrice"] = entry_ref
    return {
        "policy": "limit_retest",
        "pullback": pullback,
        "ttl_min": ttl_min,
        "entry_ref_price": entry_ref,
        "limit_price": limit_price,
        "expires_at_ms": expires_at_ms,
        "note": "runner entry policy: order placed as a pending limit; the position opens only if price pulls back to the limit before expiry, otherwise the order expires and no position is opened",
    }


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
    if allow_error is not None:
        result = dict(allow_error)
        result["tool"] = "place_order"
        result["error"] = "place_order is only available for runner-provided deterministic candidates"
        return result
    drift_error = replay_helpers.deterministic_entry_drift_error(
        str(inputs.get("symbol") or ""),
        candidate_side,
        _entry_price_for_order(inputs),
    )
    if drift_error is not None:
        result = dict(drift_error)
        result["tool"] = "place_order"
        return result
    stop_noise_error = replay_helpers.deterministic_stop_noise_error(
        str(inputs.get("symbol") or ""),
        candidate_side,
        _entry_price_for_order(inputs),
        inputs.get("stopLoss"),
    )
    if stop_noise_error is not None:
        result = dict(stop_noise_error)
        result["tool"] = "place_order"
        return result
    return None


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
    min_stop_distance_pct = 0.01
    noise_floor = _deterministic_noise_floor_stop_pct(str(inputs.get("symbol") or ""), inputs.get("as_of"))
    if noise_floor is not None:
        min_stop_distance_pct = max(min_stop_distance_pct, noise_floor)
    result = _validate_order_impl(
        final_decision=final_decision,
        price=entry_price,
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        amount=amount,
        balance_usdt=balance_usdt,
        max_position_fraction=0.20,
        max_loss_fraction=0.0075,
        min_stop_distance_pct=min_stop_distance_pct,
        round_trip_fee_fraction=fee_rate,
    )
    if noise_floor is not None and isinstance(result, dict):
        result["noise_floor_stop_pct"] = noise_floor
    return result


def _deterministic_noise_floor_stop_pct(symbol: str, as_of: str | int | float | None) -> float | None:
    """Volatility-aware minimum stop distance; a stop inside typical bar noise is invalid."""
    if not _deterministic_screener_mode():
        return None
    try:
        from traderbot_ai.screener.config import ScreenerConfig
        from traderbot_ai.screener.data import load_closed, validate_frame
        from traderbot_ai.screener.indicators import atr_wilder, median_range_pct
        from traderbot_ai.screener.market import INTERVAL_MS, normalize_symbol, parse_time_ms
        from traderbot_ai.screener.plan import noise_floor_stop_pct
        from traderbot_ai.simulator.market_cache import DEFAULT_CACHE_PATH

        cfg = ScreenerConfig()
        as_of_ms = parse_time_ms(as_of)
        if as_of_ms is None:
            return None
        bars = max(cfg.atr_period + 1, cfg.stop_noise_median_window) + 10
        cache_path = os.getenv("TRADERBOT_MARKET_CACHE_PATH") or DEFAULT_CACHE_PATH
        frame = load_closed(cache_path, normalize_symbol(symbol), INTERVAL_MS["1h"], as_of_ms, bars)
        if validate_frame(frame, as_of_ms, bars) is not None:
            return None
        close = frame.close[-1]
        atr_value = atr_wilder(frame.high, frame.low, frame.close, cfg.atr_period)[-1]
        median_range = median_range_pct(frame.high, frame.low, frame.close, cfg.stop_noise_median_window)
        return noise_floor_stop_pct(close, atr_value, median_range, cfg)
    except Exception:
        return None


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
