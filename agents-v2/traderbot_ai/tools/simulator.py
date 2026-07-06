from __future__ import annotations

import os
from typing import Any

from agents import function_tool

from traderbot_ai.simulator.clock import (
    clear_simulation_clock_state,
    guarded_simulation_as_of,
    set_simulation_clock_state,
)
from traderbot_ai.simulator.execution import ExecutionEngine, Order
from traderbot_ai.simulator.market_cache import DEFAULT_CACHE_PATH, LocalMarketCache, candle_freshness
from traderbot_ai.simulator.portfolio import SimulatedPortfolio
from traderbot_ai.tools.market import _summarize_candles, normalize_symbol, parse_time_ms
from traderbot_ai.tools.replay_helpers import deterministic_candle_deep_dive_request


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _error(message: str, **data) -> dict:
    return {"ok": False, "error": message, **data}


def _cache() -> LocalMarketCache:
    return LocalMarketCache(os.getenv("TRADERBOT_MARKET_CACHE_PATH") or DEFAULT_CACHE_PATH)


def set_simulation_clock_impl(as_of: str | int | float) -> dict[str, Any]:
    try:
        return _ok(as_of_ms=set_simulation_clock_state(as_of))
    except Exception as error:
        return _error(str(error))


def clear_simulation_clock_impl() -> dict[str, Any]:
    try:
        clear_simulation_clock_state()
        return _ok(cleared=True)
    except Exception as error:
        return _error(str(error))


def preload_market_cache_impl(
    symbols: str,
    intervals: str = "1m",
    start_time: str | int | float | None = None,
    end_time: str | int | float | None = None,
    max_candles_per_pair: int | None = None,
    include_agg_trades: bool = False,
    max_agg_trades_per_symbol: int | None = None,
) -> dict[str, Any]:
    if start_time is None or end_time is None:
        return _error("start_time and end_time are required")
    cache = _cache()
    results = []
    for symbol in _split_csv(symbols):
        for interval in _split_csv(intervals):
            try:
                results.append(
                    {
                        "ok": True,
                        **cache.preload_binance(
                            symbol=symbol,
                            interval=interval,
                            start_time=start_time,
                            end_time=end_time,
                            max_candles=max_candles_per_pair,
                        ),
                    }
                )
            except Exception as error:
                results.append({"ok": False, "symbol": symbol, "interval": interval, "error": str(error)})
        if include_agg_trades:
            try:
                results.append(
                    {
                        "ok": True,
                        "kind": "agg_trades",
                        **cache.preload_binance_agg_trades(
                            symbol=symbol,
                            start_time=start_time,
                            end_time=end_time,
                            max_trades=max_agg_trades_per_symbol,
                        ),
                    }
                )
            except Exception as error:
                results.append({"ok": False, "symbol": symbol, "kind": "agg_trades", "error": str(error)})
    return _ok(results=results, status=cache.status())


def get_cached_candles_impl(
    symbol: str,
    interval: str = "1m",
    as_of: str | int | float | None = None,
    lookback: int = 120,
    start_time: str | int | float | None = None,
    end_time: str | int | float | None = None,
) -> dict[str, Any]:
    try:
        cache = _cache()
        as_of_ms = guarded_simulation_as_of(as_of)
        start_ms = parse_time_ms(start_time)
        end_ms = parse_time_ms(end_time)
        candles = cache.get_candles(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            as_of_ms=as_of_ms,
            limit=max(1, min(int(lookback), 5000)),
        )
        compact = [candle.compact() for candle in candles]
        freshness = candle_freshness(candles[-1] if candles else None, interval, as_of_ms)
        freshness_required = as_of_ms is not None and (end_ms is None or end_ms >= as_of_ms)
        if freshness_required and not freshness["fresh"]:
            return _error(
                "stale cached candle data",
                symbol=normalize_symbol(symbol),
                interval=interval,
                source="local_cache",
                as_of_ms=as_of_ms,
                closed_only=True,
                freshness=freshness,
                summary=_summarize_candles(compact),
                candles=compact,
            )
        return _ok(
            symbol=normalize_symbol(symbol),
            interval=interval,
            source="local_cache",
            as_of_ms=as_of_ms,
            closed_only=True,
            freshness=freshness,
            summary=_summarize_candles(compact),
            candles=compact,
        )
    except Exception as error:
        return _error(str(error), symbol=symbol, interval=interval)


def get_cached_price_impl(symbol: str, interval: str = "1m", as_of: str | int | float | None = None) -> dict[str, Any]:
    try:
        cache = _cache()
        as_of_ms = guarded_simulation_as_of(as_of)
        candle = cache.latest_candle(symbol=symbol, interval=interval, as_of_ms=as_of_ms)
        if candle is None:
            return _error("no cached candle found", symbol=normalize_symbol(symbol), interval=interval)
        freshness = candle_freshness(candle, interval, as_of_ms)
        if not freshness["fresh"]:
            return _error(
                "stale cached mark price",
                symbol=normalize_symbol(symbol),
                interval=interval,
                source="local_cache",
                freshness=freshness,
            )
        return _ok(
            symbol=candle.symbol,
            interval=interval,
            source="local_cache",
            price=candle.close,
            timestamp=candle.close_time,
            freshness=freshness,
            candle=candle.compact(),
        )
    except Exception as error:
        return _error(str(error), symbol=symbol, interval=interval)


def get_market_cache_status_impl() -> dict[str, Any]:
    try:
        return _ok(**_cache().status())
    except Exception as error:
        return _error(str(error))


def reset_simulation_impl(balance_usdt: float = 1000.0, as_of: str | int | float | None = None) -> dict[str, Any]:
    try:
        return _ok(**SimulatedPortfolio().reset(balance_usdt=balance_usdt, as_of=guarded_simulation_as_of(as_of)))
    except Exception as error:
        return _error(str(error))


def get_simulation_portfolio_impl(as_of: str | int | float | None = None, mark_interval: str = "1m") -> dict[str, Any]:
    try:
        return _ok(**SimulatedPortfolio().summary(as_of=guarded_simulation_as_of(as_of), mark_interval=mark_interval))
    except Exception as error:
        return _error(str(error))


def place_simulated_order_impl(
    final_decision: str,
    symbol: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    opened_at: str | int | float,
) -> dict[str, Any]:
    try:
        opened_at_ms = guarded_simulation_as_of(opened_at)
        return _ok(
            **SimulatedPortfolio().place_order(
                final_decision=final_decision,
                symbol=symbol,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                amount=amount,
                opened_at=opened_at_ms,
            )
        )
    except Exception as error:
        return _error(str(error), symbol=symbol, final_decision=final_decision)


def settle_simulation_impl(
    as_of: str | int | float,
    interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict[str, Any]:
    try:
        return _ok(**SimulatedPortfolio().settle(as_of=guarded_simulation_as_of(as_of), interval=interval, fee_rate=fee_rate))
    except Exception as error:
        return _error(str(error), as_of=as_of, interval=interval)


def simulate_order_exit_impl(
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
    try:
        guarded_scan_until = guarded_simulation_as_of(scan_until)
        guarded_opened_at = guarded_simulation_as_of(opened_at)
        order = Order.from_values(
            kind=final_decision,
            symbol=symbol,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            amount=amount,
            opened_at=guarded_opened_at,
        )
        result = ExecutionEngine(_cache()).resolve_order(
            order=order,
            interval=interval,
            scan_until=guarded_scan_until,
            fee_rate=fee_rate,
        )
        return _ok(**result.to_dict())
    except Exception as error:
        return _error(str(error), symbol=symbol, final_decision=final_decision)


@function_tool
def preload_market_cache(
    symbols: str,
    intervals: str = "1m",
    start_time: str | None = None,
    end_time: str | None = None,
    max_candles_per_pair: int | None = None,
    include_agg_trades: bool = False,
    max_agg_trades_per_symbol: int | None = None,
) -> dict:
    """Fetch Binance candles into local cache. Args are CSV symbols and intervals."""
    return preload_market_cache_impl(
        symbols,
        intervals,
        start_time,
        end_time,
        max_candles_per_pair,
        include_agg_trades,
        max_agg_trades_per_symbol,
    )


@function_tool
def get_market_cache_status() -> dict:
    """Show local market cache coverage by symbol and interval."""
    return get_market_cache_status_impl()


@function_tool
def set_simulation_clock(as_of: str) -> dict:
    """Set the active simulation as_of guard for cached market tools."""
    return set_simulation_clock_impl(as_of)


@function_tool
def reset_simulation(balance_usdt: float = 1000.0, as_of: str | None = None) -> dict:
    """Reset simulated portfolio cash, open orders, and closed trades."""
    return reset_simulation_impl(balance_usdt=balance_usdt, as_of=as_of)


@function_tool
def get_simulation_portfolio(as_of: str | None = None, mark_interval: str = "1m") -> dict:
    """Get simulated cash, open orders, closed trades, and marked equity."""
    return get_simulation_portfolio_impl(as_of=as_of, mark_interval=mark_interval)


@function_tool
def place_simulated_order(
    final_decision: str,
    symbol: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    opened_at: str,
) -> dict:
    """Place a validated long/short order in the simulated portfolio."""
    return place_simulated_order_impl(final_decision, symbol, price, stop_loss, take_profit, amount, opened_at)


@function_tool
def settle_simulation(as_of: str, interval: str = "1m", fee_rate: float = 0.0) -> dict:
    """Close simulated orders whose TP/SL fired by as_of."""
    return settle_simulation_impl(as_of=as_of, interval=interval, fee_rate=fee_rate)


@function_tool
def get_cached_candles(
    symbol: str,
    interval: str = "1m",
    as_of: str | None = None,
    lookback: int = 120,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    """Get closed candles from local cache only. Use as_of for backtests."""
    return get_cached_candles_impl(symbol, interval, as_of, lookback, start_time, end_time)


def get_candles_impl(
    symbol: str,
    interval: str = "1m",
    as_of: str | None = None,
    limit: int = 120,
) -> dict:
    """Get closed candles from local cache. Compatible replacement for market.get_candles."""
    preflight = deterministic_candle_deep_dive_request(symbol, interval, as_of, limit, None, None)
    if preflight is not None and preflight.get("ok") is not True:
        return preflight
    if preflight is not None:
        symbol = str(preflight["symbol"])
        interval = str(preflight["interval"])
        limit = int(preflight["limit"])
    return get_cached_candles_impl(symbol=symbol, interval=interval, as_of=as_of, lookback=limit)


@function_tool
def get_candles(
    symbol: str,
    interval: str = "1m",
    as_of: str | None = None,
    limit: int = 120,
) -> dict:
    """Get closed candles from local cache. Compatible replacement for market.get_candles."""
    return get_candles_impl(symbol=symbol, interval=interval, as_of=as_of, limit=limit)


@function_tool
def get_current_price(symbol: str, interval: str = "1m", as_of: str | None = None) -> dict:
    """Get the latest cached close price at or before as_of."""
    return get_cached_price_impl(symbol=symbol, interval=interval, as_of=as_of)


@function_tool
def simulate_order_exit(
    final_decision: str,
    symbol: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    amount: float,
    opened_at: str,
    scan_until: str,
    interval: str = "1m",
    fee_rate: float = 0.0,
) -> dict:
    """Simulate long/short TP/SL exit against cached candles."""
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


def _split_csv(value: str) -> list[str]:
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]
