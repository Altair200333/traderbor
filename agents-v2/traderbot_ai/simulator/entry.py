from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.tools.market import normalize_symbol, parse_time_ms


EntryPolicy = Literal["next_open"]


@dataclass(frozen=True)
class EntryFill:
    symbol: str
    interval: str
    policy: EntryPolicy
    requested_price: float | None
    fill_price: float
    opened_at_ms: int
    candle: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "policy": self.policy,
            "requested_price": self.requested_price,
            "fill_price": self.fill_price,
            "opened_at_ms": self.opened_at_ms,
            "candle": self.candle,
        }


def resolve_entry_fill(
    cache: LocalMarketCache,
    symbol: str,
    as_of: str | int | float,
    interval: str = "1m",
    requested_price: float | None = None,
    policy: EntryPolicy = "next_open",
    before: str | int | float | None = None,
) -> EntryFill:
    if policy != "next_open":
        raise ValueError(f"unsupported entry policy: {policy}")
    as_of_ms = parse_time_ms(as_of)
    if as_of_ms is None:
        raise ValueError("as_of is required")
    before_ms = parse_time_ms(before)
    if before is not None and before_ms is None:
        raise ValueError("before is invalid")
    if before_ms is not None and before_ms <= as_of_ms:
        raise ValueError("before must be after as_of")

    normalized_symbol = normalize_symbol(symbol)
    candle = cache.first_candle_at_or_after(
        symbol=normalized_symbol,
        interval=interval,
        start_ms=as_of_ms,
        before_ms=before_ms,
    )
    if candle is None:
        suffix = f" before {before_ms}" if before_ms is not None else ""
        raise ValueError(f"no cached entry candle found for {normalized_symbol} {interval} at or after {as_of_ms}{suffix}")
    return _next_open_fill(candle, requested_price, policy)


def _next_open_fill(candle: Candle, requested_price: float | None, policy: EntryPolicy) -> EntryFill:
    return EntryFill(
        symbol=candle.symbol,
        interval=candle.interval,
        policy=policy,
        requested_price=float(requested_price) if requested_price is not None else None,
        fill_price=candle.open,
        opened_at_ms=candle.open_time,
        candle=candle.detailed(),
    )
