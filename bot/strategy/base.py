"""Strategy interface: on_bar(MarketState) -> list[Intent].

A strategy owns nothing global: it reads market state, emits intents; the risk
manager is the only component that turns intents into orders. Per-strategy
mode live|paper|off comes from config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class SymbolSeries:
    """Hourly bar history for one symbol, ascending; positional 1h grid
    (mirrors the research grid: pct_change(6) = 6 bars back)."""
    bar_ms: Sequence[int]      # bar OPEN times
    close: Sequence[float]
    oi: Sequence[float | None]

    def to_pandas(self) -> tuple[pd.Series, pd.Series]:
        idx = pd.to_datetime(list(self.bar_ms), unit="ms", utc=True)
        return (pd.Series(list(self.close), index=idx, dtype=float),
                pd.Series([float("nan") if v is None else float(v) for v in self.oi],
                          index=idx, dtype=float))


class MarketState(Protocol):
    """What a strategy may see at a bar close (fed by feed.snapshot)."""
    bar_open_ms: int           # open time of the just-completed bar
    bar_close_ms: int          # = bar_open_ms + 3600_000 (decision time)
    equity_usd: float

    def symbols(self) -> list[str]: ...
    def series(self, symbol: str, n_bars: int) -> SymbolSeries | None: ...
    def liquidity_ok(self, symbol: str) -> bool: ...


@dataclass(frozen=True)
class EntryIntent:
    strategy: str
    symbol: str                # research pair name (e.g. PEPEUSDT)
    side: str                  # "Buy" | "Sell"
    limit_price: float
    weight: float              # slot weight (overlay), cap enforced by risk
    ttl_s: int                 # self-cancel deadline for the maker limit
    stop_pct: float            # server-side disaster stop distance from entry
    exit_at_ms: int            # time-based exit (market close of position)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExitIntent:
    strategy: str
    position_id: int
    reason: str


Intent = EntryIntent | ExitIntent


class Strategy(Protocol):
    name: str
    mode: str                  # off | paper | live

    def on_bar(self, market: MarketState) -> list[Intent]: ...
