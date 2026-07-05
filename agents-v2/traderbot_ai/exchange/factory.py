from __future__ import annotations

import os
from pathlib import Path

from traderbot_ai.exchange.interface import ExchangeBackend
from traderbot_ai.exchange.simulated import DEFAULT_EXCHANGE_EVENTS_PATH, DEFAULT_EXCHANGE_STATE_PATH, SimulatedExchange
from traderbot_ai.simulator.market_cache import LocalMarketCache


EXCHANGE_BACKEND_ENV = "TRADERBOT_EXCHANGE_BACKEND"


def create_exchange_backend(
    backend: str | None = None,
    *,
    state_path: str | Path | None = None,
    events_path: str | Path | None = None,
    cache_path: str | Path | None = None,
) -> ExchangeBackend:
    backend_name = (backend or os.getenv(EXCHANGE_BACKEND_ENV) or "simulated").strip().lower()
    if backend_name != "simulated":
        raise NotImplementedError(f"exchange backend is not implemented yet: {backend_name}")
    resolved_cache_path = cache_path or os.getenv("TRADERBOT_MARKET_CACHE_PATH")
    return SimulatedExchange(
        path=state_path or os.getenv("TRADERBOT_EXCHANGE_STATE_PATH") or DEFAULT_EXCHANGE_STATE_PATH,
        events_path=events_path or os.getenv("TRADERBOT_EXCHANGE_EVENTS_PATH") or DEFAULT_EXCHANGE_EVENTS_PATH,
        cache=LocalMarketCache(resolved_cache_path) if resolved_cache_path else None,
    )
