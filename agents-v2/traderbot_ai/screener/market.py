from __future__ import annotations

from datetime import datetime, timezone

from traderbot_ai.paths import DATA_DIR


INTERVAL_MS = {
    "1s": 1000,
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

DEFAULT_PROVIDER = "binance"
DEFAULT_MARKET = "spot"
DEFAULT_CACHE_PATH = DATA_DIR / "market_cache.sqlite3"


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.upper().replace("/", "").replace("-", "").strip()
    if not cleaned:
        raise ValueError("symbol is empty")
    if cleaned.endswith("USDT"):
        return cleaned
    return f"{cleaned}USDT"


def parse_time_ms(value: str | int | float | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number > 10_000_000_000 else number * 1000

    text = str(value).strip()
    if text.isdigit():
        number = int(text)
        return number if number > 10_000_000_000 else number * 1000
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)
