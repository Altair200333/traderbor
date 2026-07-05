from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from agents import function_tool

from traderbot_ai.paths import ARTIFACTS_DIR, display_agents_path, ensure_runtime_dirs


BINANCE_BASE_URL = "https://api.binance.com"
DEFAULT_TIMEOUT = 15
MAX_ARTIFACT_CANDLES = 5000
BINANCE_LIMIT = 1000
BINANCE_AGG_TRADE_LIMIT = 1000
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


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.upper().replace("/", "").replace("-", "").strip()
    if not cleaned:
        raise ValueError("symbol is empty")
    if cleaned.endswith("USDT"):
        return cleaned
    return f"{cleaned}USDT"


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _error(message: str, **data) -> dict:
    return {"ok": False, "error": message, **data}


def _error_symbol(symbol: str) -> str:
    try:
        return normalize_symbol(symbol)
    except Exception:
        return str(symbol)


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


def candle_from_binance_row(row: list[Any]) -> dict[str, Any]:
    open_time = int(row[0])
    close_time = int(row[6])
    return {
        "timestamp": open_time,
        "time": datetime.fromtimestamp(open_time / 1000.0, tz=timezone.utc).isoformat(),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "close_timestamp": close_time,
        "close_time": datetime.fromtimestamp(close_time / 1000.0, tz=timezone.utc).isoformat(),
        "quote_volume": float(row[7]),
        "trade_count": int(row[8]),
        "taker_buy_base_volume": float(row[9]),
        "taker_buy_quote_volume": float(row[10]),
    }


def compact_candle(candle: dict[str, Any]) -> dict[str, float]:
    return {
        "t": float(candle["timestamp"]),
        "o": float(candle["open"]),
        "h": float(candle["high"]),
        "l": float(candle["low"]),
        "c": float(candle["close"]),
        "v": float(candle["volume"]),
    }


def fetch_candles(
    symbol: str,
    interval: str = "15m",
    limit: int = 120,
    start_time: str | int | float | None = None,
    end_time: str | int | float | None = None,
) -> list[dict[str, float]]:
    detailed = fetch_candle_records(
        symbol=symbol,
        interval=interval,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
    )
    return [compact_candle(item) for item in detailed]


def fetch_candle_records(
    symbol: str,
    interval: str = "15m",
    limit: int = 120,
    start_time: str | int | float | None = None,
    end_time: str | int | float | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_symbol(symbol)
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported bin size: {interval}")

    clamped_limit = max(1, min(limit, MAX_ARTIFACT_CANDLES))
    start_ms = parse_time_ms(start_time)
    end_ms = parse_time_ms(end_time)
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("start_time must be before end_time")

    candles: list[dict[str, Any]] = []
    next_start = start_ms
    while len(candles) < clamped_limit:
        request_limit = min(BINANCE_LIMIT, clamped_limit - len(candles))
        params: dict[str, Any] = {"symbol": normalized, "interval": interval, "limit": request_limit}
        if next_start is not None:
            params["startTime"] = next_start
        if end_ms is not None:
            params["endTime"] = end_ms

        response = requests.get(f"{BINANCE_BASE_URL}/api/v3/klines", params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break

        batch = [candle_from_binance_row(row) for row in rows]
        candles.extend(batch)
        if start_ms is None or len(rows) < request_limit:
            break

        next_start = int(batch[-1]["timestamp"]) + INTERVAL_MS[interval]
        if end_ms is not None and next_start >= end_ms:
            break

    return candles[:clamped_limit]


def agg_trade_from_binance_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_trade_id": int(row["a"]),
        "price": float(row["p"]),
        "quantity": float(row["q"]),
        "first_trade_id": int(row["f"]),
        "last_trade_id": int(row["l"]),
        "timestamp": int(row["T"]),
        "is_buyer_maker": bool(row["m"]),
        "is_best_match": bool(row["M"]),
    }


def fetch_agg_trade_records(
    symbol: str,
    start_time: str | int | float,
    end_time: str | int | float,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_symbol(symbol)
    start_ms = parse_time_ms(start_time)
    end_ms = parse_time_ms(end_time)
    if start_ms is None or end_ms is None:
        raise ValueError("start_time and end_time are required")
    if start_ms >= end_ms:
        raise ValueError("start_time must be before end_time")

    max_rows = max(1, int(limit)) if limit is not None else None
    trades: list[dict[str, Any]] = []
    next_from_id: int | None = None
    while max_rows is None or len(trades) < max_rows:
        request_limit = BINANCE_AGG_TRADE_LIMIT if max_rows is None else min(BINANCE_AGG_TRADE_LIMIT, max_rows - len(trades))
        params: dict[str, Any] = {
            "symbol": normalized,
            "limit": request_limit,
        }
        if next_from_id is None:
            params["startTime"] = start_ms
            params["endTime"] = end_ms - 1
        else:
            params["fromId"] = next_from_id
        response = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/aggTrades",
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break

        batch = [agg_trade_from_binance_row(row) for row in rows]
        in_range = [trade for trade in batch if start_ms <= int(trade["timestamp"]) < end_ms]
        if in_range:
            trades.extend(in_range)

        last = batch[-1]
        if int(last["timestamp"]) >= end_ms:
            break
        if len(rows) < request_limit:
            break
        next_from_id = int(last["aggregate_trade_id"]) + 1

    return trades


def _summarize_candles(candles: list[dict[str, float]]) -> dict[str, Any]:
    if not candles:
        return {"count": 0}
    first = candles[0]
    last = candles[-1]
    highs = [c["h"] for c in candles]
    lows = [c["l"] for c in candles]
    volumes = [c["v"] for c in candles]
    change = last["c"] - first["o"]
    change_pct = (change / first["o"] * 100.0) if first["o"] else 0.0
    return {
        "count": len(candles),
        "first_time": datetime.fromtimestamp(first["t"] / 1000.0, tz=timezone.utc).isoformat(),
        "last_time": datetime.fromtimestamp(last["t"] / 1000.0, tz=timezone.utc).isoformat(),
        "open": first["o"],
        "close": last["c"],
        "high": max(highs),
        "low": min(lows),
        "change": change,
        "change_pct": change_pct,
        "volume_sum": sum(volumes),
    }


def _summarize_records(candles: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_candles([compact_candle(item) for item in candles])


def save_market_artifact_impl(
    symbol: str,
    bin_size: str = "5m",
    start_time: str | int | float | None = None,
    end_time: str | int | float | None = None,
    limit: int = 500,
) -> dict:
    try:
        ensure_runtime_dirs()
        normalized = normalize_symbol(symbol)
        records = fetch_candle_records(
            symbol=normalized,
            interval=bin_size,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )
        candles = [compact_candle(item) for item in records]
        artifact_id = uuid.uuid4().hex
        path = ARTIFACTS_DIR / f"market-{normalized}-{bin_size}-{artifact_id}.json"
        payload = {
            "schema_version": "market-candles/v1",
            "provider": "binance",
            "market": "spot",
            "symbol": normalized,
            "base_asset": normalized[:-4] if normalized.endswith("USDT") else None,
            "quote_asset": "USDT" if normalized.endswith("USDT") else None,
            "timeframe": bin_size,
            "exchange_interval": bin_size,
            "time_unit": "ms",
            "volume_unit": "base_asset",
            "request": {
                "start_time": start_time,
                "start_time_ms": parse_time_ms(start_time),
                "end_time": end_time,
                "end_time_ms": parse_time_ms(end_time),
                "limit": limit,
                "used_limit": len(candles),
            },
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "summary": _summarize_candles(candles),
            "candles": candles,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return _ok(
            path=display_agents_path(path),
            absolute_path=str(path),
            symbol=normalized,
            bin_size=bin_size,
            candle_count=len(candles),
            summary=payload["summary"],
        )
    except Exception as error:
        return _error(str(error), symbol=_error_symbol(symbol), bin_size=bin_size)


@function_tool
def get_candles(symbol: str, interval: str = "15m", limit: int = 120) -> dict:
    """Get recent Binance candles. Use for quick market facts."""
    try:
        candles = fetch_candles(symbol=symbol, interval=interval, limit=limit)
        return _ok(
            symbol=normalize_symbol(symbol),
            interval=interval,
            summary=_summarize_candles(candles),
            candles=candles,
        )
    except Exception as error:
        return _error(str(error), symbol=_error_symbol(symbol), interval=interval)


@function_tool
def save_market_artifact(
    symbol: str,
    bin_size: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 500,
) -> dict:
    """Save candles to agents-v2/artifacts. Args: symbol, bin_size, start, end."""
    return save_market_artifact_impl(
        symbol=symbol,
        bin_size=bin_size,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


@function_tool
def get_current_price(symbol: str) -> dict:
    """Get current Binance spot price."""
    try:
        normalized = normalize_symbol(symbol)
        response = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/ticker/price",
            params={"symbol": normalized},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return _ok(symbol=normalized, price=float(data["price"]))
    except Exception as error:
        return _error(str(error), symbol=_error_symbol(symbol))


@function_tool
def get_order_book(symbol: str, limit: int = 20) -> dict:
    """Get shallow Binance order book."""
    clamped_limit = max(5, min(limit, 100))
    try:
        normalized = normalize_symbol(symbol)
        response = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/depth",
            params={"symbol": normalized, "limit": clamped_limit},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return _ok(
            symbol=normalized,
            last_update_id=data.get("lastUpdateId"),
            bids=[[float(price), float(size)] for price, size in data.get("bids", [])],
            asks=[[float(price), float(size)] for price, size in data.get("asks", [])],
        )
    except Exception as error:
        return _error(str(error), symbol=_error_symbol(symbol))
