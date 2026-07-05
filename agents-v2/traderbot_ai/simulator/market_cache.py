from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs
from traderbot_ai.tools.market import INTERVAL_MS, fetch_agg_trade_records, fetch_candle_records, normalize_symbol, parse_time_ms


DEFAULT_PROVIDER = "binance"
DEFAULT_MARKET = "spot"
DEFAULT_CACHE_PATH = DATA_DIR / "market_cache.sqlite3"


@dataclass(frozen=True)
class Candle:
    symbol: str
    interval: str
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    provider: str = DEFAULT_PROVIDER
    market: str = DEFAULT_MARKET
    quote_volume: float | None = None
    trade_count: int | None = None
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None

    @classmethod
    def from_record(
        cls,
        symbol: str,
        interval: str,
        record: dict[str, Any],
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> "Candle":
        return cls(
            provider=provider,
            market=market,
            symbol=normalize_symbol(symbol),
            interval=interval,
            open_time=int(record["timestamp"]),
            close_time=int(record["close_timestamp"]),
            open=float(record["open"]),
            high=float(record["high"]),
            low=float(record["low"]),
            close=float(record["close"]),
            volume=float(record["volume"]),
            quote_volume=_optional_float(record.get("quote_volume")),
            trade_count=_optional_int(record.get("trade_count")),
            taker_buy_base_volume=_optional_float(record.get("taker_buy_base_volume")),
            taker_buy_quote_volume=_optional_float(record.get("taker_buy_quote_volume")),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Candle":
        return cls(
            provider=row["provider"],
            market=row["market"],
            symbol=row["symbol"],
            interval=row["interval"],
            open_time=row["open_time"],
            close_time=row["close_time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            quote_volume=row["quote_volume"],
            trade_count=row["trade_count"],
            taker_buy_base_volume=row["taker_buy_base_volume"],
            taker_buy_quote_volume=row["taker_buy_quote_volume"],
        )

    def compact(self) -> dict[str, float]:
        return {
            "t": float(self.open_time),
            "o": self.open,
            "h": self.high,
            "l": self.low,
            "c": self.close,
            "v": self.volume,
        }

    def detailed(self) -> dict[str, Any]:
        return {
            "timestamp": self.open_time,
            "time": datetime.fromtimestamp(self.open_time / 1000.0, tz=timezone.utc).isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_timestamp": self.close_time,
            "close_time": datetime.fromtimestamp(self.close_time / 1000.0, tz=timezone.utc).isoformat(),
            "quote_volume": self.quote_volume,
            "trade_count": self.trade_count,
            "taker_buy_base_volume": self.taker_buy_base_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
        }


@dataclass(frozen=True)
class AggTrade:
    symbol: str
    aggregate_trade_id: int
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    trade_time: int
    is_buyer_maker: bool
    is_best_match: bool
    provider: str = DEFAULT_PROVIDER
    market: str = DEFAULT_MARKET

    @classmethod
    def from_record(
        cls,
        symbol: str,
        record: dict[str, Any],
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> "AggTrade":
        return cls(
            provider=provider,
            market=market,
            symbol=normalize_symbol(symbol),
            aggregate_trade_id=int(record["aggregate_trade_id"]),
            price=float(record["price"]),
            quantity=float(record["quantity"]),
            first_trade_id=int(record["first_trade_id"]),
            last_trade_id=int(record["last_trade_id"]),
            trade_time=int(record["timestamp"]),
            is_buyer_maker=bool(record["is_buyer_maker"]),
            is_best_match=bool(record["is_best_match"]),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AggTrade":
        return cls(
            provider=row["provider"],
            market=row["market"],
            symbol=row["symbol"],
            aggregate_trade_id=row["aggregate_trade_id"],
            price=row["price"],
            quantity=row["quantity"],
            first_trade_id=row["first_trade_id"],
            last_trade_id=row["last_trade_id"],
            trade_time=row["trade_time"],
            is_buyer_maker=bool(row["is_buyer_maker"]),
            is_best_match=bool(row["is_best_match"]),
        )

    def detailed(self) -> dict[str, Any]:
        return {
            "aggregate_trade_id": self.aggregate_trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "first_trade_id": self.first_trade_id,
            "last_trade_id": self.last_trade_id,
            "timestamp": self.trade_time,
            "time": datetime.fromtimestamp(self.trade_time / 1000.0, tz=timezone.utc).isoformat(),
            "is_buyer_maker": self.is_buyer_maker,
            "is_best_match": self.is_best_match,
        }


class LocalMarketCache:
    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        ensure_runtime_dirs()
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candles (
                        provider TEXT NOT NULL,
                        market TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        open_time INTEGER NOT NULL,
                        close_time INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        quote_volume REAL,
                        trade_count INTEGER,
                        taker_buy_base_volume REAL,
                        taker_buy_quote_volume REAL,
                        fetched_at INTEGER NOT NULL,
                        PRIMARY KEY (provider, market, symbol, interval, open_time)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_candles_range
                    ON candles (provider, market, symbol, interval, open_time, close_time)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agg_trades (
                        provider TEXT NOT NULL,
                        market TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        aggregate_trade_id INTEGER NOT NULL,
                        price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        first_trade_id INTEGER NOT NULL,
                        last_trade_id INTEGER NOT NULL,
                        trade_time INTEGER NOT NULL,
                        is_buyer_maker INTEGER NOT NULL,
                        is_best_match INTEGER NOT NULL,
                        fetched_at INTEGER NOT NULL,
                        PRIMARY KEY (provider, market, symbol, aggregate_trade_id)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agg_trades_range
                    ON agg_trades (provider, market, symbol, trade_time, aggregate_trade_id)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agg_trade_coverage (
                        provider TEXT NOT NULL,
                        market TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        start_time INTEGER NOT NULL,
                        end_time INTEGER NOT NULL,
                        fetched_at INTEGER NOT NULL,
                        PRIMARY KEY (provider, market, symbol, start_time, end_time)
                    )
                    """
                )

    def upsert_candles(self, candles: Iterable[Candle]) -> int:
        rows = [
            (
                candle.provider,
                candle.market,
                candle.symbol,
                candle.interval,
                candle.open_time,
                candle.close_time,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.quote_volume,
                candle.trade_count,
                candle.taker_buy_base_volume,
                candle.taker_buy_quote_volume,
                int(datetime.now(timezone.utc).timestamp() * 1000),
            )
            for candle in candles
        ]
        if not rows:
            return 0
        with closing(self.connect()) as connection:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO candles (
                        provider, market, symbol, interval, open_time, close_time,
                        open, high, low, close, volume, quote_volume, trade_count,
                        taker_buy_base_volume, taker_buy_quote_volume, fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, market, symbol, interval, open_time) DO UPDATE SET
                        close_time = excluded.close_time,
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume,
                        quote_volume = excluded.quote_volume,
                        trade_count = excluded.trade_count,
                        taker_buy_base_volume = excluded.taker_buy_base_volume,
                        taker_buy_quote_volume = excluded.taker_buy_quote_volume,
                        fetched_at = excluded.fetched_at
                    """,
                    rows,
                )
        return len(rows)

    def upsert_agg_trades(self, trades: Iterable[AggTrade]) -> int:
        fetched_at = int(datetime.now(timezone.utc).timestamp() * 1000)
        rows = [
            (
                trade.provider,
                trade.market,
                trade.symbol,
                trade.aggregate_trade_id,
                trade.price,
                trade.quantity,
                trade.first_trade_id,
                trade.last_trade_id,
                trade.trade_time,
                int(trade.is_buyer_maker),
                int(trade.is_best_match),
                fetched_at,
            )
            for trade in trades
        ]
        if not rows:
            return 0
        with closing(self.connect()) as connection:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO agg_trades (
                        provider, market, symbol, aggregate_trade_id,
                        price, quantity, first_trade_id, last_trade_id,
                        trade_time, is_buyer_maker, is_best_match, fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, market, symbol, aggregate_trade_id) DO UPDATE SET
                        price = excluded.price,
                        quantity = excluded.quantity,
                        first_trade_id = excluded.first_trade_id,
                        last_trade_id = excluded.last_trade_id,
                        trade_time = excluded.trade_time,
                        is_buyer_maker = excluded.is_buyer_maker,
                        is_best_match = excluded.is_best_match,
                        fetched_at = excluded.fetched_at
                    """,
                    rows,
                )
        return len(rows)

    def preload_binance(
        self,
        symbol: str,
        interval: str,
        start_time: str | int | float,
        end_time: str | int | float,
        max_candles: int | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_symbol(symbol)
        if interval not in INTERVAL_MS:
            raise ValueError(f"unsupported interval: {interval}")
        start_ms = parse_time_ms(start_time)
        end_ms = parse_time_ms(end_time)
        if start_ms is None or end_ms is None:
            raise ValueError("start_time and end_time are required")
        if start_ms >= end_ms:
            raise ValueError("start_time must be before end_time")

        total_rows = 0
        total_written = 0
        next_start = start_ms
        remaining = max_candles

        while next_start < end_ms and (remaining is None or remaining > 0):
            request_limit = 5000 if remaining is None else min(5000, remaining)
            records = fetch_candle_records(
                symbol=normalized,
                interval=interval,
                limit=request_limit,
                start_time=next_start,
                end_time=end_ms,
            )
            if not records:
                break
            records = [
                record
                for record in records
                if next_start <= int(record["timestamp"]) < end_ms
            ]
            if not records:
                break

            candles = [Candle.from_record(normalized, interval, record) for record in records]
            total_rows += len(candles)
            total_written += self.upsert_candles(candles)
            if remaining is not None:
                remaining -= len(candles)

            last_open = candles[-1].open_time
            advanced = last_open + INTERVAL_MS[interval]
            if advanced <= next_start or len(candles) < request_limit:
                break
            next_start = advanced

        return {
            "symbol": normalized,
            "interval": interval,
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "rows_fetched": total_rows,
            "rows_written": total_written,
            "cache_path": str(self.path),
        }

    def preload_binance_agg_trades(
        self,
        symbol: str,
        start_time: str | int | float,
        end_time: str | int | float,
        max_trades: int | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_symbol(symbol)
        start_ms = parse_time_ms(start_time)
        end_ms = parse_time_ms(end_time)
        if start_ms is None or end_ms is None:
            raise ValueError("start_time and end_time are required")
        if start_ms >= end_ms:
            raise ValueError("start_time must be before end_time")

        fetched_at = int(datetime.now(timezone.utc).timestamp() * 1000)
        records = fetch_agg_trade_records(
            symbol=normalized,
            start_time=start_ms,
            end_time=end_ms,
            limit=max_trades,
        )
        trades = [AggTrade.from_record(normalized, record) for record in records if start_ms <= int(record["timestamp"]) < end_ms]
        written = self.upsert_agg_trades(trades)
        coverage_marked = (max_trades is None or len(records) < max_trades) and end_ms <= fetched_at
        if coverage_marked:
            self._mark_agg_trade_coverage(normalized, start_ms, end_ms)
        return {
            "symbol": normalized,
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "rows_fetched": len(trades),
            "rows_written": written,
            "coverage_marked": coverage_marked,
            "cache_path": str(self.path),
        }

    def get_agg_trades(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> list[AggTrade]:
        normalized = normalize_symbol(symbol)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM agg_trades
                WHERE provider = ?
                  AND market = ?
                  AND symbol = ?
                  AND trade_time >= ?
                  AND trade_time < ?
                ORDER BY trade_time ASC, aggregate_trade_id ASC
                """,
                [provider, market, normalized, int(start_ms), int(end_ms)],
            ).fetchall()
        return [AggTrade.from_row(row) for row in rows]

    def has_agg_trade_coverage(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> bool:
        normalized = normalize_symbol(symbol)
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM agg_trade_coverage
                WHERE provider = ?
                  AND market = ?
                  AND symbol = ?
                  AND start_time <= ?
                  AND end_time >= ?
                LIMIT 1
                """,
                [provider, market, normalized, int(start_ms), int(end_ms)],
            ).fetchone()
        return row is not None

    def _mark_agg_trade_coverage(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> None:
        fetched_at = int(datetime.now(timezone.utc).timestamp() * 1000)
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO agg_trade_coverage (
                        provider, market, symbol, start_time, end_time, fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [provider, market, normalize_symbol(symbol), int(start_ms), int(end_ms), fetched_at],
                )

    def get_candles(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        as_of_ms: int | None = None,
        limit: int | None = None,
        closed_only: bool = True,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> list[Candle]:
        normalized = normalize_symbol(symbol)
        clauses = ["provider = ?", "market = ?", "symbol = ?", "interval = ?"]
        params: list[Any] = [provider, market, normalized, interval]
        if start_ms is not None:
            clauses.append("open_time >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            clauses.append("open_time < ?")
            params.append(int(end_ms))
        if as_of_ms is not None:
            clauses.append(("close_time <= ?" if closed_only else "open_time <= ?"))
            params.append(int(as_of_ms))

        where = " AND ".join(clauses)
        order = "DESC" if limit is not None else "ASC"
        query = f"SELECT * FROM candles WHERE {where} ORDER BY open_time {order}"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))

        with closing(self.connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        candles = [Candle.from_row(row) for row in rows]
        if limit is not None:
            candles.reverse()
        return candles

    def latest_candle(
        self,
        symbol: str,
        interval: str,
        as_of_ms: int | None = None,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> Candle | None:
        candles = self.get_candles(
            symbol=symbol,
            interval=interval,
            as_of_ms=as_of_ms,
            limit=1,
            provider=provider,
            market=market,
        )
        return candles[-1] if candles else None

    def first_candle_at_or_after(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        before_ms: int | None = None,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> Candle | None:
        normalized = normalize_symbol(symbol)
        before_clause = "AND open_time < ?" if before_ms is not None else ""
        params: list[Any] = [provider, market, normalized, interval, int(start_ms)]
        if before_ms is not None:
            params.append(int(before_ms))
        with closing(self.connect()) as connection:
            row = connection.execute(
                f"""
                SELECT * FROM candles
                WHERE provider = ?
                  AND market = ?
                  AND symbol = ?
                  AND interval = ?
                  AND open_time >= ?
                  {before_clause}
                ORDER BY open_time ASC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return Candle.from_row(row) if row else None

    def count(
        self,
        symbol: str | None = None,
        interval: str | None = None,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> int:
        clauses = ["provider = ?", "market = ?"]
        params: list[Any] = [provider, market]
        if symbol:
            clauses.append("symbol = ?")
            params.append(normalize_symbol(symbol))
        if interval:
            clauses.append("interval = ?")
            params.append(interval)
        with closing(self.connect()) as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM candles WHERE {' AND '.join(clauses)}", params).fetchone()
        return int(row["count"])

    def available_intervals(
        self,
        symbol: str,
        provider: str = DEFAULT_PROVIDER,
        market: str = DEFAULT_MARKET,
    ) -> list[str]:
        normalized = normalize_symbol(symbol)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT interval FROM candles
                WHERE provider = ? AND market = ? AND symbol = ?
                """,
                (provider, market, normalized),
            ).fetchall()
        return sorted((row["interval"] for row in rows), key=lambda item: INTERVAL_MS.get(item, 10**18))

    def status(self) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT symbol, interval, COUNT(*) AS count, MIN(open_time) AS first_open_time,
                       MAX(open_time) AS last_open_time, MAX(close_time) AS last_close_time
                FROM candles
                GROUP BY symbol, interval
                ORDER BY symbol, interval
                """
            ).fetchall()
            agg_rows = connection.execute(
                """
                SELECT symbol, COUNT(*) AS coverage_count, MIN(start_time) AS first_start_time,
                       MAX(end_time) AS last_end_time
                FROM agg_trade_coverage
                GROUP BY symbol
                ORDER BY symbol
                """
            ).fetchall()
        return {
            "cache_path": str(self.path),
            "markets": [
                {
                    "symbol": row["symbol"],
                    "interval": row["interval"],
                    "count": row["count"],
                    "first_open_time": row["first_open_time"],
                    "last_open_time": row["last_open_time"],
                    "last_close_time": row["last_close_time"],
                }
                for row in rows
            ],
            "agg_trade_coverage": [
                {
                    "symbol": row["symbol"],
                    "coverage_count": row["coverage_count"],
                    "first_start_time": row["first_start_time"],
                    "last_end_time": row["last_end_time"],
                }
                for row in agg_rows
            ],
        }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
