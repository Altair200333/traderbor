from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from traderbot_ai.screener.market import DEFAULT_MARKET, DEFAULT_PROVIDER, INTERVAL_MS, normalize_symbol


@dataclass(frozen=True)
class CandleFrame:
    symbol: str
    interval_ms: int
    open_time: list[int]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]

    @property
    def length(self) -> int:
        return len(self.open_time)


@dataclass(frozen=True)
class DataIssue:
    status: Literal["insufficient_data", "data_gap", "bad_data"]
    reason: str
    detail: dict[str, int | float | str | None]


def load_closed(
    db_path: str | Path,
    symbol: str,
    interval_ms: int,
    as_of_ms: int,
    limit: int,
    provider: str = DEFAULT_PROVIDER,
    market: str = DEFAULT_MARKET,
) -> CandleFrame:
    interval = _interval_name(interval_ms)
    normalized = normalize_symbol(symbol)
    latest_allowed_open = int(as_of_ms) - int(interval_ms)
    path = Path(db_path)
    if not path.exists():
        return frame_from_candles([], normalized, interval_ms)
    with closing(sqlite3.connect(path, timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT open_time, open, high, low, close, volume
                FROM candles
                WHERE provider = ?
                  AND market = ?
                  AND symbol = ?
                  AND interval = ?
                  AND open_time <= ?
                ORDER BY open_time DESC
                LIMIT ?
                """,
                [provider, market, normalized, interval, latest_allowed_open, max(1, int(limit))],
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    return frame_from_rows(list(reversed(rows)), normalized, interval_ms)


def frame_from_candles(candles: list[Any], symbol: str, interval_ms: int) -> CandleFrame:
    normalized = normalize_symbol(symbol)
    return CandleFrame(
        symbol=normalized,
        interval_ms=int(interval_ms),
        open_time=[int(candle.open_time) for candle in candles],
        open=[float(candle.open) for candle in candles],
        high=[float(candle.high) for candle in candles],
        low=[float(candle.low) for candle in candles],
        close=[float(candle.close) for candle in candles],
        volume=[float(candle.volume) for candle in candles],
    )


def frame_from_rows(rows: list[Any], symbol: str, interval_ms: int) -> CandleFrame:
    normalized = normalize_symbol(symbol)
    return CandleFrame(
        symbol=normalized,
        interval_ms=int(interval_ms),
        open_time=[int(row["open_time"]) for row in rows],
        open=[float(row["open"]) for row in rows],
        high=[float(row["high"]) for row in rows],
        low=[float(row["low"]) for row in rows],
        close=[float(row["close"]) for row in rows],
        volume=[float(row["volume"]) for row in rows],
    )


def validate_frame(frame: CandleFrame, as_of_ms: int, min_bars: int) -> DataIssue | None:
    if frame.length < int(min_bars):
        return DataIssue("insufficient_data", "insufficient_data", {"bars": frame.length, "required": int(min_bars)})
    seen: set[int] = set()
    for index, open_time in enumerate(frame.open_time):
        if open_time % frame.interval_ms != 0:
            return DataIssue("data_gap", "unaligned_open_time", {"open_time": open_time, "interval_ms": frame.interval_ms})
        if open_time in seen:
            return DataIssue("data_gap", "duplicate_open_time", {"open_time": open_time})
        seen.add(open_time)
        if index > 0 and open_time - frame.open_time[index - 1] != frame.interval_ms:
            return DataIssue(
                "data_gap",
                "gap",
                {
                    "previous_open_time": frame.open_time[index - 1],
                    "open_time": open_time,
                    "expected_delta_ms": frame.interval_ms,
                    "actual_delta_ms": open_time - frame.open_time[index - 1],
                },
            )
        if open_time + frame.interval_ms > int(as_of_ms):
            return DataIssue("data_gap", "partial_candle", {"open_time": open_time, "as_of_ms": int(as_of_ms)})
        values = (frame.open[index], frame.high[index], frame.low[index], frame.close[index], frame.volume[index])
        if not all(math.isfinite(value) for value in values):
            return DataIssue("bad_data", "non_finite", {"open_time": open_time})
        if frame.high[index] < max(frame.open[index], frame.close[index]):
            return DataIssue("bad_data", "high_below_body", {"open_time": open_time})
        if frame.low[index] > min(frame.open[index], frame.close[index]):
            return DataIssue("bad_data", "low_above_body", {"open_time": open_time})
        if frame.volume[index] < 0:
            return DataIssue("bad_data", "negative_volume", {"open_time": open_time})
    last_close = frame.open_time[-1] + frame.interval_ms
    age = int(as_of_ms) - last_close
    if age < 0 or age >= frame.interval_ms:
        return DataIssue("data_gap", "missing_tail", {"last_close_ms": last_close, "as_of_ms": int(as_of_ms), "age_ms": age})
    return None


def _interval_name(interval_ms: int) -> str:
    for name, value in INTERVAL_MS.items():
        if value == int(interval_ms):
            return name
    raise ValueError(f"unsupported interval_ms: {interval_ms}")
