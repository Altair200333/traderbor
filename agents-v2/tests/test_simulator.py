from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from traderbot_ai.tools.market import fetch_agg_trade_records
from traderbot_ai.simulator.execution import ExecutionEngine, Order
from traderbot_ai.simulator.market_cache import AggTrade, Candle, LocalMarketCache


SYMBOL = "BTCUSDT"
BASE_MS = 1_704_067_200_000


def ts(offset_ms: int) -> int:
    return BASE_MS + offset_ms


def candle(
    open_time: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    interval: str = "1m",
    duration_ms: int = 60_000,
) -> Candle:
    return Candle(
        symbol=SYMBOL,
        interval=interval,
        open_time=open_time,
        close_time=open_time + duration_ms - 1,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


def order(kind: str = "long", opened_at: int = BASE_MS) -> Order:
    if kind == "long":
        return Order.from_values("long", SYMBOL, 100.0, 95.0, 110.0, 1000.0, opened_at)
    return Order.from_values("short", SYMBOL, 100.0, 105.0, 90.0, 1000.0, opened_at)


class ExecutionEngineTests(unittest.TestCase):
    def test_long_take_profit_only(self) -> None:
        result = ExecutionEngine().resolve_order(order("long"), candles=[candle(ts(60_000), 100.0, 111.0, 99.0, 109.0)])
        self.assertEqual(result.status, "closed")
        self.assertEqual(result.exit_reason, "TP")
        self.assertEqual(result.exit_price, 110.0)
        self.assertEqual(result.pnl, 100.0)

    def test_long_stop_loss_only(self) -> None:
        result = ExecutionEngine().resolve_order(order("long"), candles=[candle(ts(60_000), 100.0, 101.0, 94.0, 96.0)])
        self.assertEqual(result.exit_reason, "SL")
        self.assertEqual(result.exit_price, 95.0)
        self.assertEqual(result.pnl, -50.0)

    def test_short_take_profit_only(self) -> None:
        result = ExecutionEngine().resolve_order(order("short"), candles=[candle(ts(60_000), 100.0, 101.0, 89.0, 92.0)])
        self.assertEqual(result.exit_reason, "TP")
        self.assertEqual(result.exit_price, 90.0)
        self.assertEqual(result.pnl, 100.0)

    def test_short_stop_loss_only(self) -> None:
        result = ExecutionEngine().resolve_order(order("short"), candles=[candle(ts(60_000), 100.0, 106.0, 98.0, 104.0)])
        self.assertEqual(result.exit_reason, "SL")
        self.assertEqual(result.exit_price, 105.0)
        self.assertEqual(result.pnl, -50.0)

    def test_both_hit_without_finer_data_is_conservative_sl_for_long(self) -> None:
        result = ExecutionEngine().resolve_order(order("long"), candles=[candle(ts(60_000), 100.0, 111.0, 94.0, 100.0)])
        self.assertEqual(result.exit_reason, "SL")
        self.assertTrue(result.ambiguous)

    def test_both_hit_without_finer_data_is_conservative_sl_for_short(self) -> None:
        result = ExecutionEngine().resolve_order(order("short"), candles=[candle(ts(60_000), 100.0, 106.0, 89.0, 100.0)])
        self.assertEqual(result.exit_reason, "SL")
        self.assertTrue(result.ambiguous)

    def test_boundary_hit_counts(self) -> None:
        result = ExecutionEngine().resolve_order(order("long"), candles=[candle(ts(60_000), 100.0, 110.0, 99.0, 100.0)])
        self.assertEqual(result.exit_reason, "TP")

    def test_candles_before_order_time_do_not_trigger(self) -> None:
        result = ExecutionEngine().resolve_order(
            order("long", opened_at=ts(60_000)),
            candles=[
                candle(BASE_MS, 100.0, 111.0, 99.0, 100.0),
                candle(ts(120_000), 100.0, 101.0, 99.0, 100.0),
            ],
        )
        self.assertEqual(result.status, "open")

    def test_entry_candle_can_trigger(self) -> None:
        result = ExecutionEngine().resolve_order(
            order("long", opened_at=ts(60_000)),
            candles=[candle(ts(60_000), 100.0, 111.0, 99.0, 110.0)],
        )
        self.assertEqual(result.exit_reason, "TP")

    def test_gap_through_stop_fills_at_open(self) -> None:
        result = ExecutionEngine().resolve_order(order("long"), candles=[candle(ts(60_000), 90.0, 91.0, 89.0, 90.0)])
        self.assertEqual(result.exit_reason, "SL")
        self.assertEqual(result.exit_price, 90.0)
        self.assertEqual(result.pnl, -100.0)

    def test_gap_through_take_profit_fills_at_open(self) -> None:
        result = ExecutionEngine().resolve_order(order("long"), candles=[candle(ts(60_000), 115.0, 116.0, 114.0, 115.0)])
        self.assertEqual(result.exit_reason, "TP")
        self.assertEqual(result.exit_price, 115.0)
        self.assertEqual(result.pnl, 150.0)

    def test_short_gap_through_stop_fills_at_open(self) -> None:
        result = ExecutionEngine().resolve_order(order("short"), candles=[candle(ts(60_000), 110.0, 111.0, 109.0, 110.0)])
        self.assertEqual(result.exit_reason, "SL")
        self.assertEqual(result.exit_price, 110.0)
        self.assertEqual(result.pnl, -100.0)

    def test_short_gap_loss_is_capped_at_order_amount(self) -> None:
        result = ExecutionEngine().resolve_order(order("short"), candles=[candle(ts(60_000), 250.0, 251.0, 249.0, 250.0)])
        self.assertEqual(result.exit_reason, "SL")
        self.assertEqual(result.exit_price, 250.0)
        self.assertEqual(result.cash_returned, 0.0)
        self.assertEqual(result.pnl, -1000.0)

    def test_fee_rate_reduces_cash_returned(self) -> None:
        result = ExecutionEngine().resolve_order(
            order("long"),
            candles=[candle(ts(60_000), 100.0, 111.0, 99.0, 109.0)],
            fee_rate=0.001,
        )
        self.assertAlmostEqual(result.fees, 2.1)
        self.assertAlmostEqual(result.pnl, 97.9)
        self.assertAlmostEqual(result.cash_returned, 1097.9)

    def test_direct_candles_must_be_closed_by_scan_until(self) -> None:
        result = ExecutionEngine().resolve_order(
            order("long"),
            candles=[candle(ts(60_000), 100.0, 111.0, 99.0, 110.0)],
            scan_until=ts(90_000),
        )
        self.assertEqual(result.status, "open")


class CacheAndResolutionTests(unittest.TestCase):
    def test_cache_returns_sorted_deduplicated_closed_candles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            first = candle(BASE_MS, 100.0, 101.0, 99.0, 100.0)
            second = candle(ts(60_000), 100.0, 102.0, 99.0, 101.0)
            cache.upsert_candles([second, first, second])
            candles = cache.get_candles(SYMBOL, "1m")
            self.assertEqual([item.open_time for item in candles], [BASE_MS, ts(60_000)])
            self.assertEqual(cache.count(SYMBOL, "1m"), 2)

    def test_as_of_uses_close_time_not_open_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            self.assertEqual(cache.get_candles(SYMBOL, "1m", as_of_ms=ts(30_000)), [])
            self.assertEqual(len(cache.get_candles(SYMBOL, "1m", as_of_ms=ts(59_999))), 1)

    def test_cached_resolution_requires_scan_until(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(ts(60_000), 100.0, 111.0, 99.0, 110.0)])
            with self.assertRaises(ValueError):
                ExecutionEngine(cache).resolve_order(order("long"), interval="1m")

    def test_preload_treats_end_time_as_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            base = BASE_MS
            with patch(
                "traderbot_ai.simulator.market_cache.fetch_candle_records",
                return_value=[
                    record(base),
                    record(base + 60_000),
                    record(base + 120_000),
                    record(base + 180_000),
                ],
            ):
                summary = cache.preload_binance(SYMBOL, "1m", base, base + 180_000)
            self.assertEqual(summary["rows_fetched"], 3)
            self.assertEqual(
                [item.open_time for item in cache.get_candles(SYMBOL, "1m")],
                [base, base + 60_000, base + 120_000],
            )

    def test_finer_cached_interval_resolves_parent_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            finer = [
                candle(ts(60_000 + i * 1000), 100.0, 101.0, 99.0, 100.0, interval="1s", duration_ms=1000)
                for i in range(60)
            ]
            finer[0] = candle(ts(60_000), 100.0, 111.0, 100.0, 110.0, interval="1s", duration_ms=1000)
            finer[1] = candle(ts(61_000), 110.0, 110.0, 94.0, 95.0, interval="1s", duration_ms=1000)
            cache.upsert_candles(
                [
                    candle(ts(60_000), 100.0, 111.0, 94.0, 100.0, interval="1m", duration_ms=60_000),
                    *finer,
                ]
            )
            result = ExecutionEngine(cache).resolve_order(order("long"), interval="1m", scan_until=ts(119_999))
            self.assertEqual(result.exit_reason, "TP")
            self.assertFalse(result.ambiguous)
            self.assertEqual(result.resolution_interval, "1s")
            self.assertEqual(result.source_interval, "1m")

    def test_partial_finer_data_falls_back_to_conservative_parent_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(ts(60_000), 100.0, 111.0, 94.0, 100.0, interval="1m", duration_ms=60_000),
                    candle(ts(60_000), 100.0, 111.0, 100.0, 110.0, interval="1s", duration_ms=1000),
                    candle(ts(61_000), 110.0, 110.0, 94.0, 95.0, interval="1s", duration_ms=1000),
                ]
            )
            result = ExecutionEngine(cache).resolve_order(order("long"), interval="1m", scan_until=ts(119_999))
            self.assertEqual(result.exit_reason, "SL")
            self.assertTrue(result.ambiguous)
            self.assertEqual(result.resolution_interval, "1m")

    def test_agg_trades_resolve_ambiguous_candle_when_covered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(ts(60_000), 100.0, 111.0, 94.0, 100.0, interval="1m", duration_ms=60_000)])
            with patch(
                "traderbot_ai.simulator.market_cache.fetch_agg_trade_records",
                return_value=[
                    agg_record(1, ts(60_100), 110.0),
                    agg_record(2, ts(60_200), 95.0),
                ],
            ):
                cache.preload_binance_agg_trades(SYMBOL, ts(60_000), ts(120_000))

            result = ExecutionEngine(cache).resolve_order(order("long"), interval="1m", scan_until=ts(119_999))

            self.assertEqual(result.exit_reason, "TP")
            self.assertEqual(result.resolution_interval, "aggTrades")
            self.assertEqual(result.source_interval, "1m")
            self.assertEqual(result.exit_time_ms, ts(60_100))
            self.assertFalse(result.ambiguous)

    def test_nested_finer_resolution_preserves_agg_trade_interval_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            parent_duration = 4 * 60 * 60_000
            finer = [
                candle(BASE_MS + i * 60_000, 100.0, 101.0, 99.0, 100.0, interval="1m", duration_ms=60_000)
                for i in range(240)
            ]
            finer[0] = candle(BASE_MS, 100.0, 111.0, 94.0, 100.0, interval="1m", duration_ms=60_000)
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 111.0, 94.0, 100.0, interval="4h", duration_ms=parent_duration),
                    *finer,
                ]
            )
            with patch(
                "traderbot_ai.simulator.market_cache.fetch_agg_trade_records",
                return_value=[
                    agg_record(1, BASE_MS + 100, 110.0),
                    agg_record(2, BASE_MS + 200, 95.0),
                ],
            ):
                cache.preload_binance_agg_trades(SYMBOL, BASE_MS, BASE_MS + 60_000)

            result = ExecutionEngine(cache).resolve_order(
                order("long"),
                interval="4h",
                scan_until=BASE_MS + parent_duration - 1,
            )

            self.assertEqual(result.exit_reason, "TP")
            self.assertEqual(result.resolution_interval, "aggTrades")
            self.assertEqual(result.source_interval, "4h")
            self.assertEqual(result.exit_time_ms, BASE_MS + 100)
            self.assertFalse(result.ambiguous)

    def test_capped_agg_trade_preload_does_not_mark_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            with patch(
                "traderbot_ai.simulator.market_cache.fetch_agg_trade_records",
                return_value=[
                    agg_record(1, ts(60_100), 110.0),
                    agg_record(2, ts(60_200), 95.0),
                ],
            ):
                summary = cache.preload_binance_agg_trades(SYMBOL, ts(60_000), ts(120_000), max_trades=2)

            self.assertFalse(summary["coverage_marked"])
            self.assertFalse(cache.has_agg_trade_coverage(SYMBOL, ts(60_000), ts(120_000)))

    def test_future_agg_trade_preload_does_not_mark_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            future_start = 4_102_444_800_000
            future_end = future_start + 60_000
            with patch("traderbot_ai.simulator.market_cache.fetch_agg_trade_records", return_value=[]):
                summary = cache.preload_binance_agg_trades(SYMBOL, future_start, future_end)

            self.assertFalse(summary["coverage_marked"])
            self.assertFalse(cache.has_agg_trade_coverage(SYMBOL, future_start, future_end))

    def test_agg_trade_fetch_paginates_with_from_id_inside_same_millisecond(self) -> None:
        calls = []

        class Response:
            def __init__(self, rows: list[dict]) -> None:
                self._rows = rows

            def raise_for_status(self) -> None:
                return None

            def json(self) -> list[dict]:
                return self._rows

        def fake_get(_url: str, params: dict, timeout: int) -> Response:
            calls.append(params)
            if "startTime" in params:
                return Response([binance_agg_row(1, ts(60_000), 100.0), binance_agg_row(2, ts(60_000), 101.0)])
            self.assertEqual(params["fromId"], 3)
            return Response([binance_agg_row(3, ts(60_000), 102.0), binance_agg_row(4, ts(60_001), 103.0)])

        with patch("traderbot_ai.tools.market.BINANCE_AGG_TRADE_LIMIT", 2):
            with patch("traderbot_ai.tools.market.requests.get", side_effect=fake_get):
                trades = fetch_agg_trade_records(SYMBOL, ts(60_000), ts(60_001))

        self.assertEqual([trade["aggregate_trade_id"] for trade in trades], [1, 2, 3])
        self.assertEqual(calls[1]["fromId"], 3)

    def test_partial_agg_trades_without_coverage_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(ts(60_000), 100.0, 111.0, 94.0, 100.0, interval="1m", duration_ms=60_000)])
            cache.upsert_agg_trades(
                [
                    AggTrade(
                        symbol=SYMBOL,
                        aggregate_trade_id=1,
                        price=110.0,
                        quantity=1.0,
                        first_trade_id=1,
                        last_trade_id=1,
                        trade_time=ts(60_100),
                        is_buyer_maker=False,
                        is_best_match=True,
                    )
                ]
            )

            result = ExecutionEngine(cache).resolve_order(order("long"), interval="1m", scan_until=ts(119_999))

            self.assertEqual(result.exit_reason, "SL")
            self.assertEqual(result.resolution_interval, "1m")
            self.assertTrue(result.ambiguous)

    def test_finer_cached_interval_resolves_short_stop_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            finer = [
                candle(ts(60_000 + i * 1000), 100.0, 101.0, 99.0, 100.0, interval="1s", duration_ms=1000)
                for i in range(60)
            ]
            finer[0] = candle(ts(60_000), 100.0, 106.0, 100.0, 105.0, interval="1s", duration_ms=1000)
            finer[1] = candle(ts(61_000), 105.0, 105.0, 89.0, 90.0, interval="1s", duration_ms=1000)
            cache.upsert_candles(
                [
                    candle(ts(60_000), 100.0, 106.0, 89.0, 100.0, interval="1m", duration_ms=60_000),
                    *finer,
                ]
            )
            result = ExecutionEngine(cache).resolve_order(order("short"), interval="1m", scan_until=ts(119_999))
            self.assertEqual(result.exit_reason, "SL")
            self.assertFalse(result.ambiguous)
            self.assertEqual(result.resolution_interval, "1s")


if __name__ == "__main__":
    unittest.main()


def record(open_time: int) -> dict:
    return {
        "timestamp": open_time,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1.0,
        "close_timestamp": open_time + 59_999,
        "quote_volume": 100.0,
        "trade_count": 1,
        "taker_buy_base_volume": 0.5,
        "taker_buy_quote_volume": 50.0,
    }


def agg_record(aggregate_trade_id: int, timestamp: int, price: float) -> dict:
    return {
        "aggregate_trade_id": aggregate_trade_id,
        "price": price,
        "quantity": 1.0,
        "first_trade_id": aggregate_trade_id,
        "last_trade_id": aggregate_trade_id,
        "timestamp": timestamp,
        "is_buyer_maker": False,
        "is_best_match": True,
    }


def binance_agg_row(aggregate_trade_id: int, timestamp: int, price: float) -> dict:
    return {
        "a": aggregate_trade_id,
        "p": str(price),
        "q": "1.0",
        "f": aggregate_trade_id,
        "l": aggregate_trade_id,
        "T": timestamp,
        "m": False,
        "M": True,
    }
