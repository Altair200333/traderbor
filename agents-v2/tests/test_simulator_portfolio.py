from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from traderbot_ai.agents.trading import build_tools
from traderbot_ai.cli import build_parser
from traderbot_ai.config import Settings
from traderbot_ai.simulator.clock import active_simulation_clock_ms, file_simulation_clock_ms
from traderbot_ai.simulator.backtest import build_config, run_backtest
from traderbot_ai.simulator.entry import resolve_entry_fill
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.simulator.portfolio import SimulatedPortfolio
from traderbot_ai.tools.portfolio import validate_paper_order_request
from traderbot_ai.tools.simulator import (
    clear_simulation_clock_impl,
    get_cached_candles_impl,
    get_simulation_portfolio_impl,
    place_simulated_order_impl,
    set_simulation_clock_impl,
    settle_simulation_impl,
    simulate_order_exit_impl,
)


SYMBOL = "BTCUSDT"
BASE_MS = 1_704_067_200_000


def candle(open_time: int, open_price: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        symbol=SYMBOL,
        interval="1m",
        open_time=open_time,
        close_time=open_time + 59_999,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class SimulatedPortfolioTests(unittest.TestCase):
    def test_place_and_settle_long_take_profit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(BASE_MS + 60_000, 100.0, 111.0, 99.0, 110.0),
                ]
            )

            portfolio.reset(balance_usdt=1000.0, as_of=BASE_MS)
            placed = portfolio.place_order("long", SYMBOL, 100.0, 95.0, 110.0, 500.0, BASE_MS)
            self.assertEqual(placed["state"]["balance"]["USDT"], 500.0)
            self.assertEqual(len(placed["state"]["open_orders"]), 1)

            settled = portfolio.settle(as_of=BASE_MS + 119_999, interval="1m")
            self.assertEqual(len(settled["closed_trades"]), 1)
            self.assertEqual(settled["closed_trades"][0]["exit_reason"], "TP")
            self.assertEqual(settled["state"]["balance"]["USDT"], 1050.0)
            self.assertEqual(settled["state"]["open_orders"], [])

    def test_place_rejects_bad_short_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            portfolio.reset(balance_usdt=1000.0, as_of=BASE_MS)
            with self.assertRaises(ValueError):
                portfolio.place_order("short", SYMBOL, 100.0, 95.0, 110.0, 500.0, BASE_MS)

    def test_summary_marks_open_order_equity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            cache.upsert_candles([candle(BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            portfolio.reset(balance_usdt=1000.0, as_of=BASE_MS)
            portfolio.place_order("long", SYMBOL, 100.0, 95.0, 110.0, 500.0, BASE_MS)
            summary = portfolio.summary(as_of=BASE_MS + 59_999)
            self.assertEqual(summary["cash_usdt"], 500.0)
            self.assertEqual(summary["open_equity_usdt"], 500.0)
            self.assertEqual(summary["total_equity_usdt"], 1000.0)

    def test_short_gap_settlement_cannot_make_cash_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(BASE_MS + 60_000, 250.0, 251.0, 249.0, 250.0),
                ]
            )

            portfolio.reset(balance_usdt=1000.0, as_of=BASE_MS)
            portfolio.place_order("short", SYMBOL, 100.0, 105.0, 90.0, 500.0, BASE_MS)
            settled = portfolio.settle(as_of=BASE_MS + 119_999, interval="1m")
            self.assertEqual(settled["closed_trades"][0]["cash_returned"], 0.0)
            self.assertEqual(settled["closed_trades"][0]["pnl"], -500.0)
            self.assertEqual(settled["state"]["balance"]["USDT"], 500.0)

    def test_short_mark_equity_is_not_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            cache.upsert_candles([candle(BASE_MS, 250.0, 251.0, 249.0, 250.0)])
            portfolio.reset(balance_usdt=1000.0, as_of=BASE_MS)
            portfolio.place_order("short", SYMBOL, 100.0, 105.0, 90.0, 500.0, BASE_MS)

            summary = portfolio.summary(as_of=BASE_MS + 59_999)
            self.assertEqual(summary["open_equity_usdt"], 0.0)
            self.assertEqual(summary["total_equity_usdt"], 500.0)


class BacktestRunnerTests(unittest.TestCase):
    def test_backtest_cli_preloads_by_default(self) -> None:
        args = build_parser().parse_args(
            [
                "backtest",
                "--symbol",
                SYMBOL,
                "--start-time",
                "2024-01-01T00:00:00Z",
                "--end-time",
                "2024-01-01T00:02:00Z",
            ]
        )

        self.assertFalse(args.no_preload)
        self.assertFalse(args.agg_trades)
        self.assertEqual(args.cache_intervals, "1m,1h,4h")

    def test_entry_fill_uses_next_execution_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(BASE_MS + 60_000, 105.0, 106.0, 104.0, 105.0),
                ]
            )

            exact = resolve_entry_fill(cache, SYMBOL, BASE_MS, requested_price=999.0)
            mid_candle = resolve_entry_fill(cache, SYMBOL, BASE_MS + 30_000, requested_price=999.0)

            self.assertEqual(exact.fill_price, 100.0)
            self.assertEqual(exact.opened_at_ms, BASE_MS)
            self.assertEqual(exact.requested_price, 999.0)
            self.assertEqual(mid_candle.fill_price, 105.0)
            self.assertEqual(mid_candle.opened_at_ms, BASE_MS + 60_000)
            with self.assertRaises(ValueError):
                resolve_entry_fill(cache, SYMBOL, BASE_MS + 30_000, before=BASE_MS + 60_000)

    def test_backtest_places_and_settles_fake_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(BASE_MS + 60_000, 100.0, 111.0, 99.0, 110.0),
                ]
            )
            calls = []

            def decide(context: dict) -> dict:
                calls.append(context)
                if len(calls) == 1:
                    return {
                        "final_decision": "long",
                        "symbol": SYMBOL,
                        "price": 999.0,
                        "stop_loss": 95.0,
                        "take_profit": 110.0,
                        "amount": 500.0,
                    }
                return {"final_decision": "hold", "symbol": SYMBOL, "amount": 0.0}

            result = run_backtest(
                build_config(
                    symbol=SYMBOL,
                    start_time=BASE_MS,
                    end_time=BASE_MS + 120_000,
                    decision_interval="1m",
                    execution_interval="1m",
                    balance_usdt=1000.0,
                ),
                decide=decide,
                cache=cache,
                portfolio=portfolio,
            )

            self.assertEqual(len(result["steps"]), 2)
            self.assertEqual(result["final_portfolio"]["closed_trade_count"], 1)
            self.assertEqual(result["final_portfolio"]["open_order_count"], 0)
            self.assertEqual(result["final_portfolio"]["cash_usdt"], 1050.0)
            first_order = result["steps"][0]["order_result"]
            self.assertEqual(first_order["entry_fill"]["requested_price"], 999.0)
            self.assertEqual(first_order["entry_fill"]["fill_price"], 100.0)
            self.assertEqual(first_order["order"]["price"], 100.0)
            self.assertIsNone(active_simulation_clock_ms())

    def test_backtest_reports_missing_entry_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)

            def decide(_context: dict) -> dict:
                return {
                    "final_decision": "long",
                    "symbol": SYMBOL,
                    "price": 100.0,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "amount": 500.0,
                }

            result = run_backtest(
                build_config(
                    symbol=SYMBOL,
                    start_time=BASE_MS,
                    end_time=BASE_MS + 60_000,
                    decision_interval="1m",
                    execution_interval="1m",
                    balance_usdt=1000.0,
                ),
                decide=decide,
                cache=cache,
                portfolio=portfolio,
            )

            self.assertFalse(result["steps"][0]["order_result"]["ok"])
            self.assertIn("no cached entry candle", result["steps"][0]["order_result"]["error"])

    def test_backtest_sets_clock_while_deciding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(BASE_MS + 60_000, 100.0, 102.0, 99.0, 101.0),
                ]
            )
            counts = []
            file_clocks = []

            def decide(context: dict) -> dict:
                file_clocks.append(file_simulation_clock_ms())
                with patch("traderbot_ai.tools.simulator._cache", return_value=cache):
                    result = get_cached_candles_impl(SYMBOL, interval="1m", lookback=10)
                counts.append(len(result["candles"]))
                return {"final_decision": "hold", "symbol": SYMBOL, "amount": 0.0}

            run_backtest(
                build_config(
                    symbol=SYMBOL,
                    start_time=BASE_MS,
                    end_time=BASE_MS + 120_000,
                    decision_interval="1m",
                    execution_interval="1m",
                    balance_usdt=1000.0,
                ),
                decide=decide,
                cache=cache,
                portfolio=portfolio,
            )

            self.assertEqual(counts, [0, 1])
            self.assertEqual(file_clocks, [BASE_MS, BASE_MS + 60_000])
            self.assertIsNone(active_simulation_clock_ms())
            self.assertIsNone(file_simulation_clock_ms())

    def test_backtest_tools_do_not_include_order_write_tools(self) -> None:
        settings = Settings(
            model="test",
            vision_model="test",
            reasoning_effort="low",
            paper_balance_usdt=1000.0,
            market_data_mode="cache",
            openai_api_key_present=False,
            enable_codex_tool=False,
            disable_tracing=True,
        )
        tool_names = {getattr(tool, "name", "") for tool in build_tools(settings, backtest=True)}

        self.assertIn("get_candles", tool_names)
        self.assertIn("simulate_order_exit", tool_names)
        self.assertNotIn("preload_market_cache", tool_names)
        self.assertNotIn("get_order_book", tool_names)
        self.assertNotIn("get_simulation_portfolio", tool_names)
        self.assertNotIn("place_simulated_order", tool_names)
        self.assertNotIn("settle_simulation", tool_names)
        self.assertNotIn("reset_simulation", tool_names)
        self.assertNotIn("paper_place_order", tool_names)
        self.assertNotIn("run_python_code", tool_names)
        self.assertNotIn("write_local_file", tool_names)
        self.assertNotIn("append_worklog", tool_names)
        self.assertNotIn("store_chart", tool_names)
        self.assertNotIn("load_chart_metadata", tool_names)
        self.assertNotIn("inspect_image", tool_names)

    def test_backtest_tools_are_cache_only_even_with_live_settings(self) -> None:
        settings = Settings(
            model="test",
            vision_model="test",
            reasoning_effort="low",
            paper_balance_usdt=1000.0,
            market_data_mode="live",
            openai_api_key_present=False,
            enable_codex_tool=False,
            disable_tracing=True,
        )
        tool_names = {getattr(tool, "name", "") for tool in build_tools(settings, backtest=True)}

        self.assertIn("get_candles", tool_names)
        self.assertIn("get_current_price", tool_names)
        self.assertNotIn("get_order_book", tool_names)
        self.assertNotIn("save_market_artifact", tool_names)
        self.assertNotIn("store_chart", tool_names)

    def test_backtest_tools_exclude_codex_even_when_enabled(self) -> None:
        settings = Settings(
            model="test",
            vision_model="test",
            reasoning_effort="low",
            paper_balance_usdt=1000.0,
            market_data_mode="cache",
            openai_api_key_present=False,
            enable_codex_tool=True,
            disable_tracing=True,
        )
        tool_names = {getattr(tool, "name", "") for tool in build_tools(settings, backtest=True)}

        self.assertNotIn("codex_code_worker", tool_names)


class SimulationClockToolTests(unittest.TestCase):
    def test_cached_candles_are_guarded_by_simulation_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(BASE_MS + 60_000, 100.0, 102.0, 99.0, 101.0),
                ]
            )
            try:
                self.assertTrue(set_simulation_clock_impl(BASE_MS + 59_999)["ok"])
                with patch("traderbot_ai.tools.simulator._cache", return_value=cache):
                    implicit = get_cached_candles_impl(SYMBOL, interval="1m", lookback=10)
                    future = get_cached_candles_impl(SYMBOL, interval="1m", as_of=BASE_MS + 119_999, lookback=10)
                self.assertTrue(implicit["ok"])
                self.assertEqual(len(implicit["candles"]), 1)
                self.assertFalse(future["ok"])
                self.assertIn("exceeds simulation clock", future["error"])
            finally:
                clear_simulation_clock_impl()

    def test_simulate_order_exit_rejects_future_scan_until(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            try:
                self.assertTrue(set_simulation_clock_impl(BASE_MS + 59_999)["ok"])
                with patch("traderbot_ai.tools.simulator._cache", return_value=cache):
                    result = simulate_order_exit_impl("long", SYMBOL, 100.0, 95.0, 110.0, 100.0, BASE_MS, BASE_MS + 119_999)
                self.assertFalse(result["ok"])
                self.assertIn("exceeds simulation clock", result["error"])
            finally:
                clear_simulation_clock_impl()

    def test_simulate_order_exit_rejects_future_opened_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            try:
                self.assertTrue(set_simulation_clock_impl(BASE_MS + 59_999)["ok"])
                with patch("traderbot_ai.tools.simulator._cache", return_value=cache):
                    result = simulate_order_exit_impl("long", SYMBOL, 100.0, 95.0, 110.0, 100.0, BASE_MS + 119_999, BASE_MS + 59_999)
                self.assertFalse(result["ok"])
                self.assertIn("exceeds simulation clock", result["error"])
            finally:
                clear_simulation_clock_impl()

    def test_settle_simulation_rejects_future_as_of(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            try:
                self.assertTrue(set_simulation_clock_impl(BASE_MS + 59_999)["ok"])
                with patch("traderbot_ai.tools.simulator.SimulatedPortfolio", return_value=portfolio):
                    result = settle_simulation_impl(BASE_MS + 119_999)
                self.assertFalse(result["ok"])
                self.assertIn("exceeds simulation clock", result["error"])
            finally:
                clear_simulation_clock_impl()

    def test_simulation_portfolio_omitted_as_of_uses_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            cache.upsert_candles(
                [
                    candle(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(BASE_MS + 60_000, 110.0, 111.0, 109.0, 110.0),
                ]
            )
            portfolio.reset(balance_usdt=1000.0, as_of=BASE_MS)
            portfolio.place_order("long", SYMBOL, 100.0, 95.0, 120.0, 500.0, BASE_MS)
            try:
                self.assertTrue(set_simulation_clock_impl(BASE_MS + 59_999)["ok"])
                with patch("traderbot_ai.tools.simulator.SimulatedPortfolio", return_value=portfolio):
                    result = get_simulation_portfolio_impl()
                self.assertTrue(result["ok"])
                self.assertEqual(result["total_equity_usdt"], 1000.0)
            finally:
                clear_simulation_clock_impl()

    def test_place_simulated_order_rejects_future_opened_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            portfolio = SimulatedPortfolio(Path(tmp) / "portfolio.json", cache=cache)
            portfolio.reset(balance_usdt=1000.0, as_of=BASE_MS)
            try:
                self.assertTrue(set_simulation_clock_impl(BASE_MS + 59_999)["ok"])
                with patch("traderbot_ai.tools.simulator.SimulatedPortfolio", return_value=portfolio):
                    result = place_simulated_order_impl("long", SYMBOL, 100.0, 95.0, 110.0, 100.0, BASE_MS + 119_999)
                self.assertFalse(result["ok"])
                self.assertIn("exceeds simulation clock", result["error"])
            finally:
                clear_simulation_clock_impl()


class PaperPortfolioValidationTests(unittest.TestCase):
    def test_paper_order_validation_rejects_bad_geometry(self) -> None:
        with self.assertRaises(ValueError):
            validate_paper_order_request("long", SYMBOL, 100.0, 105.0, 110.0, 100.0, 1000.0)

    def test_paper_order_validation_rejects_position_limit(self) -> None:
        with self.assertRaises(ValueError):
            validate_paper_order_request("long", SYMBOL, 100.0, 99.0, 110.0, 400.0, 1000.0)

    def test_paper_order_validation_rejects_loss_limit(self) -> None:
        with self.assertRaises(ValueError):
            validate_paper_order_request("long", SYMBOL, 100.0, 90.0, 110.0, 300.0, 1000.0)


if __name__ == "__main__":
    unittest.main()
