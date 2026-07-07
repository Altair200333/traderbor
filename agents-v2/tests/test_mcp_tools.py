from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from traderbot_ai.mcp import server
from traderbot_ai.mcp import tools as mcp_tools
from traderbot_ai.simulator.clock import (
    clear_file_simulation_clock_state,
    clear_process_simulation_clock_state,
    file_simulation_clock_ms,
    process_simulation_clock_ms,
    simulation_clock_path,
    set_file_simulation_clock_state,
    set_process_simulation_clock_state,
    set_simulation_clock_state,
)
from traderbot_ai.simulator.exchange_replay import exchange_tool_environment
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.tools.exchange import reset_exchange_impl


BTC = "BTCUSDT"
BASE_MS = 1_704_067_200_000


def candle_at_close(
    close_time: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    *,
    interval: str = "1m",
    duration_ms: int = 60_000,
    symbol: str = BTC,
) -> Candle:
    return Candle(
        symbol=symbol,
        interval=interval,
        open_time=close_time - duration_ms + 1,
        close_time=close_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class McpToolTests(unittest.TestCase):
    def test_registered_tools_match_strategy_surface(self) -> None:
        names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}

        self.assertIn("scan_momentum_universe", names)
        self.assertIn("get_setup_digest", names)
        self.assertIn("get_candidate_detail", names)
        self.assertIn("compute_indicators", names)
        self.assertIn("run_analysis_code", names)
        self.assertIn("get_wallet", names)
        self.assertIn("set_leverage", names)
        self.assertIn("place_order", names)
        self.assertIn("cancel_order", names)
        self.assertIn("close_position", names)
        self.assertNotIn("reset_exchange", names)
        self.assertNotIn("settle_exchange", names)
        self.assertNotIn("run_python_code", names)
        self.assertNotIn("preload_market_cache", names)
        self.assertNotIn("set_simulation_clock", names)

    def test_get_candles_uses_closed_only_cache_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle_at_close(BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle_at_close(BASE_MS + 60_000, 100.0, 105.0, 99.0, 104.0),
                ]
            )
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            try:
                os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache.path)
                result = mcp_tools.get_candles(BTC, interval="1m", as_of=BASE_MS, limit=10)
            finally:
                _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["candles"]), 1)
        self.assertEqual(result["candles"][0]["c"], 100.0)

    def test_get_candles_rejects_stale_as_of_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            try:
                os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache.path)
                result = mcp_tools.get_candles(BTC, interval="1m", as_of=BASE_MS + 120_000, limit=10)
            finally:
                _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)

        self.assertFalse(result["ok"])
        self.assertIn("stale cached candle data", result["error"])
        self.assertFalse(result["freshness"]["fresh"])

    def test_get_candles_allows_explicit_historical_range_under_active_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache.path)
                set_simulation_clock_state(BASE_MS + 10 * 60_000)
                result = mcp_tools.get_candles(
                    BTC,
                    interval="1m",
                    start_time=BASE_MS - 60_000,
                    end_time=BASE_MS + 1,
                    limit=10,
                )
            finally:
                _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertTrue(result["ok"], result)
        self.assertEqual(len(result["candles"]), 1)
        self.assertFalse(result["freshness"]["fresh"])

    def test_get_current_price_rejects_stale_mark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            try:
                os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache.path)
                result = mcp_tools.get_current_price(BTC, interval="1m", as_of=BASE_MS + 120_000)
            finally:
                _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)

        self.assertFalse(result["ok"])
        self.assertIn("stale cached mark price", result["error"])
        self.assertFalse(result["freshness"]["fresh"])

    def test_scan_momentum_universe_rejects_stale_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            interval_ms = 4 * 60 * 60_000
            candles = [
                candle_at_close(
                    BASE_MS - (59 - index) * interval_ms,
                    100.0 + index,
                    102.0 + index,
                    99.0 + index,
                    101.0 + index,
                    interval="4h",
                    duration_ms=interval_ms,
                )
                for index in range(60)
            ]
            cache.upsert_candles(candles)
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            try:
                os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache.path)
                result = mcp_tools.scan_momentum_universe([BTC], as_of=BASE_MS + interval_ms + 1)
            finally:
                _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "no fresh 1h candles at as_of")
        self.assertEqual(result["fresh_symbol_count"], 0)
        self.assertEqual(result["rejected"][0]["status"], "insufficient_data")

    def test_scan_momentum_universe_rejects_future_as_of_under_simulation_clock(self) -> None:
        previous_process_clock = process_simulation_clock_ms()
        previous_file_clock = file_simulation_clock_ms()
        try:
            clear_process_simulation_clock_state()
            set_file_simulation_clock_state(BASE_MS)
            result = mcp_tools.scan_momentum_universe([BTC], as_of=BASE_MS + 1)
        finally:
            if previous_file_clock is None:
                clear_file_simulation_clock_state()
            else:
                set_file_simulation_clock_state(previous_file_clock)
            if previous_process_clock is None:
                clear_process_simulation_clock_state()
            else:
                set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("exceeds simulation clock", result["error"])

    def test_deterministic_mcp_mode_allows_only_bounded_candidate_and_anchor_candles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle_at_close(BASE_MS, 100.0, 101.0, 99.0, 100.0, interval="1h", duration_ms=60 * 60_000),
                    candle_at_close(BASE_MS, 200.0, 201.0, 199.0, 200.0, interval="4h", duration_ms=4 * 60 * 60_000, symbol="ETHUSDT"),
                ]
            )
            old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
            old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            try:
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
                os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache.path)
                candidate = mcp_tools.get_candles(BTC, interval="1h", as_of=BASE_MS, limit=1)
                anchor = mcp_tools.get_candles("ETHUSDT", interval="4h", as_of=BASE_MS, limit=1)
                wrong_symbol = mcp_tools.get_candles("XRPUSDT", interval="1h", as_of=BASE_MS, limit=1)
                wrong_interval = mcp_tools.get_candles(BTC, interval="1m", as_of=BASE_MS, limit=1)
                too_many_1h = mcp_tools.get_candles(BTC, interval="1h", as_of=BASE_MS, limit=171)
                too_many_4h = mcp_tools.get_candles(BTC, interval="4h", as_of=BASE_MS, limit=61)
                missing_as_of = mcp_tools.get_candles(BTC, interval="1h", limit=1)
                ranged = mcp_tools.get_candles(BTC, interval="1h", as_of=BASE_MS, start_time=BASE_MS - 60_000, end_time=BASE_MS, limit=1)
                scan = mcp_tools.scan_momentum_universe([BTC], as_of=BASE_MS)
                cancel = mcp_tools.cancel_order(order_id="o1", as_of=BASE_MS)
                close = mcp_tools.close_position(position_id="p1", as_of=BASE_MS)
            finally:
                _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
                _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
                _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)

        self.assertTrue(candidate["ok"], candidate)
        self.assertEqual(candidate["symbol"], BTC)
        self.assertTrue(anchor["ok"], anchor)
        self.assertEqual(anchor["symbol"], "ETHUSDT")
        self.assertFalse(wrong_symbol["ok"])
        self.assertIn("runner candidates and BTC/ETH/SOL", wrong_symbol["error"])
        self.assertFalse(wrong_interval["ok"])
        self.assertIn("only allows 1h or 4h", wrong_interval["error"])
        self.assertFalse(too_many_1h["ok"])
        self.assertIn("limit for 1h must be <= 170", too_many_1h["error"])
        self.assertFalse(too_many_4h["ok"])
        self.assertIn("limit for 4h must be <= 60", too_many_4h["error"])
        self.assertFalse(missing_as_of["ok"])
        self.assertIn("requires exact as_of", missing_as_of["error"])
        self.assertFalse(ranged["ok"])
        self.assertIn("does not allow start_time or end_time", ranged["error"])
        self.assertFalse(scan["ok"])
        self.assertIn("broad scan already ran", scan["error"])
        self.assertFalse(cancel["ok"])
        self.assertIn("does not allow provider cancels", cancel["error"])
        self.assertFalse(close["ok"])
        self.assertIn("does not allow provider maintenance closes", close["error"])

    def test_deterministic_mcp_get_candles_rejects_future_as_of_under_simulation_clock(self) -> None:
        old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
        old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
        previous_process_clock = process_simulation_clock_ms()
        previous_file_clock = file_simulation_clock_ms()
        try:
            clear_process_simulation_clock_state()
            set_file_simulation_clock_state(BASE_MS)
            os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
            os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
            result = mcp_tools.get_candles(BTC, interval="1h", as_of=BASE_MS + 1, limit=1)
        finally:
            _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
            _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
            if previous_file_clock is None:
                clear_file_simulation_clock_state()
            else:
                set_file_simulation_clock_state(previous_file_clock)
            if previous_process_clock is None:
                clear_process_simulation_clock_state()
            else:
                set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("exceeds simulation clock", result["error"])

    def test_deterministic_setup_digest_rejects_non_finalist(self) -> None:
        old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
        old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
        try:
            os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
            os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
            wrong_symbol = mcp_tools.get_setup_digest("ETHUSDT", "long", BASE_MS)
            wrong_side = mcp_tools.get_candidate_detail(BTC, "short", BASE_MS)
        finally:
            _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
            _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)

        self.assertFalse(wrong_symbol["ok"])
        self.assertIn("only available for runner-provided deterministic candidates", wrong_symbol["error"])
        self.assertFalse(wrong_side["ok"])
        self.assertIn("only available for runner-provided deterministic candidates", wrong_side["error"])

    def test_deterministic_setup_digest_rejects_future_as_of_under_simulation_clock(self) -> None:
        old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
        old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
        previous_process_clock = process_simulation_clock_ms()
        previous_file_clock = file_simulation_clock_ms()
        try:
            clear_process_simulation_clock_state()
            set_file_simulation_clock_state(BASE_MS)
            os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
            os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
            result = mcp_tools.get_setup_digest(BTC, "long", BASE_MS + 1)
        finally:
            _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
            _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
            if previous_file_clock is None:
                clear_file_simulation_clock_state()
            else:
                set_file_simulation_clock_state(previous_file_clock)
            if previous_process_clock is None:
                clear_process_simulation_clock_state()
            else:
                set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("exceeds simulation clock", result["error"])

    def test_deterministic_place_order_rejects_non_candidate_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_path = tmp_path / "state.json"
            events_path = tmp_path / "events.jsonl"
            old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
            old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(state_path, events_path, backend="simulated", fee_rate=0.001, execution_interval="1m"):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    result = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Sell",
                        "Market",
                        qty=1.0,
                        takeProfit=90.0,
                        stopLoss=104.0,
                        orderLinkId="wrong-side",
                        as_of=BASE_MS,
                    )
                    event_text = events_path.read_text(encoding="utf-8")
            finally:
                _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
                _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("only available for runner-provided deterministic candidates", result["error"])
        event_types = [json.loads(line)["type"] for line in event_text.splitlines()]
        self.assertNotIn("place_order", event_types)

    def test_deterministic_stop_noise_error_unit(self) -> None:
        from traderbot_ai.tools import replay_helpers

        old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
        old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
        try:
            os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
            os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(
                [
                    {"symbol": BTC, "side": "long", "ref_price": 100.0, "max_drift_pct": 0.02, "noise_floor_pct": 0.02},
                    {"symbol": "ETHUSDT", "side": "short", "ref_price": 100.0, "max_drift_pct": 0.02, "noise_floor_pct": 0.02},
                ]
            )

            inside_long = replay_helpers.deterministic_stop_noise_error(BTC, "long", 100.0, 99.0)
            outside_long = replay_helpers.deterministic_stop_noise_error(BTC, "long", 100.0, 97.5)
            inside_short = replay_helpers.deterministic_stop_noise_error("ETHUSDT", "short", 100.0, 101.0)
            outside_short = replay_helpers.deterministic_stop_noise_error("ETHUSDT", "short", 100.0, 102.5)
            no_stop = replay_helpers.deterministic_stop_noise_error(BTC, "long", 100.0, None)
            unknown_symbol = replay_helpers.deterministic_stop_noise_error("SOLUSDT", "long", 100.0, 99.0)

            os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(
                [{"symbol": BTC, "side": "long", "ref_price": 100.0, "max_drift_pct": 0.02}]
            )
            no_floor = replay_helpers.deterministic_stop_noise_error(BTC, "long", 100.0, 99.0)
        finally:
            _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
            _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)

        self.assertIsNotNone(inside_long)
        self.assertFalse(inside_long["ok"])
        self.assertIn("noise floor", inside_long["error"])
        self.assertAlmostEqual(inside_long["noise_floor_pct"], 0.02)
        self.assertIsNone(outside_long)
        self.assertIsNotNone(inside_short)
        self.assertFalse(inside_short["ok"])
        self.assertIsNone(outside_short)
        self.assertIsNone(no_stop)
        self.assertIsNone(unknown_symbol)
        self.assertIsNone(no_floor)

    def test_deterministic_candidates_file_indirection(self) -> None:
        import tempfile

        from traderbot_ai.tools import replay_helpers

        old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
        old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
        old_path = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES_PATH")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                payload_file = Path(tmp) / "candidates.json"
                payload_file.write_text(json.dumps(
                    [{"symbol": BTC, "side": "long", "ref_price": 100.0, "max_drift_pct": 0.02, "noise_floor_pct": 0.02}]
                ), encoding="utf-8")
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ.pop("TRADERBOT_DETERMINISTIC_CANDIDATES", None)
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES_PATH"] = str(payload_file)

                adverse = replay_helpers.deterministic_entry_drift_error(BTC, "long", 103.0)
                stop_inside = replay_helpers.deterministic_stop_noise_error(BTC, "long", 100.0, 99.0)
        finally:
            _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
            _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
            _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES_PATH", old_path)

        self.assertIsNotNone(adverse)
        self.assertIn("drifted", adverse["error"])
        self.assertIsNotNone(stop_inside)
        self.assertIn("noise floor", stop_inside["error"])

    def test_deterministic_entry_drift_error_unit(self) -> None:
        from traderbot_ai.tools import replay_helpers

        old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
        old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
        try:
            os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
            os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(
                [
                    {"symbol": BTC, "side": "long", "ref_price": 100.0, "max_drift_pct": 0.02},
                    {"symbol": "ETHUSDT", "side": "short", "ref_price": 100.0, "max_drift_pct": 0.02},
                ]
            )

            adverse_long = replay_helpers.deterministic_entry_drift_error(BTC, "long", 103.0)
            within_long = replay_helpers.deterministic_entry_drift_error(BTC, "long", 101.9)
            favorable_long = replay_helpers.deterministic_entry_drift_error(BTC, "long", 95.0)
            adverse_short = replay_helpers.deterministic_entry_drift_error("ETHUSDT", "short", 97.0)
            favorable_short = replay_helpers.deterministic_entry_drift_error("ETHUSDT", "short", 103.0)
            unknown_symbol = replay_helpers.deterministic_entry_drift_error("SOLUSDT", "long", 103.0)

            os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
            no_ref = replay_helpers.deterministic_entry_drift_error(BTC, "long", 103.0)
        finally:
            _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
            _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)

        self.assertIsNotNone(adverse_long)
        self.assertFalse(adverse_long["ok"])
        self.assertIn("drifted", adverse_long["error"])
        self.assertAlmostEqual(adverse_long["entry_drift_pct"], 0.03)
        self.assertIsNone(within_long)
        self.assertIsNone(favorable_long)
        self.assertIsNotNone(adverse_short)
        self.assertAlmostEqual(adverse_short["entry_drift_pct"], -0.03)
        self.assertIsNone(favorable_short)
        self.assertIsNone(unknown_symbol)
        self.assertIsNone(no_ref)

    def test_deterministic_place_order_rejects_adverse_entry_drift_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_path = tmp_path / "state.json"
            events_path = tmp_path / "events.jsonl"
            cache = LocalMarketCache(tmp_path / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 103.0, 103.5, 102.5, 103.0)])
            old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
            old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(
                    [{"symbol": BTC, "side": "long", "ref_price": 100.0, "max_drift_pct": 0.02}]
                )
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(state_path, events_path, backend="simulated", fee_rate=0.0, cache_path=cache.path, execution_interval="1m"):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    result = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=1.0,
                        takeProfit=110.0,
                        stopLoss=100.5,
                        orderLinkId="drift-reject",
                        as_of=BASE_MS,
                    )
                    event_text = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
            finally:
                _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
                _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("drifted", result["error"])
        event_types = [json.loads(line)["type"] for line in event_text.splitlines() if line.strip()]
        self.assertNotIn("place_order", event_types)

    def test_deterministic_place_order_allows_entry_within_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_path = tmp_path / "state.json"
            events_path = tmp_path / "events.jsonl"
            cache = LocalMarketCache(tmp_path / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 101.5, 102.0, 101.0, 101.5)])
            old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
            old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(
                    [{"symbol": BTC, "side": "long", "ref_price": 100.0, "max_drift_pct": 0.02}]
                )
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(state_path, events_path, backend="simulated", fee_rate=0.0, cache_path=cache.path, execution_interval="1m"):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    result = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=1.0,
                        takeProfit=106.5,
                        stopLoss=99.0,
                        orderLinkId="drift-allow",
                        as_of=BASE_MS,
                    )
                    event_text = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
            finally:
                _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
                _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertTrue(result["ok"], msg=str(result))
        event_types = [json.loads(line)["type"] for line in event_text.splitlines() if line.strip()]
        self.assertIn("place_order", event_types)

    def test_limit_retest_policy_rewrites_market_entry_into_pending_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_path = tmp_path / "state.json"
            events_path = tmp_path / "events.jsonl"
            cache = LocalMarketCache(tmp_path / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 101.5, 102.0, 101.0, 101.5)])
            old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
            old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(
                    [{"symbol": BTC, "side": "long", "ref_price": 100.0, "max_drift_pct": 0.02}]
                )
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(
                    state_path,
                    events_path,
                    backend="simulated",
                    fee_rate=0.0,
                    cache_path=cache.path,
                    execution_interval="1m",
                    entry_policy="limit_retest",
                    retest_pullback=0.4,
                    retest_ttl_min=120,
                ):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    result = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=1.0,
                        takeProfit=106.5,
                        stopLoss=99.0,
                        orderLinkId="retest-rewrite",
                        as_of=BASE_MS,
                    )
                    event_text = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
            finally:
                _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
                _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertTrue(result["ok"], msg=str(result))
        policy_info = result["entry_policy"]
        self.assertEqual(policy_info["policy"], "limit_retest")
        self.assertEqual(policy_info["entry_ref_price"], 101.5)
        # 0.4 * (101.5 - 99.0) = 1.0 below the reference entry
        self.assertAlmostEqual(policy_info["limit_price"], 100.5)
        self.assertEqual(policy_info["expires_at_ms"], BASE_MS + 120 * 60_000)
        order = result["order"]
        self.assertEqual(order["orderType"], "Limit")
        self.assertEqual(order["status"], "New")
        self.assertAlmostEqual(order["price"], 100.5)
        self.assertEqual(order["expires_at_ms"], BASE_MS + 120 * 60_000)
        self.assertEqual(order["entry_policy"], "limit_retest")
        self.assertEqual(order["stopLoss"], 99.0)
        self.assertEqual(order["takeProfit"], 106.5)
        payloads = [json.loads(line) for line in event_text.splitlines() if line.strip()]
        order_events = [item["payload"] for item in payloads if item["type"] == "place_order"]
        self.assertEqual(len(order_events), 1)
        self.assertEqual(order_events[0]["orderType"], "Limit")
        self.assertEqual(order_events[0]["entry_policy"], "limit_retest")

    def test_deterministic_place_order_enforces_volatility_noise_floor_stop(self) -> None:
        one_hour = 3_600_000
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_path = tmp_path / "state.json"
            events_path = tmp_path / "events.jsonl"
            cache = LocalMarketCache(tmp_path / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 100.0, 100.5, 99.5, 100.0)])
            cache.upsert_candles(
                [
                    Candle(
                        symbol=BTC,
                        interval="1h",
                        open_time=BASE_MS - (k + 1) * one_hour,
                        close_time=BASE_MS - k * one_hour - 1,
                        open=100.0,
                        high=100.8,
                        low=99.2,
                        close=100.0,
                        volume=1.0,
                    )
                    for k in range(0, 50)
                ]
            )
            old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
            old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(state_path, events_path, backend="simulated", fee_rate=0.0, cache_path=cache.path, execution_interval="1m"):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    too_tight = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=1.0,
                        takeProfit=104.0,
                        stopLoss=98.4,
                        orderLinkId="noise-tight",
                        as_of=BASE_MS,
                    )
                    wide_enough = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=1.0,
                        takeProfit=106.0,
                        stopLoss=97.3,
                        orderLinkId="noise-wide",
                        as_of=BASE_MS,
                    )
            finally:
                _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
                _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(too_tight["ok"])
        self.assertIn("risk validation failed", too_tight["error"])
        self.assertAlmostEqual(too_tight["risk_validation"]["noise_floor_stop_pct"], 0.024)
        self.assertTrue(any("stop" in error.lower() for error in too_tight["risk_validation"]["errors"]))
        self.assertTrue(wide_enough["ok"], msg=str(wide_enough))
        self.assertAlmostEqual(wide_enough["risk_validation"]["noise_floor_stop_pct"], 0.024)

    def test_write_tools_require_as_of_and_order_link_id(self) -> None:
        missing_as_of = mcp_tools.place_order("linear", BTC, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=96.0, orderLinkId="o1")
        missing_link = mcp_tools.place_order("linear", BTC, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=96.0, as_of=BASE_MS)

        self.assertFalse(missing_as_of["ok"])
        self.assertIn("requires as_of", missing_as_of["error"])
        self.assertFalse(missing_link["ok"])
        self.assertIn("orderLinkId", missing_link["error"])

    def test_place_order_runs_existing_risk_validation_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = LocalMarketCache(tmp_path / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            state_path = tmp_path / "state.json"
            events_path = tmp_path / "events.jsonl"
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(state_path, events_path, backend="simulated", fee_rate=0.001, cache_path=cache.path, execution_interval="1m"):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    mcp_tools.set_leverage("linear", BTC, "1", "1")
                    result = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=100.0,
                        takeProfit=110.0,
                        stopLoss=96.0,
                        orderLinkId="too-large",
                        as_of=BASE_MS,
                    )
            finally:
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("risk validation failed", result["error"])
        self.assertIn("exceeds max position", result["risk_validation"]["errors"][0])
        event_text = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
        event_types = [json.loads(line)["type"] for line in event_text.splitlines()]
        self.assertNotIn("place_order", event_types)

    def test_get_recent_trade_events_filters_by_replay_payload_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            records = [
                {"type": "place_order", "payload": {"symbol": BTC, "created_at_ms": BASE_MS - 10 * 60 * 60_000}},
                {"type": "place_order", "payload": {"symbol": BTC, "created_at_ms": BASE_MS - 60 * 60_000}},
                {"type": "set_leverage", "payload": {"symbol": BTC}},
            ]
            events_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            old_events = os.environ.get("TRADERBOT_EXCHANGE_EVENTS_PATH")
            try:
                os.environ["TRADERBOT_EXCHANGE_EVENTS_PATH"] = str(events_path)
                result = mcp_tools.get_recent_trade_events(BASE_MS, lookback_hours=2)
            finally:
                _restore_env("TRADERBOT_EXCHANGE_EVENTS_PATH", old_events)

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["payload"]["created_at_ms"], BASE_MS - 60 * 60_000)

    def test_get_recent_trade_events_rejects_future_as_of_under_simulation_clock(self) -> None:
        previous_process_clock = process_simulation_clock_ms()
        previous_file_clock = file_simulation_clock_ms()
        try:
            clear_process_simulation_clock_state()
            set_file_simulation_clock_state(BASE_MS)
            result = mcp_tools.get_recent_trade_events(BASE_MS + 1, lookback_hours=2)
        finally:
            if previous_file_clock is None:
                clear_file_simulation_clock_state()
            else:
                set_file_simulation_clock_state(previous_file_clock)
            if previous_process_clock is None:
                clear_process_simulation_clock_state()
            else:
                set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("exceeds simulation clock", result["error"])

    def test_write_tools_use_replay_env_paths_and_compact_exchange_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = LocalMarketCache(tmp_path / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            state_path = tmp_path / "state.json"
            events_path = tmp_path / "events.jsonl"
            audit_path = tmp_path / "audit.jsonl"
            old_audit = os.environ.get("TRADERBOT_MCP_AUDIT_PATH")
            old_run_id = os.environ.get("TRADERBOT_MCP_RUN_ID")
            old_step_id = os.environ.get("TRADERBOT_MCP_STEP_ID")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                os.environ["TRADERBOT_MCP_AUDIT_PATH"] = str(audit_path)
                os.environ["TRADERBOT_MCP_RUN_ID"] = "run-1"
                os.environ["TRADERBOT_MCP_STEP_ID"] = "step-1"
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(state_path, events_path, backend="simulated", fee_rate=0.001, cache_path=cache.path, execution_interval="1m"):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    leverage = mcp_tools.set_leverage("linear", BTC, "1", "1")
                    placed = mcp_tools.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=1.0,
                        takeProfit=110.0,
                        stopLoss=96.0,
                        orderLinkId="run-1-step-1-btc-long",
                        as_of=BASE_MS,
                    )
                    compact_wallet = mcp_tools.get_wallet_compact(BTC, as_of=BASE_MS)
                    event_text = events_path.read_text(encoding="utf-8")
                    audit_text = audit_path.read_text(encoding="utf-8")
            finally:
                _restore_env("TRADERBOT_MCP_AUDIT_PATH", old_audit)
                _restore_env("TRADERBOT_MCP_RUN_ID", old_run_id)
                _restore_env("TRADERBOT_MCP_STEP_ID", old_step_id)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertTrue(leverage["ok"])
        self.assertTrue(placed["ok"], placed)
        self.assertIn("state_summary", placed)
        self.assertNotIn("state", placed)
        self.assertEqual(compact_wallet["open_position_count"], 1)
        self.assertEqual(compact_wallet["equity_usdt"], 1000.0)
        event_types = [json.loads(line)["type"] for line in event_text.splitlines()]
        self.assertIn("place_order", event_types)
        audit_tools = [json.loads(line)["tool"] for line in audit_text.splitlines()]
        self.assertEqual(audit_tools, ["set_leverage", "place_order"])

    def test_file_clock_blocks_future_mcp_market_and_write_reads(self) -> None:
        previous_process_clock = process_simulation_clock_ms()
        previous_file_clock = file_simulation_clock_ms()
        try:
            clear_process_simulation_clock_state()
            set_file_simulation_clock_state(BASE_MS)
            result = mcp_tools.get_current_price(BTC, as_of=BASE_MS + 1)
        finally:
            if previous_file_clock is None:
                clear_file_simulation_clock_state()
            else:
                set_file_simulation_clock_state(previous_file_clock)
            if previous_process_clock is None:
                clear_process_simulation_clock_state()
            else:
                set_process_simulation_clock_state(previous_process_clock)

        self.assertFalse(result["ok"])
        self.assertIn("exceeds simulation clock", result["error"])

    def test_simulation_clock_path_honors_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom_path = Path(tmp) / "custom-clock.json"
            old_clock_path = os.environ.get("TRADERBOT_SIMULATION_CLOCK_PATH")
            previous_process_clock = process_simulation_clock_ms()
            try:
                clear_process_simulation_clock_state()
                os.environ["TRADERBOT_SIMULATION_CLOCK_PATH"] = str(custom_path)
                set_file_simulation_clock_state(BASE_MS)
                self.assertEqual(simulation_clock_path(), custom_path)
                self.assertEqual(file_simulation_clock_ms(), BASE_MS)
                self.assertTrue(custom_path.exists())
                clear_file_simulation_clock_state()
                self.assertFalse(custom_path.exists())
            finally:
                _restore_env("TRADERBOT_SIMULATION_CLOCK_PATH", old_clock_path)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
