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
) -> Candle:
    return Candle(
        symbol=BTC,
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
        self.assertIn("get_candidate_detail", names)
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
        self.assertEqual(result["error"], "no fresh 4h candles at as_of")
        self.assertEqual(result["fresh_symbol_count"], 0)
        self.assertFalse(result["rejected"][0]["data_fresh"])

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
