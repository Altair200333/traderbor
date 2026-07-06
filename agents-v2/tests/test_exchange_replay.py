from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from traderbot_ai.agents.trading import build_tools
from traderbot_ai.cli import build_parser
from traderbot_ai.config import Settings
from traderbot_ai.exchange import SimulatedExchange, SimulatedExchangeBackend, create_exchange_backend
from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.plan import PlanPrimitives
from traderbot_ai.screener.screener import ScanResult, SymbolRow
from traderbot_ai.simulator.clock import (
    clear_file_simulation_clock_state,
    clear_process_simulation_clock_state,
    file_simulation_clock_ms,
    process_simulation_clock_ms,
    set_file_simulation_clock_state,
    set_process_simulation_clock_state,
    set_simulation_clock_state,
)
from traderbot_ai.simulator.exchange_replay import _apply_runner_maintenance, _validate_decision_exchange_consistency, build_exchange_replay_config, exchange_tool_environment, run_exchange_replay
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.tools.exchange import _agent_tool_response, place_order_impl, reset_exchange_impl, set_leverage_impl, settle_exchange_impl


BTC = "BTCUSDT"
ETH = "ETHUSDT"
BASE_MS = 1_704_067_200_000
FOUR_HOURS_MS = 4 * 60 * 60_000


def candle_at_close(symbol: str, close_time: int, open_price: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        symbol=symbol,
        interval="1m",
        open_time=close_time - 59_999,
        close_time=close_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class ExchangeReplayTests(unittest.TestCase):
    def test_exchange_replay_tools_are_cache_only_and_exchange_only(self) -> None:
        settings = Settings(
            model="test",
            vision_model="test",
            reasoning_effort="low",
            paper_balance_usdt=1000.0,
            market_data_mode="live",
            openai_api_key_present=False,
            enable_codex_tool=True,
            disable_tracing=True,
        )

        names = {getattr(tool, "name", "") for tool in build_tools(settings, exchange_replay=True)}

        self.assertIn("get_candles", names)
        self.assertIn("get_current_price", names)
        self.assertIn("scan_momentum_universe", names)
        self.assertIn("get_candidate_detail", names)
        self.assertIn("get_wallet_compact", names)
        self.assertIn("get_open_positions", names)
        self.assertIn("get_recent_trade_events", names)
        self.assertIn("get_wallet", names)
        self.assertIn("place_order", names)
        self.assertIn("cancel_order", names)
        self.assertIn("settle_exchange", names)
        self.assertIn("close_position", names)
        self.assertNotIn("reset_exchange", names)
        self.assertNotIn("place_simulated_order", names)
        self.assertNotIn("get_simulation_portfolio", names)
        self.assertNotIn("paper_place_order", names)
        self.assertNotIn("get_order_book", names)
        self.assertNotIn("codex_code_worker", names)

    def test_deterministic_exchange_replay_tools_hide_broad_scan_raw_candles_and_maintenance_writes(self) -> None:
        settings = Settings(
            model="test",
            vision_model="test",
            reasoning_effort="low",
            paper_balance_usdt=1000.0,
            market_data_mode="live",
            openai_api_key_present=False,
            enable_codex_tool=True,
            disable_tracing=True,
        )

        names = {getattr(tool, "name", "") for tool in build_tools(settings, exchange_replay=True, screener_mode="deterministic")}

        self.assertIn("get_current_price", names)
        self.assertIn("get_setup_digest", names)
        self.assertIn("get_wallet_compact", names)
        self.assertIn("get_recent_trade_events", names)
        self.assertIn("get_wallet", names)
        self.assertIn("place_order", names)
        self.assertIn("set_leverage", names)
        self.assertIn("validate_order", names)
        self.assertIn("calculate_position_size", names)
        self.assertNotIn("get_candles", names)
        self.assertNotIn("scan_momentum_universe", names)
        self.assertNotIn("get_candidate_detail", names)
        self.assertNotIn("close_position", names)
        self.assertNotIn("cancel_order", names)
        self.assertNotIn("settle_exchange", names)

    def test_deterministic_hold_rejects_provider_position_close_events(self) -> None:
        decision = {"final_decision": "hold", "symbol": BTC, "amount": 0.0}
        events = [{"type": "position_closed", "payload": {"symbol": BTC}}]

        _validate_decision_exchange_consistency(decision, events, allow_hold_position_closes=True)
        with self.assertRaisesRegex(RuntimeError, "hold decision produced disallowed exchange events"):
            _validate_decision_exchange_consistency(decision, events, allow_hold_position_closes=False)

    def test_deterministic_entry_validation_requires_runner_candidate_and_rejects_extra_writes(self) -> None:
        decision = {
            "final_decision": "long",
            "symbol": BTC,
            "amount": 100.0,
            "stop_loss": 96.0,
            "take_profit": 110.0,
        }
        place_order_event = {
            "type": "place_order",
            "payload": {
                "category": "linear",
                "symbol": BTC,
                "side": "Buy",
                "orderType": "Market",
                "status": "Filled",
                "qty": 1.0,
                "price": 100.0,
                "notional_usdt": 100.0,
                "stopLoss": 96.0,
                "takeProfit": 110.0,
                "position": {"category": "linear", "notional_usdt": 100.0},
            },
        }
        leverage_event = {"type": "set_leverage", "payload": {"category": "linear", "symbol": BTC}}

        _validate_decision_exchange_consistency(
            decision,
            [leverage_event, place_order_event],
            allowed_entry_candidates={(BTC, "long")},
            strict_entry_events=True,
        )
        with self.assertRaisesRegex(RuntimeError, "not in deterministic screener candidates"):
            _validate_decision_exchange_consistency(
                decision,
                [place_order_event],
                allowed_entry_candidates={(ETH, "long")},
                strict_entry_events=True,
            )
        with self.assertRaisesRegex(RuntimeError, "produced disallowed exchange events"):
            _validate_decision_exchange_consistency(
                decision,
                [place_order_event, {"type": "position_closed", "payload": {"symbol": ETH}}],
                allowed_entry_candidates={(BTC, "long")},
                strict_entry_events=True,
            )

    def test_runner_maintenance_surfaces_impulse_break_data_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            wallet = {
                "open_positions": [
                    {
                        "symbol": BTC,
                        "side": "Buy",
                        "opened_at_ms": BASE_MS,
                        "position_id": "p1",
                    }
                ]
            }

            actions, warnings = _apply_runner_maintenance(exchange, cache, wallet, BASE_MS + FOUR_HOURS_MS, 0.0, ScreenerConfig())

        self.assertEqual(actions, [])
        self.assertTrue(any("impulse_break_unchecked" in warning for warning in warnings))

    def test_exchange_replay_cli_has_hold_decision_mode(self) -> None:
        args = build_parser().parse_args(
            [
                "exchange-replay",
                "--start-time",
                "2024-01-01T00:00:00Z",
                "--end-time",
                "2024-01-01T04:00:00Z",
                "--decision-mode",
                "hold",
            ]
        )

        self.assertEqual(args.command, "exchange-replay")
        self.assertEqual(args.decision_mode, "hold")

    def test_exchange_replay_cli_has_screener_mode(self) -> None:
        args = build_parser().parse_args(
            [
                "exchange-replay",
                "--start-time",
                "2024-01-01T00:00:00Z",
                "--end-time",
                "2024-01-01T04:00:00Z",
                "--screener-mode",
                "deterministic",
            ]
        )

        self.assertEqual(args.screener_mode, "deterministic")

    def test_agent_tool_response_compacts_state_and_reports_runtime_overrides(self) -> None:
        old_fee = os.environ.get("TRADERBOT_EXCHANGE_FEE_RATE")
        old_interval = os.environ.get("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL")
        try:
            os.environ["TRADERBOT_EXCHANGE_FEE_RATE"] = "0.001"
            os.environ["TRADERBOT_EXCHANGE_EXECUTION_INTERVAL"] = "1m"

            result = _agent_tool_response(
                {
                    "ok": True,
                    "state": {
                        "as_of_ms": BASE_MS,
                        "balances": {"USDT": {"free": 1000.0, "locked": 0.0}},
                        "orders": [{"order_id": "o1"}],
                        "positions": [{"position_id": "p1"}],
                        "closed_positions": [{"position_id": "p0"}],
                    },
                }
            )
            reset_result = _agent_tool_response(
                {
                    "ok": True,
                    "balances": {"USDT": {"free": 1000.0, "locked": 0.0}},
                    "orders": [],
                    "positions": [],
                    "closed_positions": [],
                    "as_of_ms": BASE_MS,
                }
            )
        finally:
            if old_fee is None:
                os.environ.pop("TRADERBOT_EXCHANGE_FEE_RATE", None)
            else:
                os.environ["TRADERBOT_EXCHANGE_FEE_RATE"] = old_fee
            if old_interval is None:
                os.environ.pop("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", None)
            else:
                os.environ["TRADERBOT_EXCHANGE_EXECUTION_INTERVAL"] = old_interval

        self.assertNotIn("state", result)
        self.assertEqual(result["state_summary"]["as_of_ms"], BASE_MS)
        self.assertEqual(result["state_summary"]["open_order_count"], 1)
        self.assertEqual(result["state_summary"]["open_position_count"], 1)
        self.assertEqual(result["state_summary"]["closed_position_count"], 1)
        self.assertEqual(result["runtime_overrides"]["fee_rate"]["value"], "0.001")
        self.assertEqual(result["runtime_overrides"]["execution_interval"]["value"], "1m")
        self.assertNotIn("balances", reset_result)
        self.assertEqual(reset_result["state_summary"]["balances"]["USDT"]["free"], 1000.0)

    def test_exchange_tool_environment_uses_run_specific_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            events_path = Path(tmp) / "events.jsonl"

            with exchange_tool_environment(state_path, events_path):
                result = reset_exchange_impl('{"USDT": 123}', as_of=BASE_MS)

            self.assertTrue(result["ok"])
            self.assertTrue(state_path.exists())
            self.assertTrue(events_path.exists())
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["balances"]["USDT"]["free"], 123.0)

    def test_exchange_backend_factory_defaults_to_simulated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = create_exchange_backend(state_path=Path(tmp) / "state.json", events_path=Path(tmp) / "events.jsonl")

            self.assertIsInstance(backend, SimulatedExchangeBackend)

    def test_exchange_backend_factory_rejects_unknown_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(NotImplementedError):
                create_exchange_backend("bybit", state_path=Path(tmp) / "state.json", events_path=Path(tmp) / "events.jsonl")

    def test_build_exchange_replay_config_rejects_non_finite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            build_exchange_replay_config([BTC], BASE_MS, BASE_MS + FOUR_HOURS_MS, fee_rate=float("nan"))
        with self.assertRaises(ValueError):
            build_exchange_replay_config([BTC], BASE_MS, BASE_MS + FOUR_HOURS_MS, linear_leverage=float("inf"))

    def test_build_exchange_replay_config_preserves_explicit_empty_balances(self) -> None:
        config = build_exchange_replay_config([BTC], BASE_MS, BASE_MS + FOUR_HOURS_MS, balances={})

        self.assertEqual(config.balances, {})

    def test_exchange_tool_environment_clears_optional_outer_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            events_path = Path(tmp) / "events.jsonl"
            old_backend = os.environ.get("TRADERBOT_EXCHANGE_BACKEND")
            old_fee = os.environ.get("TRADERBOT_EXCHANGE_FEE_RATE")
            old_interval = os.environ.get("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL")
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            os.environ["TRADERBOT_EXCHANGE_BACKEND"] = "bybit"
            os.environ["TRADERBOT_EXCHANGE_FEE_RATE"] = "0.5"
            os.environ["TRADERBOT_EXCHANGE_EXECUTION_INTERVAL"] = "4h"
            os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(Path(tmp) / "outer.sqlite3")
            try:
                with exchange_tool_environment(state_path, events_path):
                    self.assertNotIn("TRADERBOT_EXCHANGE_BACKEND", os.environ)
                    self.assertNotIn("TRADERBOT_EXCHANGE_FEE_RATE", os.environ)
                    self.assertNotIn("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", os.environ)
                    self.assertNotIn("TRADERBOT_MARKET_CACHE_PATH", os.environ)
            finally:
                if old_backend is None:
                    os.environ.pop("TRADERBOT_EXCHANGE_BACKEND", None)
                else:
                    os.environ["TRADERBOT_EXCHANGE_BACKEND"] = old_backend
                if old_fee is None:
                    os.environ.pop("TRADERBOT_EXCHANGE_FEE_RATE", None)
                else:
                    os.environ["TRADERBOT_EXCHANGE_FEE_RATE"] = old_fee
                if old_interval is None:
                    os.environ.pop("TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", None)
                else:
                    os.environ["TRADERBOT_EXCHANGE_EXECUTION_INTERVAL"] = old_interval
                if old_cache is None:
                    os.environ.pop("TRADERBOT_MARKET_CACHE_PATH", None)
                else:
                    os.environ["TRADERBOT_MARKET_CACHE_PATH"] = old_cache

    def test_empty_env_overrides_still_validate_tool_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            events_path = Path(tmp) / "events.jsonl"
            with exchange_tool_environment(state_path, events_path):
                os.environ["TRADERBOT_EXCHANGE_FEE_RATE"] = ""
                os.environ["TRADERBOT_EXCHANGE_EXECUTION_INTERVAL"] = ""
                bad_fee = settle_exchange_impl(as_of=BASE_MS, fee_rate=-1.0)
                nan_fee = settle_exchange_impl(as_of=BASE_MS, fee_rate=float("nan"))
                bad_interval = settle_exchange_impl(as_of=BASE_MS, interval="bad")

        self.assertFalse(bad_fee["ok"])
        self.assertIn("non-negative", bad_fee["error"])
        self.assertFalse(nan_fee["ok"])
        self.assertIn("non-negative", nan_fee["error"])
        self.assertFalse(bad_interval["ok"])
        self.assertIn("unsupported", bad_interval["error"])

    def test_exchange_tools_reject_future_as_of_under_simulation_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            events_path = Path(tmp) / "events.jsonl"
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                set_simulation_clock_state(BASE_MS)
                with exchange_tool_environment(state_path, events_path):
                    reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                    result = place_order_impl("spot", BTC, "Buy", "Market", qty=100.0, as_of=BASE_MS + 1)
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

    def test_exchange_tool_environment_applies_fee_rate_and_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle_at_close(BTC, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle_at_close(BTC, BASE_MS + FOUR_HOURS_MS, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            state_path = Path(tmp) / "state.json"
            events_path = Path(tmp) / "events.jsonl"

            with exchange_tool_environment(state_path, events_path, fee_rate=0.001, cache_path=cache.path):
                reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                set_leverage_impl("linear", BTC, "5", "5")
                placed = place_order_impl("linear", BTC, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=95.0, fee_rate=1e-9, as_of=BASE_MS)
                settled = settle_exchange_impl(as_of=BASE_MS + FOUR_HOURS_MS, fee_rate=1e-9)

            self.assertTrue(placed["ok"])
            self.assertTrue(settled["ok"])
            self.assertAlmostEqual(settled["closed_positions"][0]["fees_usdt"], 0.21)
            self.assertAlmostEqual(settled["state"]["balances"]["USDT"]["free"], 1009.79)

    def test_exchange_tool_environment_applies_fee_rate_to_spot_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BTC, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            state_path = Path(tmp) / "state.json"
            events_path = Path(tmp) / "events.jsonl"

            with exchange_tool_environment(state_path, events_path, fee_rate=0.001, cache_path=cache.path):
                reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                placed = place_order_impl("spot", BTC, "Buy", "Market", qty=500.0, fee_rate=1e-9, as_of=BASE_MS)

            self.assertTrue(placed["ok"])
            self.assertEqual(placed["order"]["fee_asset"], "BTC")
            self.assertAlmostEqual(placed["order"]["fee_amount"], 0.00001)
            self.assertAlmostEqual(placed["state"]["balances"]["BTC"]["free"], 0.00999)

    def test_exchange_tool_environment_pins_execution_interval_for_place_order_advance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BTC, BASE_MS + 60_000, 100.0, 101.0, 99.0, 100.0)])
            state_path = Path(tmp) / "state.json"
            events_path = Path(tmp) / "events.jsonl"

            with exchange_tool_environment(state_path, events_path, cache_path=cache.path, execution_interval="1m"):
                reset_exchange_impl('{"USDT": 1000}', as_of=BASE_MS)
                first = place_order_impl("spot", BTC, "Buy", "Limit", qty=2.0, price=100.0, as_of=BASE_MS)
                second = place_order_impl("spot", BTC, "Sell", "Limit", qty=1.0, price=200.0, mark_interval="4h", as_of=BASE_MS + 60_000)

            self.assertTrue(first["ok"])
            self.assertTrue(second["ok"])
            self.assertEqual(len(second["advanced_before_order"]["filled_orders"]), 1)
            self.assertEqual(second["state"]["balances"]["USDT"]["free"], 800.0)
            self.assertEqual(second["state"]["balances"]["BTC"]["free"], 1.0)
            self.assertEqual(second["state"]["balances"]["BTC"]["locked"], 1.0)

    def test_run_exchange_replay_steps_agent_over_multi_symbol_wallet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle_at_close(BTC, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle_at_close(ETH, BASE_MS, 10.0, 11.0, 9.0, 10.0),
                    candle_at_close(BTC, BASE_MS + FOUR_HOURS_MS, 100.0, 112.0, 99.0, 111.0),
                    candle_at_close(ETH, BASE_MS + FOUR_HOURS_MS, 10.0, 10.5, 9.5, 10.0),
                    candle_at_close(BTC, BASE_MS + 2 * FOUR_HOURS_MS, 111.0, 112.0, 110.0, 111.0),
                    candle_at_close(ETH, BASE_MS + 2 * FOUR_HOURS_MS, 10.0, 10.5, 9.5, 10.0),
                ]
            )
            state_path = Path(tmp) / "exchange.json"
            events_path = Path(tmp) / "exchange-events.jsonl"
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC, ETH],
                start_time=BASE_MS,
                end_time=BASE_MS + 2 * FOUR_HOURS_MS,
                decision_interval="4h",
                execution_interval="1m",
                balance_usdt=1000.0,
                fee_rate=0.001,
                state_path=state_path,
                events_path=events_path,
                replay_path=replay_path,
                run_id="test-replay",
            )
            exchange = SimulatedExchange(state_path, events_path, cache=cache)
            calls = []

            def decide(context: dict) -> dict:
                calls.append(context["as_of_ms"])
                if len(calls) == 1:
                    exchange.set_leverage("linear", BTC, "5", "5")
                    exchange.place_order(
                        "linear",
                        BTC,
                        "Buy",
                        "Market",
                        qty=1.0,
                        takeProfit=110.0,
                        stopLoss=96.0,
                        fee_rate=context["fee_rate"],
                        as_of=context["as_of_ms"],
                    )
                    return {"final_decision": "long", "symbol": BTC, "amount": 100.0, "stop_loss": 96.0, "take_profit": 110.0}
                return {"final_decision": "hold", "symbol": BTC, "amount": 0.0}

            result = run_exchange_replay(config=config, decide=decide, cache=cache, exchange=exchange)

            self.assertTrue(result["ok"])
            self.assertEqual(calls, [BASE_MS, BASE_MS + FOUR_HOURS_MS])
            self.assertEqual(len(result["steps"]), 2)
            self.assertEqual(result["steps"][0]["wallet_after"]["open_positions"][0]["symbol"], BTC)
            self.assertIn("place_order", [event["type"] for event in result["steps"][0]["agent_exchange_events"]])
            self.assertEqual(result["steps"][1]["settlement"]["closed_positions"][0]["exit_reason"], "TP")
            self.assertAlmostEqual(result["final_wallet"]["totals"]["free_usdt"], 1009.79)
            self.assertTrue(replay_path.exists())
            replay_events = [json.loads(line)["type"] for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(replay_events, ["replay_started", "step_started", "step_completed", "step_started", "step_completed", "replay_completed"])
            exchange_events = [json.loads(line)["type"] for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertIn("place_order", exchange_events)
            self.assertIn("position_closed", exchange_events)

    def test_run_exchange_replay_reuses_passed_exchange_cache_when_cache_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BTC, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            state_path = Path(tmp) / "exchange.json"
            events_path = Path(tmp) / "exchange-events.jsonl"
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                decision_interval="4h",
                execution_interval="1m",
                balance_usdt=1000.0,
                state_path=state_path,
                events_path=events_path,
                replay_path=replay_path,
                run_id="exchange-cache-inheritance",
            )
            exchange = SimulatedExchange(state_path, events_path, cache=cache)

            def decide(context: dict) -> dict:
                placed = place_order_impl("spot", BTC, "Buy", "Market", qty=500.0, as_of=context["as_of_ms"])
                return {"placed_ok": placed["ok"], "error": placed.get("error")}

            result = run_exchange_replay(config=config, decide=decide, exchange=exchange)

            self.assertTrue(result["ok"])
            self.assertTrue(result["steps"][0]["decision"]["placed_ok"])
            self.assertEqual(result["final_wallet"]["balances"][0]["asset"], "BTC")

    def test_run_exchange_replay_uses_env_market_cache_when_cache_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "env-market.sqlite3"
            cache = LocalMarketCache(cache_path)
            cache.upsert_candles([candle_at_close(BTC, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            state_path = Path(tmp) / "exchange.json"
            events_path = Path(tmp) / "exchange-events.jsonl"
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                decision_interval="4h",
                execution_interval="1m",
                balance_usdt=1000.0,
                state_path=state_path,
                events_path=events_path,
                replay_path=replay_path,
                run_id="env-cache",
            )
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache_path)
            try:
                result = run_exchange_replay(config=config, decide=lambda context: {"final_decision": "hold"})
            finally:
                if old_cache is None:
                    os.environ.pop("TRADERBOT_MARKET_CACHE_PATH", None)
                else:
                    os.environ["TRADERBOT_MARKET_CACHE_PATH"] = old_cache

            self.assertTrue(result["ok"])
            self.assertEqual(result["final_wallet"]["balances"][0]["asset"], "BTC")

    def test_run_exchange_replay_deterministic_screener_auto_hold_skips_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BTC, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=Path(tmp) / "exchange.json",
                events_path=Path(tmp) / "exchange-events.jsonl",
                replay_path=Path(tmp) / "replay.jsonl",
                screener_mode="deterministic",
                run_id="deterministic-empty",
            )
            scan_result = _scan_result([SymbolRow(symbol=BTC, status="insufficient_data", failed_gates=["insufficient_data"])], candidates=[])

            def decide(_: dict) -> dict:
                raise AssertionError("provider should not be called")

            with patch("traderbot_ai.simulator.exchange_replay.run_screener", return_value=scan_result), patch(
                "traderbot_ai.simulator.exchange_replay.write_scan_artifacts",
                return_value={"artifact_path": str(Path(tmp) / "scan.json"), "sha256": "hash"},
            ):
                result = run_exchange_replay(config=config, decide=decide, cache=cache)

        self.assertTrue(result["ok"])
        self.assertEqual(result["steps"][0]["decision"]["final_decision"], "hold")
        self.assertEqual(result["steps"][0]["decision"]["scan_hash"], "hash")
        self.assertEqual(result["steps"][0]["scan"]["scan_hash"], "hash")
        self.assertEqual(result["steps"][0]["agent_exchange_events"], [])

    def test_run_exchange_replay_deterministic_screener_passes_context_when_candidate_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BTC, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=Path(tmp) / "exchange.json",
                events_path=Path(tmp) / "exchange-events.jsonl",
                replay_path=Path(tmp) / "replay.jsonl",
                screener_mode="deterministic",
                run_id="deterministic-candidate",
            )
            row = SymbolRow(
                symbol=BTC,
                status="ok",
                close=100.0,
                signal_candidate_before_state="long",
                candidate="long",
                plan=PlanPrimitives(
                    pattern_used="P1",
                    boundary_price=99.0,
                    invalidation_price=98.4,
                    d_atr=0.02,
                    d_struct=0.016,
                    d_final=0.02,
                    stop_feasible=True,
                    tp_rr_default=2.5,
                    ref_entry=100.0,
                ),
            )
            scan_result = _scan_result([row], candidates=[BTC])
            contexts = []
            old_candidates_env = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")

            def decide(context: dict) -> dict:
                contexts.append(context)
                self.assertEqual(os.environ.get("TRADERBOT_SCREENER_MODE"), "deterministic")
                self.assertIn(BTC, os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES", ""))
                return {"final_decision": "hold", "symbol": BTC, "amount": 0.0}

            with patch("traderbot_ai.simulator.exchange_replay.run_screener", return_value=scan_result), patch(
                "traderbot_ai.simulator.exchange_replay.write_scan_artifacts",
                return_value={"artifact_path": str(Path(tmp) / "scan.json"), "sha256": "hash"},
            ):
                result = run_exchange_replay(config=config, decide=decide, cache=cache)

        self.assertTrue(result["ok"])
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["screener_mode"], "deterministic")
        self.assertIn("scan_markdown", contexts[0])
        self.assertEqual(contexts[0]["scan_hash"], "hash")
        self.assertEqual(contexts[0]["screener_candidates"], [BTC])
        self.assertEqual(contexts[0]["candidate_primitives"][0]["side"], "long")
        self.assertEqual(os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES"), old_candidates_env)

    def test_deterministic_place_order_impl_rejects_non_candidate_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "exchange.json"
            events_path = Path(tmp) / "exchange-events.jsonl"
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
                    result = place_order_impl(
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
                if old_mode is None:
                    os.environ.pop("TRADERBOT_SCREENER_MODE", None)
                else:
                    os.environ["TRADERBOT_SCREENER_MODE"] = old_mode
                if old_candidates is None:
                    os.environ.pop("TRADERBOT_DETERMINISTIC_CANDIDATES", None)
                else:
                    os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = old_candidates
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

    def test_run_exchange_replay_deterministic_runner_max_hold_close_is_not_agent_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle_at_close(BTC, BASE_MS + minute * 60_000, 100.0, 100.2, 99.8, 100.0)
                    for minute in range(0, 28 * 60 + 1)
                ]
            )
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + 7 * FOUR_HOURS_MS,
                state_path=Path(tmp) / "exchange.json",
                events_path=Path(tmp) / "exchange-events.jsonl",
                replay_path=Path(tmp) / "replay.jsonl",
                screener_mode="deterministic",
                run_id="deterministic-max-hold",
            )
            row = SymbolRow(
                symbol=BTC,
                status="ok",
                close=100.0,
                signal_candidate_before_state="long",
                candidate="long",
                plan=PlanPrimitives(
                    pattern_used="P2",
                    boundary_price=99.0,
                    invalidation_price=98.0,
                    d_atr=0.02,
                    d_struct=0.02,
                    d_final=0.02,
                    stop_feasible=True,
                    tp_rr_default=2.0,
                    ref_entry=100.0,
                ),
            )
            candidate_scan = _scan_result([row], candidates=[BTC])
            empty_scan = _scan_result([SymbolRow(symbol=BTC, status="ok", close=100.0)], candidates=[])
            scans = [candidate_scan, empty_scan, empty_scan, empty_scan, empty_scan, empty_scan, empty_scan]
            provider_calls = []

            def decide(context: dict) -> dict:
                provider_calls.append(context["as_of_ms"])
                place_order_impl(
                    "linear",
                    BTC,
                    "Buy",
                    "Market",
                    qty=1.0,
                    takeProfit=104.0,
                    stopLoss=98.0,
                    orderLinkId="deterministic-max-hold-btc-long",
                    as_of=context["as_of_ms"],
                    fee_rate=context["fee_rate"],
                )
                return {
                    "final_decision": "long",
                    "symbol": BTC,
                    "timeframe": "4h",
                    "thesis": "test entry",
                    "price": 100.0,
                    "stop_loss": 98.0,
                    "take_profit": 104.0,
                    "amount": 100.0,
                    "confidence": 0.7,
                    "risk_summary": "test",
                    "tool_summary": ["place_order"],
                    "worklog_path": None,
                }

            with patch("traderbot_ai.simulator.exchange_replay.run_screener", side_effect=scans), patch(
                "traderbot_ai.simulator.exchange_replay.write_scan_artifacts",
                return_value={"artifact_path": str(Path(tmp) / "scan.json"), "sha256": "hash"},
            ):
                result = run_exchange_replay(config=config, decide=decide, cache=cache)

        self.assertTrue(result["ok"])
        self.assertEqual(provider_calls, [BASE_MS])
        max_hold_step = result["steps"][6]
        self.assertEqual(max_hold_step["maintenance_actions"][0]["reason"], "max_hold")
        self.assertEqual([event["type"] for event in max_hold_step["maintenance_exchange_events"]], ["position_closed"])
        self.assertEqual(max_hold_step["agent_exchange_events"], [])
        self.assertEqual(max_hold_step["decision"]["final_decision"], "hold")
        self.assertEqual(result["final_wallet"]["open_positions"], [])

    def test_run_exchange_replay_rejects_mismatched_exchange_and_replay_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange_cache = LocalMarketCache(Path(tmp) / "exchange-market.sqlite3")
            replay_cache = LocalMarketCache(Path(tmp) / "replay-market.sqlite3")
            state_path = Path(tmp) / "exchange.json"
            events_path = Path(tmp) / "exchange-events.jsonl"
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=state_path,
                events_path=events_path,
                replay_path=replay_path,
                run_id="mismatched-cache",
            )
            exchange = SimulatedExchange(state_path, events_path, cache=exchange_cache)

            with self.assertRaises(ValueError):
                run_exchange_replay(config=config, decide=lambda context: {}, cache=replay_cache, exchange=exchange)

    def test_run_exchange_replay_rejects_mismatched_exchange_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=Path(tmp) / "config-exchange.json",
                events_path=Path(tmp) / "config-events.jsonl",
                replay_path=Path(tmp) / "replay.jsonl",
                run_id="mismatched-paths",
            )
            exchange = SimulatedExchange(Path(tmp) / "actual-exchange.json", Path(tmp) / "actual-events.jsonl", cache=cache)

            with self.assertRaises(ValueError):
                run_exchange_replay(config=config, decide=lambda context: {}, cache=cache, exchange=exchange)

    def test_run_exchange_replay_writes_failed_event_on_decide_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle_at_close(BTC, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle_at_close(ETH, BASE_MS, 10.0, 11.0, 9.0, 10.0),
                ]
            )
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC, ETH],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=Path(tmp) / "exchange.json",
                events_path=Path(tmp) / "exchange-events.jsonl",
                replay_path=replay_path,
                run_id="test-failure",
            )

            def decide(_: dict) -> dict:
                raise RuntimeError("model failed")

            with self.assertRaises(RuntimeError):
                run_exchange_replay(config=config, decide=decide, cache=cache)

            replay_events = [json.loads(line)["type"] for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(replay_events[-1], "replay_failed")

    def test_run_exchange_replay_rejects_trade_decision_without_place_order_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=Path(tmp) / "exchange.json",
                events_path=Path(tmp) / "exchange-events.jsonl",
                replay_path=replay_path,
            )

            with self.assertRaisesRegex(RuntimeError, "did not produce a place_order"):
                run_exchange_replay(config=config, decide=lambda context: {"final_decision": "long", "symbol": BTC, "amount": 100.0})

            replay_events = [json.loads(line)["type"] for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(replay_events[-1], "replay_failed")

    def test_run_exchange_replay_rejects_hold_with_non_maintenance_write_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=Path(tmp) / "exchange.json",
                events_path=Path(tmp) / "exchange-events.jsonl",
                replay_path=replay_path,
            )

            def decide(_: dict) -> dict:
                set_leverage_impl("linear", BTC, "1", "1")
                return {"final_decision": "hold", "symbol": BTC, "amount": 0.0}

            with self.assertRaisesRegex(RuntimeError, "hold decision produced disallowed exchange events"):
                run_exchange_replay(config=config, decide=decide)

            replay_events = [json.loads(line)["type"] for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(replay_events[-1], "replay_failed")

    def test_run_exchange_replay_rejects_decision_that_mismatches_place_order_payload(self) -> None:
        cases = [
            ("wrong_symbol", {"final_decision": "long", "symbol": ETH, "amount": 100.0}),
            ("wrong_side", {"final_decision": "short", "symbol": BTC, "amount": 100.0}),
            ("wrong_amount", {"final_decision": "long", "symbol": BTC, "amount": 250.0}),
            ("wrong_take_profit", {"final_decision": "long", "symbol": BTC, "amount": 100.0, "take_profit": 111.0}),
        ]
        for name, decision in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
                    cache.upsert_candles(
                        [
                            candle_at_close(BTC, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                            candle_at_close(ETH, BASE_MS, 10.0, 11.0, 9.0, 10.0),
                        ]
                    )
                    state_path = Path(tmp) / "exchange.json"
                    events_path = Path(tmp) / "exchange-events.jsonl"
                    replay_path = Path(tmp) / "replay.jsonl"
                    config = build_exchange_replay_config(
                        symbols=[BTC, ETH],
                        start_time=BASE_MS,
                        end_time=BASE_MS + FOUR_HOURS_MS,
                        state_path=state_path,
                        events_path=events_path,
                        replay_path=replay_path,
                    )
                    exchange = SimulatedExchange(state_path, events_path, cache=cache)

                    def decide(context: dict) -> dict:
                        exchange.set_leverage("linear", BTC, "1", "1")
                        exchange.place_order(
                            "linear",
                            BTC,
                            "Buy",
                            "Market",
                            qty=1.0,
                            takeProfit=110.0,
                            stopLoss=96.0,
                            fee_rate=context["fee_rate"],
                            as_of=context["as_of_ms"],
                        )
                        return decision

                    with self.assertRaisesRegex(RuntimeError, "does not match the place_order"):
                        run_exchange_replay(config=config, decide=decide, cache=cache, exchange=exchange)

                    replay_events = [json.loads(line)["type"] for line in replay_path.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(replay_events[-1], "replay_failed")

    def test_run_exchange_replay_rejects_spot_order_for_trade_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle_at_close(BTC, BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            state_path = Path(tmp) / "exchange.json"
            events_path = Path(tmp) / "exchange-events.jsonl"
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=state_path,
                events_path=events_path,
                replay_path=replay_path,
            )
            exchange = SimulatedExchange(state_path, events_path, cache=cache)

            def decide(context: dict) -> dict:
                exchange.place_order("spot", BTC, "Buy", "Market", qty=100.0, marketUnit="quoteCoin", as_of=context["as_of_ms"])
                return {
                    "final_decision": "long",
                    "symbol": BTC,
                    "amount": 100.0,
                    "stop_loss": 96.0,
                    "take_profit": 110.0,
                }

            with self.assertRaisesRegex(RuntimeError, "does not match the place_order"):
                run_exchange_replay(config=config, decide=decide, cache=cache, exchange=exchange)

            replay_events = [json.loads(line)["type"] for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(replay_events[-1], "replay_failed")

    def test_run_exchange_replay_allows_failed_mcp_write_attempt_without_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            audit_path.write_text(
                json.dumps({"tool": "place_order", "result": {"ok": False, "error": "risk validation failed"}}) + "\n",
                encoding="utf-8",
            )
            replay_path = Path(tmp) / "replay.jsonl"
            config = build_exchange_replay_config(
                symbols=[BTC],
                start_time=BASE_MS,
                end_time=BASE_MS + FOUR_HOURS_MS,
                state_path=Path(tmp) / "exchange.json",
                events_path=Path(tmp) / "exchange-events.jsonl",
                replay_path=replay_path,
            )

            def decide(_: dict) -> dict:
                return {"final_decision": "hold", "symbol": BTC, "amount": 0.0, "codex_mcp_audit_path": str(audit_path)}

            result = run_exchange_replay(config=config, decide=decide)

            replay_events = [json.loads(line)["type"] for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(result["ok"])
            self.assertEqual(replay_events[-1], "replay_completed")


def _scan_result(rows: list[SymbolRow], candidates: list[str]) -> ScanResult:
    return ScanResult(
        as_of_ms=BASE_MS,
        as_of_iso="2024-01-01T00:00:00+00:00",
        screener_version="test",
        config_hash="cfg",
        symbols=rows,
        candidates=candidates,
        global_blocks=[],
        btc_roc_4h=0.0,
        data_warnings=[],
    )


if __name__ == "__main__":
    unittest.main()
