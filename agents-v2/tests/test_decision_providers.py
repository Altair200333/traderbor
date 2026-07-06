from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from traderbot_ai.config import Settings
from traderbot_ai.decision.codex_cli_provider import CodexCliMcpDecisionProvider, CodexCliOptions, _trade_decision_output_schema
from traderbot_ai.decision.hold_provider import HoldDecisionProvider
from traderbot_ai.decision.openai_agents_provider import OpenAIAgentsDecisionProvider
from traderbot_ai.decision.prompts import build_exchange_replay_prompt


def settings() -> Settings:
    return Settings(
        model="gpt-5.5",
        vision_model="gpt-5.5",
        reasoning_effort="medium",
        paper_balance_usdt=1000.0,
        market_data_mode="cache",
        openai_api_key_present=False,
        enable_codex_tool=False,
        disable_tracing=True,
    )


def replay_context() -> dict:
    return {
        "run_id": "run-1",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "as_of_ms": 1_704_067_200_000,
        "as_of_iso": "2024-01-01T00:00:00+00:00",
        "next_as_of_ms": 1_704_081_600_000,
        "decision_interval": "4h",
        "execution_interval": "1m",
        "fee_rate": 0.001,
        "wallet": {"ok": True, "total_equity_usdt": 1000.0, "positions": []},
        "settlement": {"ok": True, "closed_positions": []},
        "exchange_state_path": "state.json",
        "exchange_events_path": "events.jsonl",
        "replay_path": "replay.jsonl",
    }


def deterministic_replay_context() -> dict:
    context = replay_context()
    context.update(
        {
            "screener_mode": "deterministic",
            "maintenance_actions": [],
            "scan_artifact_path": "worklog/screener/run-1/1704067200000.json",
            "scan_hash": "sha256-test",
            "candidate_primitives": [
                {
                    "symbol": "ETHUSDT",
                    "side": "long",
                    "signal_candidate_before_state": "long",
                    "plan": {
                        "pattern": "P2",
                        "entry_price": 100.0,
                        "stop_loss": 98.0,
                        "take_profit": 104.0,
                        "stop_distance_pct": 0.02,
                        "tp_rr": 2.0,
                    },
                }
            ],
            "screener_candidates": ["ETHUSDT"],
            "global_blocks": [],
            "data_warnings": [],
            "scan_markdown": "| symbol | candidate | pattern |\n| ETHUSDT | long | P2 |",
        }
    )
    return context


class DecisionProviderTests(unittest.TestCase):
    def test_hold_provider_preserves_legacy_partial_shape(self) -> None:
        output = HoldDecisionProvider().decide(replay_context())

        self.assertEqual(output["final_decision"], "hold")
        self.assertEqual(output["symbol"], "BTCUSDT")
        self.assertEqual(output["amount"], 0.0)
        self.assertNotIn("timeframe", output)

    def test_exchange_replay_prompt_contains_contract_fields(self) -> None:
        prompt = build_exchange_replay_prompt(replay_context())

        self.assertIn("Run id: run-1", prompt)
        self.assertIn("Symbols: BTCUSDT, ETHUSDT", prompt)
        self.assertIn("Wallet:", prompt)
        self.assertIn("Settlement just applied:", prompt)
        self.assertIn("scan_momentum_universe", prompt)
        self.assertIn("get_candidate_detail", prompt)
        self.assertIn("Never request 1m candles for signal analysis.", prompt)
        self.assertIn("Never call exchange write tools on hold paths except mandatory position-maintenance close_position calls.", prompt)
        self.assertIn("Do not call settle_exchange", prompt)
        self.assertIn("place_order.qty is base-asset quantity", prompt)

    def test_deterministic_exchange_replay_prompt_uses_runner_scan(self) -> None:
        prompt = build_exchange_replay_prompt(deterministic_replay_context())

        self.assertIn("Screener mode: deterministic.", prompt)
        self.assertIn("sha256-test", prompt)
        self.assertIn("| ETHUSDT | long | P2 |", prompt)
        self.assertIn("Do not call scan_momentum_universe", prompt)
        self.assertIn("use get_setup_digest", prompt)
        self.assertIn("Never call close_position, cancel_order, or settle_exchange", prompt)
        self.assertNotIn("coarse-scan symbols", prompt)
        self.assertNotIn("Use raw get_candles only", prompt)

    def test_openai_provider_passes_screener_mode_to_agent_builder(self) -> None:
        with patch("traderbot_ai.decision.openai_agents_provider.build_trading_agent", return_value=object()) as build_agent:
            OpenAIAgentsDecisionProvider(settings=settings(), session_name="session", max_turns=7, screener_mode="deterministic")

        build_agent.assert_called_once()
        self.assertEqual(build_agent.call_args.kwargs["screener_mode"], "deterministic")

    def test_openai_provider_preserves_step_session_and_run_log(self) -> None:
        provider = OpenAIAgentsDecisionProvider(settings=settings(), session_name="session", max_turns=7)
        fake_session = SimpleNamespace(close=lambda: None)
        fake_result = SimpleNamespace(final_output={"final_decision": "hold", "symbol": "BTCUSDT", "amount": 0.0})

        with (
            patch("traderbot_ai.decision.openai_agents_provider.get_session", return_value=fake_session) as get_session,
            patch("traderbot_ai.decision.openai_agents_provider.Runner.run_sync", return_value=fake_result) as run_sync,
            patch("traderbot_ai.decision.openai_agents_provider.write_run_record", return_value="run-log.json") as write_run_record,
        ):
            output = provider.decide(replay_context())

        get_session.assert_called_once()
        session_call = get_session.call_args
        self.assertEqual(session_call.args[0], "session-1704067200000")
        self.assertFalse(session_call.kwargs["enable_compaction"])
        self.assertEqual(run_sync.call_args.kwargs["max_turns"], 7)
        write_run_record.assert_called_once()
        self.assertEqual(output["agent_run_log"], "run-log.json")

    def test_codex_provider_builds_command_env_and_parses_final_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            options = CodexCliOptions(
                model="gpt-5.5",
                reasoning_effort="medium",
                sandbox="read-only",
                output_dir=tmp_path,
                codex_executable="codex-test",
                python_executable="python-test",
            )
            provider = CodexCliMcpDecisionProvider(settings=settings(), session_name="codex-session", options=options)
            stale_audit_path = tmp_path / "codex-session" / "1704067200000.mcp-audit.jsonl"
            stale_audit_path.parent.mkdir(parents=True)
            stale_audit_path.write_text('{"tool":"place_order","result":{"ok":false}}\n', encoding="utf-8")
            old_env = {
                key: os.environ.get(key)
                for key in (
                    "OPENAI_API_KEY",
                    "TRADERBOT_EXCHANGE_STATE_PATH",
                    "TRADERBOT_EXCHANGE_EVENTS_PATH",
                    "TRADERBOT_MARKET_CACHE_PATH",
                    "TRADERBOT_EXCHANGE_BACKEND",
                    "TRADERBOT_EXCHANGE_FEE_RATE",
                    "TRADERBOT_EXCHANGE_EXECUTION_INTERVAL",
                )
            }
            os.environ["OPENAI_API_KEY"] = "do-not-pass"
            os.environ["TRADERBOT_EXCHANGE_STATE_PATH"] = "state.json"
            os.environ["TRADERBOT_EXCHANGE_EVENTS_PATH"] = "events.jsonl"
            os.environ["TRADERBOT_MARKET_CACHE_PATH"] = "cache.sqlite3"
            os.environ["TRADERBOT_EXCHANGE_BACKEND"] = "simulated"
            os.environ["TRADERBOT_EXCHANGE_FEE_RATE"] = "0.001"
            os.environ["TRADERBOT_EXCHANGE_EXECUTION_INTERVAL"] = "1m"

            def fake_run(command, input, text, capture_output, timeout, env, cwd):
                self.assertFalse(stale_audit_path.exists())
                self.assertIn("System instructions:", input)
                self.assertIn("Replay step prompt:", input)
                self.assertIn("Run id: run-1", input)
                self.assertIn("Use the configured traderbot MCP tools", input)
                self.assertIn("Use scan_momentum_universe once for the broad symbol scan", input)
                self.assertIn("Use get_candidate_detail for at most 2 finalists", input)
                self.assertIn("linear place_order.qty is base-asset quantity", input)
                self.assertIn("--ignore-user-config", command)
                self.assertIn("--full-auto", command)
                self.assertIn('approval_policy="never"', command)
                self.assertIn("--json", command)
                self.assertIn("--output-schema", command)
                self.assertIn("--output-last-message", command)
                self.assertIn("-c", command)
                self.assertIn("-", command)
                self.assertIn('model_reasoning_effort="medium"', command)
                self.assertNotIn("OPENAI_API_KEY", env)
                command_text = "\n".join(command)
                self.assertIn("mcp_servers.traderbot.env.TRADERBOT_EXCHANGE_STATE_PATH", command_text)
                self.assertIn("mcp_servers.traderbot.env.TRADERBOT_EXCHANGE_EVENTS_PATH", command_text)
                self.assertIn("mcp_servers.traderbot.env.TRADERBOT_MARKET_CACHE_PATH", command_text)
                self.assertIn("mcp_servers.traderbot.env.TRADERBOT_EXCHANGE_BACKEND", command_text)
                self.assertIn("mcp_servers.traderbot.env.TRADERBOT_EXCHANGE_FEE_RATE", command_text)
                self.assertIn("mcp_servers.traderbot.env.TRADERBOT_EXCHANGE_EXECUTION_INTERVAL", command_text)
                self.assertIn('mcp_servers.traderbot.default_tools_approval_mode="approve"', command_text)
                self.assertIn("mcp_servers.traderbot.required=true", command_text)
                final_path = Path(command[command.index("--output-last-message") + 1])
                final_path.write_text(
                    json.dumps(
                        {
                            "final_decision": "hold",
                            "symbol": "BTCUSDT",
                            "timeframe": "4h",
                            "thesis": "No valid setup.",
                            "price": None,
                            "stop_loss": None,
                            "take_profit": None,
                            "amount": 0.0,
                            "confidence": 0.0,
                            "risk_summary": "hold",
                            "tool_summary": [],
                            "worklog_path": None,
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout='{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":40,"output_tokens":7}}\n',
                    stderr="",
                )

            try:
                with patch("traderbot_ai.decision.codex_cli_provider.subprocess.run", side_effect=fake_run):
                    output = provider.decide(replay_context())
                    decision_path_exists = Path(output["codex_decision_path"]).exists()
                    prompt_text = Path(output["codex_prompt_path"]).read_text(encoding="utf-8")
            finally:
                for key, value in old_env.items():
                    _restore_env(key, value)

        self.assertEqual(output["final_decision"], "hold")
        self.assertTrue(Path(output["codex_stdout_log"]).name.endswith(".jsonl"))
        self.assertTrue(Path(output["codex_prompt_path"]).name.endswith(".prompt.txt"))
        self.assertIn("Run id: run-1", prompt_text)
        self.assertTrue(Path(output["codex_mcp_audit_path"]).name.endswith(".mcp-audit.jsonl"))
        self.assertEqual(output["codex_usage"]["input_tokens"], 100)
        self.assertEqual(output["codex_usage"]["cached_input_tokens"], 40)
        self.assertEqual(output["codex_usage"]["uncached_input_tokens"], 60)
        self.assertEqual(output["codex_usage"]["output_tokens"], 7)
        self.assertEqual(output["codex_usage"]["total_tokens"], 107)
        self.assertTrue(decision_path_exists)

    def test_codex_provider_deterministic_prompt_and_mcp_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CodexCliMcpDecisionProvider(
                settings=settings(),
                session_name="codex-session",
                screener_mode="deterministic",
                options=CodexCliOptions(output_dir=Path(tmp), codex_executable="codex-test", python_executable="python-test"),
            )

            def fake_run(command, input, text, capture_output, timeout, env, cwd):
                self.assertIn("Use the scan table and candidate_primitives", input)
                self.assertIn("Do not call scan_momentum_universe", input)
                self.assertIn("deterministic MCP mode rejects broad/raw market-data bypasses", input)
                self.assertNotIn("Use scan_momentum_universe once for the broad symbol scan", input)
                self.assertEqual(env["TRADERBOT_SCREENER_MODE"], "deterministic")
                command_text = "\n".join(command)
                self.assertIn("mcp_servers.traderbot.env.TRADERBOT_SCREENER_MODE", command_text)
                final_path = Path(command[command.index("--output-last-message") + 1])
                final_path.write_text(
                    json.dumps(
                        {
                            "final_decision": "hold",
                            "symbol": "ETHUSDT",
                            "timeframe": "4h",
                            "thesis": "No valid setup.",
                            "price": None,
                            "stop_loss": None,
                            "take_profit": None,
                            "amount": 0.0,
                            "confidence": 0.0,
                            "risk_summary": "hold",
                            "tool_summary": [],
                            "worklog_path": None,
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("traderbot_ai.decision.codex_cli_provider.subprocess.run", side_effect=fake_run):
                output = provider.decide(deterministic_replay_context())

        self.assertEqual(output["final_decision"], "hold")

    def test_codex_provider_fail_open_hold_on_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CodexCliMcpDecisionProvider(
                settings=settings(),
                session_name="codex-session",
                options=CodexCliOptions(output_dir=Path(tmp), fail_open="hold"),
            )

            def fake_run(command, input, text, capture_output, timeout, env, cwd):
                final_path = Path(command[command.index("--output-last-message") + 1])
                final_path.write_text("not json", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("traderbot_ai.decision.codex_cli_provider.subprocess.run", side_effect=fake_run):
                output = provider.decide(replay_context())

        self.assertEqual(output["final_decision"], "hold")
        self.assertIn("did not match TradeDecision", output["risk_summary"])

    def test_codex_provider_rejects_partial_json_even_when_pydantic_has_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CodexCliMcpDecisionProvider(
                settings=settings(),
                session_name="codex-session",
                options=CodexCliOptions(output_dir=Path(tmp), fail_open="hold"),
            )

            def fake_run(command, input, text, capture_output, timeout, env, cwd):
                final_path = Path(command[command.index("--output-last-message") + 1])
                final_path.write_text(
                    json.dumps(
                        {
                            "final_decision": "hold",
                            "symbol": "BTCUSDT",
                            "timeframe": "4h",
                            "thesis": "No valid setup.",
                            "risk_summary": "hold",
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("traderbot_ai.decision.codex_cli_provider.subprocess.run", side_effect=fake_run):
                output = provider.decide(replay_context())

        self.assertEqual(output["final_decision"], "hold")
        self.assertIn("missing required fields", output["risk_summary"])

    def test_codex_output_schema_is_strict_for_all_trade_decision_fields(self) -> None:
        schema = _trade_decision_output_schema()

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"].keys()))
        self.assertNotIn("default", json.dumps(schema))


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
