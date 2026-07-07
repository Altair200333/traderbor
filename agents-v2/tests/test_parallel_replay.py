from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from traderbot_ai.simulator.parallel_replay import (
    build_session_command,
    extract_candidate_bars,
    summarize_session,
)

BASE_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z
HOUR = 3_600_000


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class ExtractCandidateBarsTests(unittest.TestCase):
    def test_extracts_only_bars_with_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay.jsonl"
            _write_jsonl(replay, [
                {"type": "replay_started", "payload": {}},
                {"type": "step_completed", "payload": {"as_of_ms": BASE_MS, "scan": {"candidates": []}}},
                {"type": "step_completed", "payload": {"as_of_ms": BASE_MS + HOUR, "scan": {"candidates": ["BTCUSDT"]}}},
                {"type": "step_completed", "payload": {"as_of_ms": BASE_MS + 2 * HOUR, "scan": {}}},
                {"type": "step_completed", "payload": {"as_of_ms": BASE_MS + 3 * HOUR, "scan": {"candidates": ["ETHUSDT", "SOLUSDT"]}}},
                {"type": "replay_completed", "payload": {}},
            ])
            self.assertEqual(extract_candidate_bars(replay), [BASE_MS + HOUR, BASE_MS + 3 * HOUR])


class SummarizeSessionTests(unittest.TestCase):
    def test_pnl_decision_and_flatness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "session.replay.jsonl"
            state = Path(tmp) / "session.exchange.json"
            _write_jsonl(replay, [
                {"type": "step_completed", "payload": {
                    "as_of_ms": BASE_MS,
                    "scan": {"candidates": ["BTCUSDT"]},
                    "decision": {"final_decision": "long", "symbol": "BTCUSDT"},
                }},
                {"type": "step_completed", "payload": {"as_of_ms": BASE_MS + HOUR, "scan": {}, "decision": {"final_decision": "hold"}}},
                {"type": "replay_completed", "payload": {"final_wallet": {"totals": {"equity_usdt": 1024.5}}}},
            ])
            state.write_text(json.dumps({
                "positions": [],
                "orders": [{"status": "Filled"}, {"status": "Cancelled"}],
                "closed_positions": [{"close_reason": "take_profit"}],
            }), encoding="utf-8")

            summary = summarize_session(replay, state, budget_usdt=1000.0)

        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["candidates"], ["BTCUSDT"])
        self.assertEqual(summary["decision"], "long")
        self.assertEqual(summary["closed_positions"], 1)
        self.assertEqual(summary["close_reasons"], ["take_profit"])
        self.assertTrue(summary["flat"])
        self.assertAlmostEqual(summary["pnl_usdt"], 24.5)

    def test_pending_order_marks_session_not_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "session.replay.jsonl"
            state = Path(tmp) / "session.exchange.json"
            _write_jsonl(replay, [
                {"type": "step_completed", "payload": {"as_of_ms": BASE_MS, "scan": {"candidates": ["BTCUSDT"]}, "decision": {"final_decision": "long", "symbol": "BTCUSDT"}}},
                {"type": "replay_completed", "payload": {"final_wallet": {"totals": {"equity_usdt": 1000.0}}}},
            ])
            state.write_text(json.dumps({"positions": [], "orders": [{"status": "New"}], "closed_positions": []}), encoding="utf-8")
            summary = summarize_session(replay, state, budget_usdt=1000.0)
        self.assertFalse(summary["flat"])


class SessionCommandTests(unittest.TestCase):
    def test_codex_session_command_shape(self) -> None:
        command = build_session_command(
            sys.executable,
            symbols="BTCUSDT,ETHUSDT",
            start_ms=BASE_MS,
            end_ms=BASE_MS + 30 * HOUR,
            execution_interval="5m",
            balance_usdt=1000.0,
            fee_rate=0.001,
            scanner_provider="v2",
            session_id="par-test-1",
            out_dir=Path("out"),
            codex_model="gpt-5.5",
            codex_reasoning_effort="high",
        )
        text = " ".join(command)
        for expected in (
            "exchange-replay",
            "--decide-first-bar-only",
            "--end-when-flat",
            "--screener-mode deterministic",
            "--decision-provider codex-cli-mcp",
            "--scanner-provider v2",
            "--codex-model gpt-5.5",
            "--codex-reasoning-effort high",
            "--no-preload",
            "--start-time 2024-01-01T00:00:00Z",
        ):
            self.assertIn(expected, text)

    def test_phase1_hold_command_has_no_session_flags(self) -> None:
        command = build_session_command(
            sys.executable,
            symbols="BTCUSDT",
            start_ms=BASE_MS,
            end_ms=BASE_MS + 24 * HOUR,
            execution_interval="5m",
            balance_usdt=1000.0,
            fee_rate=0.001,
            scanner_provider=None,
            session_id="phase1",
            out_dir=Path("out"),
            decision_provider="hold",
            first_bar_only=False,
        )
        text = " ".join(command)
        self.assertIn("--decision-provider hold", text)
        self.assertNotIn("--decide-first-bar-only", text)
        self.assertNotIn("--end-when-flat", text)
        self.assertNotIn("--codex-model", text)


if __name__ == "__main__":
    unittest.main()
