from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from traderbot_ai.cli import build_parser
from traderbot_ai.exchange import SimulatedExchange
from traderbot_ai.simulator.exchange_replay import build_exchange_replay_config, run_exchange_replay
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.simulator.replay_report import build_replay_report, render_replay_markdown, write_replay_markdown_report


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


class ReplayReportTests(unittest.TestCase):
    def test_replay_report_summarizes_steps_events_and_wallet_delta(self) -> None:
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
            exchange = SimulatedExchange(state_path, events_path, cache=cache)
            config = build_exchange_replay_config(
                symbols=[BTC, ETH],
                start_time=BASE_MS,
                end_time=BASE_MS + 2 * FOUR_HOURS_MS,
                fee_rate=0.001,
                state_path=state_path,
                events_path=events_path,
                replay_path=replay_path,
                run_id="report-test",
            )
            calls = []

            def decide(context: dict) -> dict:
                calls.append(context["as_of_ms"])
                if len(calls) == 1:
                    exchange.set_leverage("linear", BTC, "5", "5")
                    exchange.place_order("linear", BTC, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=96.0, as_of=context["as_of_ms"])
                    return {
                        "final_decision": "long",
                        "symbol": BTC,
                        "amount": 100.0,
                        "take_profit": 110.0,
                        "stop_loss": 96.0,
                        "thesis": "test long",
                        "risk_summary": "ok",
                    }
                return {"final_decision": "hold", "symbol": BTC, "thesis": "wait", "risk_summary": "done"}

            run_exchange_replay(config=config, decide=decide, cache=cache, exchange=exchange)

            report = build_replay_report(replay_path=replay_path, exchange_events_path=events_path)
            markdown = render_replay_markdown(report)

            self.assertTrue(report["ok"])
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["step_count"], 2)
            self.assertAlmostEqual(report["equity_delta_usdt"], 9.79)
            self.assertEqual(report["steps"][0]["decision"]["final_decision"], "long")
            self.assertIn("place_order", report["steps"][0]["agent_exchange_event_types"])
            self.assertEqual(report["steps"][1]["settlement_closed_count"], 1)
            self.assertEqual(report["exchange_event_counts"]["position_closed"], 1)
            self.assertIn("Exchange replay report", markdown)
            self.assertIn("Decision: long BTCUSDT", markdown)

    def test_write_replay_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.jsonl"
            replay_path.write_text(
                "\n".join(
                    [
                        '{"type":"replay_started","timestamp":"2024-01-01T00:00:00+00:00","payload":{"config":{"run_id":"x","symbols":["BTCUSDT"],"start_ms":1704067200000,"end_ms":1704081600000,"decision_interval":"4h","execution_interval":"1m"}}}',
                        '{"type":"replay_completed","timestamp":"2024-01-01T04:00:00+00:00","payload":{"final_wallet":{"totals":{"equity_usdt":1000.0}}}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = Path(tmp) / "report.md"

            result = write_replay_markdown_report(replay_path, output_path)

            self.assertEqual(result["markdown_path"], str(output_path))
            self.assertTrue(output_path.exists())
            self.assertIn("Status: completed", output_path.read_text(encoding="utf-8"))

    def test_replay_report_uses_last_run_and_marks_incomplete_not_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.jsonl"
            replay_path.write_text(
                "\n".join(
                    [
                        '{"type":"replay_started","timestamp":"2024-01-01T00:00:00+00:00","payload":{"config":{"run_id":"old"}}}',
                        '{"type":"replay_failed","timestamp":"2024-01-01T00:01:00+00:00","payload":{"error":"old failure","error_type":"RuntimeError"}}',
                        '{"type":"replay_started","timestamp":"2024-01-02T00:00:00+00:00","payload":{"config":{"run_id":"new"}}}',
                        '{"type":"step_completed","timestamp":"2024-01-02T00:01:00+00:00","payload":{"as_of_ms":1704067200000,"wallet_before":{"totals":{"equity_usdt":100.0}},"wallet_after":{"totals":{"equity_usdt":101.0}},"decision":{"final_decision":"hold","symbol":"BTCUSDT"},"settlement":{},"agent_exchange_events":[]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_replay_report(replay_path)

            self.assertFalse(report["ok"])
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["config"]["run_id"], "new")
            self.assertEqual(report["step_count"], 1)

    def test_replay_report_skips_truncated_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.jsonl"
            replay_path.write_text(
                '{"type":"replay_started","timestamp":"2024-01-01T00:00:00+00:00","payload":{"config":{"run_id":"x"}}}\n123\n{"type":',
                encoding="utf-8",
            )

            report = build_replay_report(replay_path)

            self.assertEqual(report["status"], "incomplete")
            self.assertFalse(report["ok"])
            self.assertEqual(report["skipped_replay_lines"], 2)

    def test_replay_report_surfaces_ambiguous_closes_and_incomplete_valuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.jsonl"
            closed_position = {
                "symbol": "BTCUSDT",
                "side": "Buy",
                "status": "Closed",
                "exit_reason": "SL",
                "realized_pnl_usdt": -5.0,
                "ambiguous": True,
                "resolution_interval": "1m",
                "source_interval": "4h",
            }
            events = [
                {"type": "replay_started", "timestamp": "2024-01-01T00:00:00+00:00", "payload": {"config": {"run_id": "x"}}},
                {
                    "type": "step_completed",
                    "timestamp": "2024-01-01T04:00:00+00:00",
                    "payload": {
                        "as_of_ms": 1704081600000,
                        "wallet_before": {"totals": {"equity_usdt": 100.0, "valuation_complete": False, "unpriced_assets": ["BTC"]}},
                        "wallet_after": {"totals": {"equity_usdt": 120.0, "valuation_complete": True}},
                        "decision": {"final_decision": "hold", "symbol": "BTCUSDT"},
                        "settlement": {"closed_positions": [closed_position]},
                        "agent_exchange_events": [{"type": "position_closed", "payload": {"closed_position": closed_position}}],
                    },
                },
                {
                    "type": "replay_completed",
                    "timestamp": "2024-01-01T04:01:00+00:00",
                    "payload": {"final_wallet": {"totals": {"equity_usdt": 120.0, "valuation_complete": True}}},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

            report = build_replay_report(replay_path)
            markdown = render_replay_markdown(report)

            self.assertIsNone(report["initial_equity_usdt"])
            self.assertIsNone(report["equity_delta_usdt"])
            self.assertEqual(report["steps"][0]["settlement_ambiguous_count"], 1)
            self.assertEqual(report["steps"][0]["settlement_resolution_intervals"], ["1m"])
            self.assertEqual(report["steps"][0]["settlement_source_intervals"], ["4h"])
            self.assertTrue(report["steps"][0]["agent_exchange_events"][0]["ambiguous"])
            self.assertIn("Ambiguous settlement closes: 1 (resolution: 1m; source: 4h)", markdown)

    def test_replay_report_tolerates_null_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.jsonl"
            replay_path.write_text(
                '{"type":"replay_started","timestamp":"2024-01-01T00:00:00+00:00","payload":null}\n'
                '{"type":"step_completed","timestamp":"2024-01-01T00:01:00+00:00","payload":null}\n',
                encoding="utf-8",
            )

            report = build_replay_report(replay_path)

            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["step_count"], 1)

    def test_replay_report_missing_file_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.jsonl"

            with self.assertRaises(FileNotFoundError) as context:
                build_replay_report(missing)

            self.assertIn("file not found", str(context.exception))

    def test_cli_has_replay_report_command(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["replay-report", "--replay-path", "x.jsonl"])

        self.assertEqual(args.command, "replay-report")


if __name__ == "__main__":
    unittest.main()
