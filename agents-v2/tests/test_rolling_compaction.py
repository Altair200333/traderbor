from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from agents import SQLiteSession

from traderbot_ai.config import Settings
from traderbot_ai.runtime.run_context import agent_run_context
from traderbot_ai.runtime.sessions import (
    OpenAIContextCompactor,
    RollingCompactionConfig,
    RollingCompactionSession,
    count_message_like_items,
    get_session,
)
from traderbot_ai.runtime.usage import usage_totals
from traderbot_ai.tools.workspace import append_worklog_record


def message(label: str, role: str = "user") -> dict:
    return {"role": role, "content": label}


class FakeCompactor:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def compact(self, *, session_id: str, items: list[dict], compacted_message_count: int) -> str:
        del session_id, compacted_message_count
        self.calls.append(items)
        return "summary of " + json.dumps([item.get("content") or item.get("type") for item in items])


class FailingCompactor:
    def __init__(self) -> None:
        self.calls = 0

    def compact(self, *, session_id: str, items: list[dict], compacted_message_count: int) -> str:
        del session_id, items, compacted_message_count
        self.calls += 1
        raise RuntimeError("boom")


class RollingCompactionSessionTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self, compactor=None) -> RollingCompactionSession:
        inner = SQLiteSession(session_id="test", db_path=":memory:")
        session = RollingCompactionSession(
            inner,
            compactor=compactor or FakeCompactor(),
            config=RollingCompactionConfig(threshold_messages=10, compact_messages=7, max_rolls=8),
        )
        self.addCleanup(session.close)
        return session

    async def test_does_not_compact_before_threshold(self) -> None:
        compactor = FakeCompactor()
        session = self.make_session(compactor)
        await session.add_items([message(f"m{i}") for i in range(9)])

        stored = await session.get_items()

        self.assertEqual(9, count_message_like_items(stored))
        self.assertEqual([], compactor.calls)

    async def test_compacts_ten_messages_to_summary_plus_latest_three(self) -> None:
        compactor = FakeCompactor()
        session = self.make_session(compactor)
        await session.add_items([message(f"m{i}") for i in range(10)])

        stored = await session.get_items()

        self.assertEqual(4, count_message_like_items(stored))
        self.assertIn("Rolling context compaction summary", stored[0]["content"])
        self.assertEqual(["m7", "m8", "m9"], [item["content"] for item in stored[1:]])
        self.assertEqual(["m0", "m1", "m2", "m3", "m4", "m5", "m6"], [item["content"] for item in compactor.calls[0]])

    async def test_repeated_rolls_keep_prior_summary_and_do_not_drop_middle_messages(self) -> None:
        compactor = FakeCompactor()
        session = self.make_session(compactor)
        await session.add_items([message(f"m{i}") for i in range(17)])

        stored = await session.get_items()
        serialized = json.dumps(stored)

        self.assertEqual(2, len(compactor.calls))
        self.assertIn("Rolling context compaction summary", compactor.calls[1][0]["content"])
        self.assertEqual(["m13", "m14", "m15", "m16"], [item["content"] for item in stored[1:]])
        for index in range(17):
            self.assertIn(f"m{index}", serialized)

    async def test_large_imported_history_rolls_without_data_loss(self) -> None:
        compactor = FakeCompactor()
        session = self.make_session(compactor)
        await session.add_items([message(f"m{i}") for i in range(25)])

        stored = await session.get_items()
        serialized = json.dumps(stored)

        self.assertEqual(3, len(compactor.calls))
        self.assertEqual(["m19", "m20", "m21", "m22", "m23", "m24"], [item["content"] for item in stored[1:]])
        for index in range(25):
            self.assertIn(f"m{index}", serialized)

    async def test_tool_call_tail_is_not_orphaned_by_message_boundary(self) -> None:
        compactor = FakeCompactor()
        session = self.make_session(compactor)
        items = [message(f"m{i}") for i in range(7)]
        items.extend(
            [
                {"type": "function_call", "call_id": "call-1", "name": "tool", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call-1", "output": "{}"},
                message("m7", "assistant"),
                message("m8"),
                message("m9", "assistant"),
            ]
        )
        await session.add_items(items)

        stored = await session.get_items()

        self.assertEqual("function_call", stored[1]["type"])
        self.assertEqual("function_call_output", stored[2]["type"])
        self.assertEqual(["m7", "m8", "m9"], [item["content"] for item in stored[3:]])

    async def test_get_items_is_read_only(self) -> None:
        inner = SQLiteSession(session_id="test-read", db_path=":memory:")
        self.addCleanup(inner.close)
        await inner.add_items([message(f"m{i}") for i in range(10)])
        compactor = FakeCompactor()
        session = RollingCompactionSession(inner, compactor=compactor)

        stored = await session.get_items()

        self.assertEqual(10, count_message_like_items(stored))
        self.assertEqual([], compactor.calls)

    async def test_limit_pop_and_clear_after_compaction(self) -> None:
        session = self.make_session()
        await session.add_items([message(f"m{i}") for i in range(10)])

        limited = await session.get_items(limit=2)
        popped = await session.pop_item()
        after_pop = await session.get_items()
        await session.clear_session()

        self.assertEqual(["m8", "m9"], [item["content"] for item in limited])
        self.assertEqual("m9", popped["content"])
        self.assertEqual(3, count_message_like_items(after_pop))
        self.assertEqual([], await session.get_items())

    async def test_compactor_failure_preserves_uncompacted_history(self) -> None:
        compactor = FailingCompactor()
        session = self.make_session(compactor)
        await session.add_items([message(f"m{i}") for i in range(10)])

        stored = await session.get_items()

        self.assertEqual(1, compactor.calls)
        self.assertEqual([f"m{i}" for i in range(10)], [item["content"] for item in stored])

    async def test_max_rolls_warns_only_when_history_remains_above_threshold(self) -> None:
        compactor = FakeCompactor()
        inner = SQLiteSession(session_id="test-max-rolls", db_path=":memory:")
        session = RollingCompactionSession(
            inner,
            compactor=compactor,
            config=RollingCompactionConfig(threshold_messages=10, compact_messages=7, max_rolls=1),
        )
        self.addCleanup(session.close)

        with self.assertLogs("traderbot_ai.runtime.sessions", level="WARNING") as logs:
            await session.add_items([message(f"m{i}") for i in range(17)])

        self.assertEqual(1, len(compactor.calls))
        self.assertEqual(11, count_message_like_items(await session.get_items()))
        self.assertIn("stopped after 1 rolls", "\n".join(logs.output))


class OpenAICompactorTests(unittest.TestCase):
    def test_real_compactor_request_shape_is_low_effort_and_detailed(self) -> None:
        class FakeResponses:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_text": "detailed summary"})()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        client = FakeClient()
        compactor = OpenAIContextCompactor(
            model="gpt-5.5",
            reasoning_effort="low",
            max_output_tokens=12000,
            client=client,
        )

        summary = compactor.compact(session_id="s1", items=[message("hello")], compacted_message_count=1)

        self.assertEqual("detailed summary", summary)
        self.assertEqual("gpt-5.5", client.responses.kwargs["model"])
        self.assertEqual({"effort": "low"}, client.responses.kwargs["reasoning"])
        self.assertEqual({"verbosity": "medium"}, client.responses.kwargs["text"])
        self.assertEqual(12000, client.responses.kwargs["max_output_tokens"])
        self.assertIn("Preserve: user requests", client.responses.kwargs["input"][1]["content"])

    def test_empty_compactor_response_raises(self) -> None:
        class FakeResponses:
            def create(self, **kwargs):
                del kwargs
                return type("Response", (), {"output_text": "  "})()

        class FakeClient:
            responses = FakeResponses()

        compactor = OpenAIContextCompactor(client=FakeClient())

        with self.assertRaisesRegex(RuntimeError, "empty summary"):
            compactor.compact(session_id="s1", items=[message("hello")], compacted_message_count=1)

    def test_incomplete_compactor_response_raises(self) -> None:
        class FakeResponses:
            def create(self, **kwargs):
                del kwargs
                return type("Response", (), {"output_text": "partial", "status": "incomplete"})()

        class FakeClient:
            responses = FakeResponses()

        compactor = OpenAIContextCompactor(client=FakeClient())

        with self.assertRaisesRegex(RuntimeError, "status incomplete"):
            compactor.compact(session_id="s1", items=[message("hello")], compacted_message_count=1)


class SessionFactoryTests(unittest.TestCase):
    def make_settings(self, **overrides) -> Settings:
        values = {
            "model": "test",
            "vision_model": "test",
            "reasoning_effort": "low",
            "paper_balance_usdt": 1000.0,
            "market_data_mode": "cache",
            "openai_api_key_present": False,
            "enable_codex_tool": False,
            "disable_tracing": True,
        }
        values.update(overrides)
        return Settings(**values)

    def test_get_session_can_disable_compaction(self) -> None:
        settings = self.make_settings(context_compaction_enabled=True)
        session = get_session(
            f"test-factory-{uuid.uuid4().hex}",
            settings=settings,
            enable_compaction=False,
        )
        self.addCleanup(session.close)

        self.assertIsInstance(session, SQLiteSession)

    def test_get_session_wires_compaction_settings_and_compactor(self) -> None:
        compactor = FakeCompactor()
        settings = self.make_settings(
            context_compaction_threshold_messages=5,
            context_compaction_compact_messages=99,
            context_compaction_max_rolls=0,
        )
        session = get_session(
            f"test-factory-{uuid.uuid4().hex}",
            settings=settings,
            compactor=compactor,
        )
        self.addCleanup(session.close)

        self.assertIsInstance(session, RollingCompactionSession)
        self.assertEqual(5, session.config.threshold_messages)
        self.assertEqual(4, session.config.compact_messages)
        self.assertEqual(1, session.config.max_rolls)


class WorklogIsolationTests(unittest.TestCase):
    def test_worklog_uses_current_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("traderbot_ai.tools.workspace.WORKLOG_DIR", root):
                with agent_run_context("run/one"):
                    result = append_worklog_record("hello")

            path = Path(result["path"])
            self.assertEqual("run-one", result["run_id"])
            self.assertEqual(root / "runs" / "run-one" / path.name, path)
            self.assertIn("hello", path.read_text(encoding="utf-8"))

    def test_worklog_fallback_and_explicit_run_do_not_cross_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("traderbot_ai.tools.workspace.WORKLOG_DIR", root):
                fallback = append_worklog_record("shared")
                explicit = append_worklog_record("isolated", run_id="run-two")

            fallback_path = Path(fallback["path"])
            explicit_path = Path(explicit["path"])
            self.assertIsNone(fallback["run_id"])
            self.assertEqual(root / fallback_path.name, fallback_path)
            self.assertEqual(root / "runs" / "run-two" / explicit_path.name, explicit_path)
            self.assertIn("shared", fallback_path.read_text(encoding="utf-8"))
            self.assertIn("isolated", explicit_path.read_text(encoding="utf-8"))


class UsageTotalsTests(unittest.TestCase):
    def test_usage_totals_extract_cached_tokens_from_dicts_and_strings(self) -> None:
        totals = usage_totals(
            [
                {"usage": [{"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40}, "output_tokens": 10, "total_tokens": 110}]},
                {"input_tokens": 25, "cached_input_tokens": 10, "output_tokens": 2, "total_tokens": 27},
                "Usage(requests=1, input_tokens=50, input_tokens_details=InputTokensDetails(cached_tokens=20), output_tokens=5, total_tokens=55)",
            ]
        )

        self.assertEqual(175, totals.input_tokens)
        self.assertEqual(70, totals.cached_input_tokens)
        self.assertEqual(17, totals.output_tokens)
        self.assertEqual(192, totals.total_tokens)
        self.assertEqual(105, totals.uncached_input_tokens)
        self.assertAlmostEqual(0.4, totals.cache_hit_ratio)
