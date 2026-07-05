from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from agents import SQLiteSession
from agents.items import TResponseInputItem
from openai import OpenAI

from traderbot_ai.config import Settings, load_settings
from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs


logger = logging.getLogger(__name__)


class ContextCompactor(Protocol):
    def compact(
        self,
        *,
        session_id: str,
        items: list[TResponseInputItem],
        compacted_message_count: int,
    ) -> str:
        ...


@dataclass(frozen=True)
class RollingCompactionConfig:
    threshold_messages: int = 10
    compact_messages: int = 7
    max_rolls: int = 8

    @classmethod
    def from_settings(cls, settings: Settings) -> "RollingCompactionConfig":
        threshold = max(2, settings.context_compaction_threshold_messages)
        compact = max(1, min(settings.context_compaction_compact_messages, threshold - 1))
        return cls(
            threshold_messages=threshold,
            compact_messages=compact,
            max_rolls=max(1, settings.context_compaction_max_rolls),
        )


class OpenAIContextCompactor:
    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        reasoning_effort: str = "low",
        max_output_tokens: int = 12000,
        verbosity: str = "medium",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.verbosity = verbosity
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIContextCompactor":
        return cls(
            model=settings.context_compaction_model,
            reasoning_effort=settings.context_compaction_reasoning_effort,
            max_output_tokens=settings.context_compaction_max_tokens,
        )

    def compact(
        self,
        *,
        session_id: str,
        items: list[TResponseInputItem],
        compacted_message_count: int,
    ) -> str:
        payload = json.dumps(items, ensure_ascii=False, indent=2, default=str)
        client = self.client or OpenAI()
        response = client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            text={"verbosity": self.verbosity},
            max_output_tokens=self.max_output_tokens,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You compact old chat context for a local trading agent. "
                        "Create a detailed, factual summary that preserves all durable state. "
                        "Do not add new facts or advice."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Session id: {session_id}\n"
                        f"Compacted message-like items: {compacted_message_count}\n\n"
                        "Summarize the JSON session prefix below. Preserve: user requests, "
                        "assistant decisions, market facts, wallet/order facts, tool outputs, "
                        "code/worklog references, unresolved questions, failed attempts, and "
                        "next useful steps. Make it detailed enough that the original prefix "
                        "can be replaced by the summary in a future model call.\n\n"
                        f"{payload}"
                    ),
                },
            ],
        )
        status = getattr(response, "status", None)
        if status and status != "completed":
            raise RuntimeError(f"context compactor response ended with status {status}")
        text = getattr(response, "output_text", "") or ""
        if not text.strip():
            raise RuntimeError("context compactor returned empty summary")
        return text.strip()


class RollingCompactionSession:
    def __init__(
        self,
        inner: SQLiteSession,
        *,
        compactor: ContextCompactor,
        config: RollingCompactionConfig | None = None,
    ) -> None:
        self.inner = inner
        self.compactor = compactor
        self.config = config or RollingCompactionConfig()
        self.session_id = inner.session_id
        self.session_settings = inner.session_settings
        self._lock = asyncio.Lock()

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        return await self.inner.get_items(limit=limit)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self.inner.add_items(items)
        if items:
            await self._compact_if_needed()

    async def pop_item(self) -> TResponseInputItem | None:
        return await self.inner.pop_item()

    async def clear_session(self) -> None:
        await self.inner.clear_session()

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    async def _compact_if_needed(self) -> None:
        async with self._lock:
            rolls = 0
            while rolls < self.config.max_rolls:
                items = await self.inner.get_items()
                message_count = count_message_like_items(items)
                if message_count < self.config.threshold_messages:
                    return
                boundary = boundary_after_message_count(items, self.config.compact_messages)
                if boundary is None:
                    return
                prefix = items[:boundary]
                tail = items[boundary:]
                try:
                    summary = await asyncio.to_thread(
                        self.compactor.compact,
                        session_id=self.session_id,
                        items=prefix,
                        compacted_message_count=count_message_like_items(prefix),
                    )
                    replacement = [build_compaction_message(self.session_id, prefix, summary)] + tail
                    await replace_sqlite_session_items(self.inner, replacement)
                    rolls += 1
                except Exception as error:
                    logger.warning("context compaction failed for session %s: %s", self.session_id, error)
                    return
            items = await self.inner.get_items()
            if count_message_like_items(items) >= self.config.threshold_messages:
                logger.warning(
                    "context compaction stopped after %s rolls for session %s",
                    self.config.max_rolls,
                    self.session_id,
                )


def count_message_like_items(items: list[TResponseInputItem]) -> int:
    return sum(1 for item in items if is_message_like_item(item))


def is_message_like_item(item: TResponseInputItem) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") == "message":
        return True
    return item.get("role") in {"system", "developer", "user", "assistant"}


def boundary_after_message_count(items: list[TResponseInputItem], message_count: int) -> int | None:
    seen = 0
    for index, item in enumerate(items):
        if is_message_like_item(item):
            seen += 1
            if seen == message_count:
                return index + 1
    return None


def build_compaction_message(
    session_id: str,
    compacted_items: list[TResponseInputItem],
    summary: str,
) -> TResponseInputItem:
    now = datetime.now(timezone.utc).isoformat()
    message_count = count_message_like_items(compacted_items)
    return {
        "role": "user",
        "content": (
            "Rolling context compaction summary.\n"
            "This message replaces older chat history. Treat it as historical context, "
            "not as a new user instruction.\n\n"
            f"Session id: {session_id}\n"
            f"Compacted at: {now}\n"
            f"Compacted message-like items: {message_count}\n"
            f"Compacted raw session items: {len(compacted_items)}\n\n"
            f"{summary.strip()}"
        ),
    }


async def replace_sqlite_session_items(
    session: SQLiteSession,
    items: list[TResponseInputItem],
) -> None:
    def _replace_sync() -> None:
        # SQLiteSession has no public replace-all API in agents 0.17.7. Keep this small and
        # tested so an SDK upgrade breaks loudly instead of silently corrupting history.
        if not hasattr(session, "_locked_connection") or not hasattr(session, "_insert_items"):
            raise RuntimeError("SQLiteSession no longer exposes the expected private storage hooks")
        with session._locked_connection() as conn:  # type: ignore[attr-defined]
            try:
                conn.execute(
                    f"DELETE FROM {session.messages_table} WHERE session_id = ?",
                    (session.session_id,),
                )
                session._insert_items(conn, items)  # type: ignore[attr-defined]
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    await asyncio.to_thread(_replace_sync)


def get_session(
    session_id: str,
    *,
    settings: Settings | None = None,
    enable_compaction: bool | None = None,
    compactor: ContextCompactor | None = None,
) -> SQLiteSession | RollingCompactionSession:
    ensure_runtime_dirs()
    settings = settings or load_settings()
    inner = SQLiteSession(session_id=session_id, db_path=DATA_DIR / "sessions.sqlite")
    enabled = settings.context_compaction_enabled if enable_compaction is None else enable_compaction
    if not enabled:
        return inner
    return RollingCompactionSession(
        inner,
        compactor=compactor or OpenAIContextCompactor.from_settings(settings),
        config=RollingCompactionConfig.from_settings(settings),
    )
