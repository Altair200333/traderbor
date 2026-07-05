from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator


_RUN_ID = ContextVar[str | None]("traderbot_ai_run_id", default=None)
_SAFE_RUN_ID = re.compile(r"[^A-Za-z0-9_.=-]+")


def normalize_run_id(run_id: str | None) -> str | None:
    if run_id is None:
        return None
    cleaned = _SAFE_RUN_ID.sub("-", str(run_id).strip()).strip(".-")
    return cleaned or None


def get_current_run_id() -> str | None:
    return _RUN_ID.get()


def current_worklog_root() -> str:
    run_id = get_current_run_id()
    if run_id:
        return f"worklog/runs/{run_id}"
    return "worklog"


@contextmanager
def agent_run_context(run_id: str | None) -> Iterator[str | None]:
    normalized = normalize_run_id(run_id)
    token = _RUN_ID.set(normalized)
    try:
        yield normalized
    finally:
        _RUN_ID.reset(token)
