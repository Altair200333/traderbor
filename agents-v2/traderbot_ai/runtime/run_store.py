from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from traderbot_ai.paths import RUNS_DIR, ensure_runtime_dirs


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _usage_from_raw_responses(raw_responses: list[Any]) -> list[Any]:
    usage = []
    for response in raw_responses:
        item = getattr(response, "usage", None)
        if item is not None:
            usage.append(_jsonable(item))
    return usage


def write_run_record(session_id: str, prompt: str, result: Any) -> str:
    ensure_runtime_dirs()
    now = datetime.now(timezone.utc)
    raw_responses = list(getattr(result, "raw_responses", []) or [])
    record = {
        "ts": now.isoformat(),
        "session_id": session_id,
        "prompt": prompt,
        "final_output": _jsonable(getattr(result, "final_output", None)),
        "last_agent": getattr(getattr(result, "last_agent", None), "name", None),
        "new_items": [str(type(item).__name__) for item in (getattr(result, "new_items", []) or [])],
        "usage": _usage_from_raw_responses(raw_responses),
    }

    path = RUNS_DIR / f"{now.date().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return str(path)
