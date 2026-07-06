from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        return [jsonable(item) for item in value]
    elif isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    elif not isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    return value
