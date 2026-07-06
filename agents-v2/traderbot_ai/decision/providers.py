from __future__ import annotations

from typing import Any, Protocol


class DecisionProvider(Protocol):
    name: str

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...
