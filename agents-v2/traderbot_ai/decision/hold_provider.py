from __future__ import annotations

from typing import Any


class HoldDecisionProvider:
    name = "hold"

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "final_decision": "hold",
            "symbol": context["symbols"][0],
            "amount": 0.0,
            "risk_summary": "deterministic hold decision mode",
        }

    def close(self) -> None:
        return None
