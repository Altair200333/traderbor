from __future__ import annotations

from traderbot_ai.decision.providers import DecisionProvider

__all__ = [
    "CodexCliMcpDecisionProvider",
    "CodexCliOptions",
    "DecisionProvider",
    "HoldDecisionProvider",
    "OpenAIAgentsDecisionProvider",
]


def __getattr__(name: str) -> object:
    if name in {"CodexCliMcpDecisionProvider", "CodexCliOptions"}:
        from traderbot_ai.decision.codex_cli_provider import CodexCliMcpDecisionProvider, CodexCliOptions

        return {"CodexCliMcpDecisionProvider": CodexCliMcpDecisionProvider, "CodexCliOptions": CodexCliOptions}[name]
    if name == "HoldDecisionProvider":
        from traderbot_ai.decision.hold_provider import HoldDecisionProvider

        return HoldDecisionProvider
    if name == "OpenAIAgentsDecisionProvider":
        from traderbot_ai.decision.openai_agents_provider import OpenAIAgentsDecisionProvider

        return OpenAIAgentsDecisionProvider
    raise AttributeError(name)
