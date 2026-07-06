from __future__ import annotations

from typing import Any

from agents import Runner

from traderbot_ai.agents.trading import build_trading_agent
from traderbot_ai.config import Settings
from traderbot_ai.decision.prompts import build_exchange_replay_prompt
from traderbot_ai.decision.serialization import jsonable
from traderbot_ai.runtime.run_store import write_run_record
from traderbot_ai.runtime.sessions import get_session


def _close_session(session: Any) -> None:
    close = getattr(session, "close", None)
    if callable(close):
        close()


class OpenAIAgentsDecisionProvider:
    name = "openai-agents"

    def __init__(self, *, settings: Settings, session_name: str, max_turns: int) -> None:
        self._settings = settings
        self._session_name = session_name
        self._max_turns = max_turns
        self._agent = build_trading_agent(settings, exchange_replay=True)

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        step_session = get_session(
            f"{self._session_name}-{context['as_of_ms']}",
            settings=self._settings,
            enable_compaction=False,
        )
        prompt = build_exchange_replay_prompt(context)
        try:
            result = Runner.run_sync(self._agent, prompt, session=step_session, max_turns=self._max_turns)
        finally:
            _close_session(step_session)
        run_log = write_run_record(self._session_name, prompt, result)
        output = jsonable(result.final_output)
        if isinstance(output, dict):
            return {**output, "agent_run_log": run_log}
        return {"final_output": output, "agent_run_log": run_log}

    def close(self) -> None:
        return None
