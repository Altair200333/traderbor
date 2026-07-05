from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from traderbot_ai.paths import AGENTS_V2_ROOT, PROJECT_ROOT, ensure_runtime_dirs


@dataclass(frozen=True)
class Settings:
    model: str
    vision_model: str
    reasoning_effort: str
    paper_balance_usdt: float
    market_data_mode: str
    openai_api_key_present: bool
    enable_codex_tool: bool
    disable_tracing: bool
    max_tokens: int = 64000
    context_compaction_enabled: bool = True
    context_compaction_model: str = "gpt-5.5"
    context_compaction_reasoning_effort: str = "low"
    context_compaction_max_tokens: int = 12000
    context_compaction_threshold_messages: int = 10
    context_compaction_compact_messages: int = 7
    context_compaction_max_rolls: int = 8


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(AGENTS_V2_ROOT / ".env", override=True)
    ensure_runtime_dirs()

    settings = Settings(
        model=os.getenv("TRADERBOT_MODEL", "gpt-5.5"),
        vision_model=os.getenv("TRADERBOT_VISION_MODEL", os.getenv("TRADERBOT_MODEL", "gpt-5.5")),
        reasoning_effort=os.getenv("TRADERBOT_REASONING_EFFORT", "high"),
        max_tokens=int(os.getenv("TRADERBOT_MAX_TOKENS", "64000")),
        context_compaction_enabled=os.getenv("TRADERBOT_CONTEXT_COMPACTION_ENABLED", "1") not in {"0", "false", "False"},
        context_compaction_model=os.getenv("TRADERBOT_CONTEXT_COMPACTION_MODEL", "gpt-5.5"),
        context_compaction_reasoning_effort=os.getenv("TRADERBOT_CONTEXT_COMPACTION_REASONING_EFFORT", "low"),
        context_compaction_max_tokens=int(os.getenv("TRADERBOT_CONTEXT_COMPACTION_MAX_TOKENS", "12000")),
        context_compaction_threshold_messages=int(os.getenv("TRADERBOT_CONTEXT_COMPACTION_THRESHOLD_MESSAGES", "10")),
        context_compaction_compact_messages=int(os.getenv("TRADERBOT_CONTEXT_COMPACTION_COMPACT_MESSAGES", "7")),
        context_compaction_max_rolls=int(os.getenv("TRADERBOT_CONTEXT_COMPACTION_MAX_ROLLS", "8")),
        paper_balance_usdt=float(os.getenv("TRADERBOT_PAPER_BALANCE_USDT", "1000")),
        market_data_mode=os.getenv("TRADERBOT_MARKET_DATA_MODE", "live").lower(),
        openai_api_key_present=bool(os.getenv("OPENAI_API_KEY")),
        enable_codex_tool=os.getenv("TRADERBOT_ENABLE_CODEX_TOOL", "1") not in {"0", "false", "False"},
        disable_tracing=os.getenv("TRADERBOT_DISABLE_TRACING", "1") not in {"0", "false", "False"},
    )
    if settings.disable_tracing:
        try:
            from agents import set_tracing_disabled

            set_tracing_disabled(True)
        except Exception:
            pass
    return settings
