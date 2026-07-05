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


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(AGENTS_V2_ROOT / ".env", override=True)
    ensure_runtime_dirs()

    settings = Settings(
        model=os.getenv("TRADERBOT_MODEL", "gpt-5.5"),
        vision_model=os.getenv("TRADERBOT_VISION_MODEL", os.getenv("TRADERBOT_MODEL", "gpt-5.5")),
        reasoning_effort=os.getenv("TRADERBOT_REASONING_EFFORT", "medium"),
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
