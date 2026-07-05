from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from traderbot_ai.config import Settings, load_settings
from traderbot_ai.schemas import ProviderStatus


def provider_status(live: bool = False, settings: Settings | None = None) -> list[ProviderStatus]:
    settings = settings or load_settings()
    statuses = [
        ProviderStatus(
            provider="openai",
            configured=settings.openai_api_key_present,
            live_ok=None,
            model=settings.model,
            detail="OPENAI_API_KEY present" if settings.openai_api_key_present else "OPENAI_API_KEY missing",
        ),
        ProviderStatus(
            provider="google-search",
            configured=bool(os.getenv("GOOGLE_SEARCH_API_KEY")),
            live_ok=None,
            model=None,
            detail="Configured as source env possibility; no agent provider implemented.",
        ),
    ]
    if live and settings.openai_api_key_present:
        statuses[0] = _probe_openai(settings)
    return statuses


def _probe_openai(settings: Settings) -> ProviderStatus:
    try:
        client = OpenAI()
        response = client.responses.create(
            model=settings.model,
            input="Return exactly OK.",
            max_output_tokens=16,
            reasoning={"effort": settings.reasoning_effort},
            text={"verbosity": "low"},
        )
        text = getattr(response, "output_text", "") or str(response)
        ok = "OK" in text.upper()
        return ProviderStatus(
            provider="openai",
            configured=True,
            live_ok=ok,
            model=settings.model,
            detail="Live probe succeeded" if ok else f"Live probe returned unexpected text: {text[:120]}",
        )
    except Exception as error:
        return ProviderStatus(
            provider="openai",
            configured=True,
            live_ok=False,
            model=settings.model,
            detail=str(error),
        )
