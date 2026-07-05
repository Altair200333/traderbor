from __future__ import annotations

import base64
import mimetypes
from typing import Literal

from agents import function_tool
from openai import OpenAI

from traderbot_ai.config import load_settings
from traderbot_ai.paths import display_agents_path, safe_agents_path


ImageDetail = Literal["low", "high", "auto", "original"]

MAX_IMAGE_BYTES = 12 * 1024 * 1024
SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _error(message: str, **data) -> dict:
    return {"ok": False, "error": message, **data}


def _mime_type_for_path(path: str) -> str | None:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type


def inspect_image_impl(
    path: str,
    question: str = "What do you see in this image?",
    detail: ImageDetail = "high",
    max_output_chars: int = 1200,
) -> dict:
    try:
        settings = load_settings()
        if not settings.openai_api_key_present:
            return _error("OPENAI_API_KEY is not configured")

        image_path = safe_agents_path(path)
        if not image_path.exists():
            return _error("image file does not exist", path=display_agents_path(image_path))
        if not image_path.is_file():
            return _error("path is not a file", path=display_agents_path(image_path))

        mime_type = _mime_type_for_path(str(image_path))
        if mime_type not in SUPPORTED_MIME_TYPES:
            return _error(
                "unsupported image type; use png, jpg, webp, or gif",
                path=display_agents_path(image_path),
                mime_type=mime_type,
            )

        size = image_path.stat().st_size
        if size <= 0:
            return _error("image file is empty", path=display_agents_path(image_path))
        if size > MAX_IMAGE_BYTES:
            return _error(
                "image file is too large",
                path=display_agents_path(image_path),
                bytes=size,
                max_bytes=MAX_IMAGE_BYTES,
            )

        data_url = f"data:{mime_type};base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = (
            "Inspect this image directly. Answer briefly and only from visible evidence. "
            "If it is a trading chart, mention chart type, price direction, volume if visible, "
            "and any obvious visual caveats. User question: "
            f"{question}"
        )

        response = OpenAI().responses.create(
            model=settings.vision_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url, "detail": detail},
                    ],
                }
            ],
            max_output_tokens=600,
            reasoning={"effort": settings.reasoning_effort},
            store=False,
        )
        analysis = (getattr(response, "output_text", "") or "").strip()
        if max_output_chars > 0:
            analysis = analysis[: max(1, min(max_output_chars, 6000))]

        return _ok(
            path=display_agents_path(image_path),
            mime_type=mime_type,
            bytes=size,
            model=settings.vision_model,
            detail=detail,
            analysis=analysis,
        )
    except Exception as error:
        return _error(str(error), path=path)


@function_tool
def inspect_image(
    path: str,
    question: str = "What do you see in this image?",
    detail: ImageDetail = "high",
    max_output_chars: int = 1200,
) -> dict:
    """Look at an image under agents-v2. Returns short visual notes."""
    return inspect_image_impl(
        path=path,
        question=question,
        detail=detail,
        max_output_chars=max_output_chars,
    )

