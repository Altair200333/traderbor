"""Telegram notifier: plain Bot API over HTTPS (no framework), queued sends,
rate-limit + dedupe, failure-only heartbeat semantics handled by callers.

No token configured => log-only (dev mode). The token/chat id come ONLY from
the environment (.env); nothing is ever hardcoded.
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

log = logging.getLogger("bot.notify")

DEDUPE_S = 120.0
MIN_INTERVAL_S = 1.05          # Telegram per-chat rate courtesy


class Notifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.token = bot_token
        self.chat_id = chat_id
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        self._recent: dict[str, float] = {}
        self._client: httpx.AsyncClient | None = None

    async def send(self, text: str) -> None:
        now = time.monotonic()
        last = self._recent.get(text)
        if last is not None and now - last < DEDUPE_S:
            return
        self._recent[text] = now
        if len(self._recent) > 200:
            cutoff = now - DEDUPE_S
            self._recent = {k: v for k, v in self._recent.items() if v > cutoff}
        try:
            self.queue.put_nowait(text)
        except asyncio.QueueFull:
            log.warning("notify queue full, dropping: %s", text[:80])

    async def alarm(self, text: str) -> None:
        await self.send(f"\U0001F6A8 ALARM: {text}")

    async def run(self) -> None:
        """Worker task: drain the queue at a polite rate."""
        if not self.token or not self.chat_id:
            log.info("telegram not configured: notifications go to log only")
        self._client = httpx.AsyncClient(timeout=15.0)
        try:
            while True:
                text = await self.queue.get()
                log.info("notify: %s", text)
                if self.token and self.chat_id:
                    await self._post(text)
                await asyncio.sleep(MIN_INTERVAL_S)
        finally:
            await self._client.aclose()

    async def _post(self, text: str) -> None:
        assert self._client is not None
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for attempt in range(3):
            try:
                r = await self._client.post(url, json={"chat_id": self.chat_id,
                                                       "text": text[:4000]})
                if r.status_code == 429:
                    retry = float(r.json().get("parameters", {}).get("retry_after", 5))
                    await asyncio.sleep(retry)
                    continue
                if r.status_code != 200:
                    log.warning("telegram HTTP %s: %s", r.status_code, r.text[:200])
                return
            except httpx.TransportError as exc:
                log.warning("telegram send failed (%s), attempt %d", exc, attempt + 1)
                await asyncio.sleep(2.0 * (attempt + 1))
