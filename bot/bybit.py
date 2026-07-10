"""Thin Bybit v5 client: REST (httpx, hand-rolled HMAC signing) + WS (websockets).

pybit was unavailable/unpinned at build time; the plan's sanctioned fallback is
~20 lines of HMAC signing over httpx + websockets. Demo/live contour = base URL
switch only (config.py). No uvloop, ProactorEventLoop-safe.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Awaitable, Callable

import httpx
import websockets

log = logging.getLogger("bot.bybit")

RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
# retCodes that indicate transient conditions worth retrying
RETRYABLE_RET = {10002, 10006, 10016}  # timestamp err, rate limit, server error


class BybitError(Exception):
    def __init__(self, ret_code: int, msg: str):
        super().__init__(f"retCode={ret_code}: {msg}")
        self.ret_code = ret_code


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


class BybitRest:
    def __init__(self, base_url: str, api_key: str = "", api_secret: str = "",
                 recv_window_ms: int = 5000, timeout_s: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = str(recv_window_ms)
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    async def close(self) -> None:
        await self.client.aclose()

    async def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                       auth: bool = False, max_tries: int = 5) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_tries):
            try:
                headers = {}
                if auth:
                    ts = str(int(time.time() * 1000))
                    if method == "GET":
                        qs = "&".join(f"{k}={v}" for k, v in params.items())
                        payload = ts + self.api_key + self.recv_window + qs
                    else:
                        payload = ts + self.api_key + self.recv_window + json.dumps(params)
                    headers = {
                        "X-BAPI-API-KEY": self.api_key,
                        "X-BAPI-TIMESTAMP": ts,
                        "X-BAPI-RECV-WINDOW": self.recv_window,
                        "X-BAPI-SIGN": _sign(self.api_secret, payload),
                    }
                if method == "GET":
                    resp = await self.client.get(path, params=params, headers=headers)
                else:
                    headers["Content-Type"] = "application/json"
                    resp = await self.client.post(path, content=json.dumps(params),
                                                  headers=headers)
                if resp.status_code in RETRYABLE_HTTP:
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}",
                                                request=resp.request, response=resp)
                data = resp.json()
                ret = int(data.get("retCode", -1))
                if ret == 0:
                    return data.get("result", {})
                if ret in RETRYABLE_RET and attempt < max_tries - 1:
                    raise BybitError(ret, data.get("retMsg", ""))
                raise BybitError(ret, data.get("retMsg", ""))
            except (httpx.TransportError, httpx.HTTPStatusError, BybitError) as exc:
                retryable = not isinstance(exc, BybitError) or exc.ret_code in RETRYABLE_RET
                last_exc = exc
                if not retryable or attempt == max_tries - 1:
                    raise
                log.warning("bybit %s %s failed (%s), retry in %.1fs", method, path, exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise last_exc  # pragma: no cover

    # ------------------------------------------------------------ market --
    async def server_time_ms(self) -> int:
        r = await self._request("GET", "/v5/market/time")
        return int(r["timeNano"]) // 1_000_000

    async def tickers_linear(self) -> list[dict]:
        r = await self._request("GET", "/v5/market/tickers", {"category": "linear"})
        return r.get("list", [])

    async def best_bid(self, symbol: str) -> float | None:
        """bid1Price from the v5 tickers endpoint (None if unavailable)."""
        r = await self._request("GET", "/v5/market/tickers",
                                {"category": "linear", "symbol": symbol})
        lst = r.get("list") or []
        bid = lst[0].get("bid1Price") if lst else None
        return float(bid) if bid else None

    async def kline(self, symbol: str, interval: str, limit: int = 200,
                    start: int | None = None, end: int | None = None) -> list[list[str]]:
        r = await self._request("GET", "/v5/market/kline",
                                {"category": "linear", "symbol": symbol,
                                 "interval": interval, "limit": limit,
                                 "start": start, "end": end})
        return r.get("list", [])  # newest first: [startMs, o, h, l, c, volume, turnover]

    async def open_interest(self, symbol: str, interval_time: str = "1h",
                            limit: int = 200) -> list[dict]:
        r = await self._request("GET", "/v5/market/open-interest",
                                {"category": "linear", "symbol": symbol,
                                 "intervalTime": interval_time, "limit": limit})
        return r.get("list", [])  # newest first: {openInterest, timestamp}

    async def instruments_linear(self) -> list[dict]:
        out, cursor = [], None
        while True:
            r = await self._request("GET", "/v5/market/instruments-info",
                                    {"category": "linear", "limit": 1000, "cursor": cursor})
            out.extend(r.get("list", []))
            cursor = r.get("nextPageCursor")
            if not cursor:
                return out

    async def announcements(self, limit: int = 20) -> list[dict]:
        r = await self._request("GET", "/v5/announcements/index",
                                {"locale": "en-US", "type": "delistings", "limit": limit})
        return r.get("list", [])

    # ------------------------------------------------------------- trade --
    async def place_order(self, **kw: Any) -> dict:
        return await self._request("POST", "/v5/order/create",
                                   {"category": "linear", **kw}, auth=True, max_tries=1)

    async def cancel_order(self, symbol: str, order_link_id: str) -> dict:
        return await self._request("POST", "/v5/order/cancel",
                                   {"category": "linear", "symbol": symbol,
                                    "orderLinkId": order_link_id}, auth=True, max_tries=1)

    async def open_orders(self) -> list[dict]:
        out, cursor = [], None
        while True:
            r = await self._request("GET", "/v5/order/realtime",
                                    {"category": "linear", "settleCoin": "USDT",
                                     "limit": 50, "cursor": cursor}, auth=True)
            out.extend(r.get("list", []))
            cursor = r.get("nextPageCursor")
            if not cursor:
                return out

    async def order_history(self, order_link_id: str) -> list[dict]:
        r = await self._request("GET", "/v5/order/history",
                                {"category": "linear", "orderLinkId": order_link_id},
                                auth=True)
        return r.get("list", [])

    async def positions(self) -> list[dict]:
        out, cursor = [], None
        while True:
            r = await self._request("GET", "/v5/position/list",
                                    {"category": "linear", "settleCoin": "USDT",
                                     "limit": 200, "cursor": cursor}, auth=True)
            out.extend(r.get("list", []))
            cursor = r.get("nextPageCursor")
            if not cursor:
                return out

    async def wallet_equity_usd(self) -> float:
        r = await self._request("GET", "/v5/account/wallet-balance",
                                {"accountType": "UNIFIED"}, auth=True)
        acct = (r.get("list") or [{}])[0]
        return float(acct.get("totalEquity") or 0.0)


class BybitWS:
    """Reconnecting WS consumer. on_message receives every parsed data frame;
    on_reconnect fires after each (re)connect (used for REST re-sync)."""

    def __init__(self, url: str, topics: list[str],
                 on_message: Callable[[dict], Awaitable[None]],
                 api_key: str = "", api_secret: str = "",
                 on_reconnect: Callable[[], Awaitable[None]] | None = None,
                 name: str = "ws"):
        self.url = url
        self.topics = topics
        self.on_message = on_message
        self.on_reconnect = on_reconnect
        self.api_key = api_key
        self.api_secret = api_secret
        self.name = name
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, ping_interval=20,
                                              ping_timeout=15) as ws:
                    if self.api_key:
                        expires = int(time.time() * 1000) + 10_000
                        sig = _sign(self.api_secret, f"GET/realtime{expires}")
                        await ws.send(json.dumps({"op": "auth",
                                                  "args": [self.api_key, expires, sig]}))
                        await ws.recv()  # auth ack
                    # Bybit caps args per subscribe request; chunk to be safe
                    for i in range(0, len(self.topics), 10):
                        await ws.send(json.dumps({"op": "subscribe",
                                                  "args": self.topics[i:i + 10]}))
                    log.info("%s connected: %d topics", self.name, len(self.topics))
                    if self.on_reconnect:
                        await self.on_reconnect()
                    delay = 1.0
                    while not self._stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        msg = json.loads(raw)
                        if "topic" in msg:
                            await self.on_message(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop.is_set():
                    return
                log.warning("%s dropped (%s); reconnect in %.0fs", self.name, exc, delay)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, 60.0)
