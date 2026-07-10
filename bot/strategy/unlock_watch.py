"""Unlock watch (S2, PAPER ONLY): post-cliff LONG on unlock events >= 3% of
supply, entry after the cliff, hold ~7 days.

Spec: docs/notes/2026-07-09/unlock-liqrev-glue.md (Q18 S2 leg: DEV +281bps ->
LIVE +605bps/trade, n thin => paper only) + research/scanner_lab/
unlock_trade_sim.py (entry close of event_date, exit close of event_date+7,
threshold frac_supply >= 0.03).

Data source: DefiLlama FREE datasets CDN (same access pattern as
research/scanner_lab/fetch_unlocks.py):
  https://defillama-datasets.llama.fi/emissionsProtocolsList  -> slugs
  https://defillama-datasets.llama.fi/emissions/{slug}        -> unlockEvents
(api.llama.fi/emissions* is a PAID endpoint — never used.)

Daily refresh writes cliff events for universe-matched tokens to the journal
and notifies upcoming >= 3% cliffs. on_bar emits paper entry intents on the
first bar after a cliff date (00:00 UTC entry approximates the research
daily-close convention; documented paper-level approximation).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import httpx

from bot.config import UnlockConfig
from bot.journal import Journal
from bot.strategy.base import EntryIntent, Intent, MarketState

log = logging.getLogger("bot.unlock")

CDN = "https://defillama-datasets.llama.fi"
UA = {"User-Agent": "Mozilla/5.0 research"}
DAY_MS = 86_400_000
H_MS = 3_600_000


def extract_cliffs(payload: dict, symbol: str, slug: str,
                   min_frac: float) -> list[tuple[str, str, str, float]]:
    """(slug, symbol, event_date, frac_supply) rows from an emissions payload.
    Cliff = discrete tranche (linear drips are not events), fraction of max
    supply — same convention as research fetch_unlocks.py."""
    meta = payload.get("metadata", {}) or {}
    denom = 0.0
    supply = (payload.get("supplyMetrics") or {}).get("maxSupply")
    if supply:
        denom = float(supply)
    if not denom and meta.get("total"):
        denom = float(meta["total"])
    if denom <= 0:
        return []
    rows = []
    for ev in meta.get("unlockEvents") or []:
        ts = ev.get("timestamp")
        if not ts:
            continue
        cliff = 0.0
        summary = ev.get("summary") or {}
        if summary.get("totalTokensCliff"):
            cliff = float(summary["totalTokensCliff"])
        else:
            cliff = sum(float(a.get("amount") or 0)
                        for a in ev.get("cliffAllocations") or [])
        frac = cliff / denom
        if frac >= min_frac:
            date = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
            rows.append((slug, symbol, date, frac))
    return rows


class UnlockWatch:
    name = "unlock_s2"

    def __init__(self, cfg: UnlockConfig, journal: Journal,
                 universe_pairs: list[str], notifier=None):
        self.cfg = cfg
        self.mode = cfg.mode
        self.journal = journal
        self.symbols = {p.removesuffix("USDT"): p for p in universe_pairs}
        self.notifier = notifier

    # ------------------------------------------------------ daily refresh --
    async def refresh_calendar(self) -> int:
        """Fetch the DefiLlama unlock calendar; store universe cliffs >= 3%.
        Symbol mapping follows fetch_unlocks.py: slug -> symbol and
        gecko_id -> symbol via the free api.llama.fi/protocols list."""
        async with httpx.AsyncClient(timeout=60.0, headers=UA) as client:
            protocols = (await client.get("https://api.llama.fi/protocols")).json()
            slug2sym: dict[str, str] = {}
            gecko2sym: dict[str, str] = {}
            for p in protocols:
                s = p.get("symbol")
                if not s or s == "-":
                    continue
                if p.get("slug"):
                    slug2sym[p["slug"]] = s.upper()
                if p.get("gecko_id"):
                    gecko2sym[p["gecko_id"]] = s.upper()
            slugs = (await client.get(f"{CDN}/emissionsProtocolsList")).json()
            rows: list[tuple[str, str, str, float]] = []
            for slug in slugs:
                try:
                    r = await client.get(f"{CDN}/emissions/{slug}")
                    if r.status_code != 200:
                        continue
                    payload = r.json()
                except (httpx.HTTPError, json.JSONDecodeError):
                    continue
                sym = (gecko2sym.get(payload.get("gecko_id") or "")
                       or slug2sym.get(slug) or "")
                pair = self.symbols.get(sym)
                if pair is None:
                    continue
                rows.extend(extract_cliffs(payload, pair, slug, self.cfg.min_frac_supply))
                await asyncio.sleep(0.2)      # be polite to the free CDN
        if rows:
            self.journal.write_unlock_events(rows)
        self.journal.kv_set("unlock_last_refresh",
                            datetime.now(timezone.utc).isoformat())
        upcoming = [r for r in rows
                    if datetime.now(timezone.utc).strftime("%Y-%m-%d") <= r[2]
                    <= (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%d")]
        if upcoming and self.notifier:
            lines = [f"{sym} {date} {frac:.1%}" for _, sym, date, frac in upcoming[:15]]
            await self.notifier.send("Unlock cliffs >=3% next 14d:\n" + "\n".join(lines))
        log.info("unlock refresh: %d universe cliff rows", len(rows))
        return len(rows)

    # -------------------------------------------------------------- on_bar --
    def on_bar(self, market: MarketState) -> list[Intent]:
        """First bar at/after a cliff date 00:00 UTC -> paper long, hold 7d."""
        if self.mode == "off":
            return []
        intents: list[Intent] = []
        bar_close = market.bar_close_ms
        for ev in self.journal.unlock_events():
            event_start_ms = int(datetime.strptime(ev["event_date"], "%Y-%m-%d")
                                 .replace(tzinfo=timezone.utc).timestamp() * 1000)
            entry_due = event_start_ms + DAY_MS       # close of event_date (UTC)
            if not (entry_due <= bar_close < entry_due + H_MS):
                continue
            sym = ev["symbol"]
            if self.journal.last_signal_ms(self.name, sym) is not None and \
                    bar_close - (self.journal.last_signal_ms(self.name, sym) or 0) < 30 * DAY_MS:
                continue                              # 30d same-symbol dedupe
            s = market.series(sym, 2)
            if s is None or not s.close or s.bar_ms[-1] != market.bar_open_ms:
                continue
            px = float(s.close[-1])
            self.journal.write_signal(strategy=self.name, mode=self.mode, symbol=sym,
                                      ts_ms=bar_close, weight=1.0, limit_px=px,
                                      mw_tag=f"frac={ev['frac_supply']:.3f}")
            intents.append(EntryIntent(
                strategy=self.name, symbol=sym, side="Buy", limit_price=px,
                weight=1.0, ttl_s=3600, stop_pct=0.99,   # no stop in spec; inert placeholder
                exit_at_ms=bar_close + self.cfg.hold_days * DAY_MS,
                meta={"frac_supply": ev["frac_supply"], "event_date": ev["event_date"],
                      "signal_ts_ms": bar_close}))
            log.info("unlock S2 paper entry %s @%s (cliff %s, %.1f%%)",
                     sym, px, ev["event_date"], ev["frac_supply"] * 100)
        return intents
