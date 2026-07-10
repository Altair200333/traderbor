"""Feeds: hourly REST snapshot loop (the detector clock), cold-start backfill,
private WS dispatch, allLiquidation logger, watchdog + daily summary.

The detector runs on deterministic hourly REST snapshots (plan decision:
replayable, immune to WS-gap subtleties). Sockets carry fills (instant
Telegram) and the liquidation research feed only.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from bot.bybit import BybitError, BybitRest, BybitWS
from bot.config import BotConfig
from bot.execution import Executor, PaperBroker
from bot.journal import Journal, now_ms
from bot.notify import Notifier
from bot.risk import RiskManager
from bot.strategy.base import EntryIntent, MarketState, Strategy, SymbolSeries
from bot.universe import Universe

log = logging.getLogger("bot.feed")

H_MS = 3_600_000
DAY_MS = 86_400_000
SNAPSHOT_DELAY_S = 5.0        # poll at :00:05, bar is closed by then


class JournalMarketState(MarketState):
    """MarketState view over journal snapshots + the universe gate."""

    def __init__(self, journal: Journal, universe: Universe, bar_open_ms: int,
                 equity_usd: float):
        self.journal = journal
        self.universe = universe
        self.bar_open_ms = bar_open_ms
        self.bar_close_ms = bar_open_ms + H_MS
        self.equity_usd = equity_usd

    def symbols(self) -> list[str]:
        return self.universe.tradeable_pairs()

    def series(self, symbol: str, n_bars: int) -> SymbolSeries | None:
        rows = self.journal.get_series(symbol, n_bars)
        if not rows:
            return None
        return SymbolSeries(bar_ms=[r["bar_ms"] for r in rows],
                            close=[r["close"] for r in rows],
                            oi=[r["oi"] for r in rows])

    def liquidity_ok(self, symbol: str) -> bool:
        return self.universe.liquidity_ok(symbol, self.bar_close_ms)


class Engine:
    """Wires: snapshot tick -> strategies -> risk -> execution/paper."""

    def __init__(self, cfg: BotConfig, journal: Journal, universe: Universe,
                 risk: RiskManager, executor: Executor, paper: PaperBroker,
                 strategies: list[Strategy], notify: Notifier):
        self.cfg = cfg
        self.journal = journal
        self.universe = universe
        self.risk = risk
        self.executor = executor
        self.paper = paper
        self.strategies = strategies
        self.notify = notify

    def equity(self, mode: str) -> float:
        if mode == "live":
            return self.journal.last_equity("live") or 0.0
        return self.journal.last_equity("paper") or self.cfg.paper_equity_usd

    async def on_bar(self, bar_open_ms: int) -> None:
        # paper broker first: resolves last hour's pending paper orders/exits
        await self.paper.on_bar(bar_open_ms)
        for strat in self.strategies:
            if strat.mode == "off":
                continue
            mkt = JournalMarketState(self.journal, self.universe, bar_open_ms,
                                     self.equity(strat.mode))
            try:
                intents = strat.on_bar(mkt)
            except Exception:
                log.exception("strategy %s failed on bar %d", strat.name, bar_open_ms)
                self.journal.event("alarm", "strategy_error", f"{strat.name} on_bar failed")
                continue
            for intent in intents:
                if not isinstance(intent, EntryIntent):
                    continue
                ins = self.universe.instruments.get(intent.symbol)
                dec = self.risk.evaluate(intent, mkt.equity_usd, strat.mode,
                                         ins.qty_step if ins else 0.0,
                                         ins.min_qty if ins else 0.0)
                self.journal.set_signal_result(   # strategy already wrote the row
                    intent.strategy, intent.symbol,
                    intent.meta.get("signal_ts_ms", mkt.bar_close_ms),
                    dec.approved, dec.reason or None)
                if not dec.approved:
                    log.info("VETO %s %s: %s", strat.name, intent.symbol, dec.reason)
                    await self.notify.send(f"veto {intent.symbol} ({strat.name}): {dec.reason}")
                    continue
                await self.executor.submit_entry(intent, dec, strat.mode)
            if strat.mode == "live":
                self.risk.check_fill_rate_gate(strat.name)


class SnapshotLoop:
    def __init__(self, cfg: BotConfig, rest: BybitRest, journal: Journal,
                 universe: Universe, engine: Engine, notify: Notifier):
        self.cfg = cfg
        self.rest = rest
        self.journal = journal
        self.universe = universe
        self.engine = engine
        self.notify = notify

    async def run(self) -> None:
        while True:
            now = time.time()
            next_tick = (int(now // 3600) + 1) * 3600 + SNAPSHOT_DELAY_S
            await asyncio.sleep(max(0.5, next_tick - now))
            bar_open_ms = (int(next_tick) // 3600 - 1) * 3600 * 1000
            try:
                await self.snapshot(bar_open_ms)
            except (BybitError, Exception):
                log.exception("snapshot tick failed for bar %d", bar_open_ms)
                self.journal.event("alarm", "snapshot_fail",
                                   f"missed bar {bar_open_ms} (skip; no wrong orders)")
                await self.notify.alarm(f"snapshot tick FAILED for bar "
                                        f"{datetime.fromtimestamp(bar_open_ms/1000, tz=timezone.utc):%H:%M}")
                continue
            await self.engine.on_bar(bar_open_ms)

    async def snapshot(self, bar_open_ms: int) -> None:
        tickers = await self.rest.tickers_linear()
        by_sym = {t["symbol"]: t for t in tickers}
        rows = []
        for pair, ins in self.universe.instruments.items():
            t = by_sym.get(ins.bybit_symbol)
            if t is None:
                continue
            rows.append({"symbol": pair,
                         "close": float(t["lastPrice"]),
                         "mark": float(t.get("markPrice") or 0) or None,
                         "oi": float(t.get("openInterest") or 0) or None,
                         "funding": float(t.get("fundingRate") or 0) or None})
        self.journal.write_snapshots(bar_open_ms, rows)
        log.info("snapshot bar=%d symbols=%d", bar_open_ms, len(rows))


async def backfill(cfg: BotConfig, rest: BybitRest, journal: Journal,
                   universe: Universe) -> None:
    """Cold start: hourly close+OI history (detector needs 6h) and 31 daily
    quote volumes (liquidity gate) per symbol, via REST klines/open-interest."""
    last = journal.last_bar_ms()
    fresh = last is not None and now_ms() - last < 2 * H_MS
    n_bars = 8
    for pair, ins in universe.instruments.items():
        if not ins.tradeable:
            continue
        try:
            if not fresh:
                kl = await rest.kline(ins.bybit_symbol, "60", limit=n_bars)
                oi_rows = await rest.open_interest(ins.bybit_symbol, "1h", limit=n_bars)
                oi_by_ms = {int(r["timestamp"]): float(r["openInterest"]) for r in oi_rows}
                for r in kl:                     # newest first
                    bar_ms = int(r[0])
                    if bar_ms + H_MS > now_ms():
                        continue                 # still-forming bar
                    journal.write_snapshots(bar_ms, [{
                        "symbol": pair, "close": float(r[4]),
                        # open-interest endpoint stamps interval START; the bar's
                        # closing OI is the next interval's value
                        "oi": oi_by_ms.get(bar_ms + H_MS, oi_by_ms.get(bar_ms))}])
            dk = await rest.kline(ins.bybit_symbol, "D", limit=31)
            journal.write_daily_volume(
                pair, [(int(r[0]), float(r[6])) for r in dk])
        except BybitError as exc:
            log.warning("backfill %s failed: %s", pair, exc)
        await asyncio.sleep(0.12)                # stay far under rate limits
    log.info("backfill done (fresh=%s)", fresh)


def make_liq_logger(cfg: BotConfig, journal: Journal, universe: Universe) -> BybitWS:
    """Public allLiquidation stream -> journal (research gold, day-1)."""
    topics = [f"allLiquidation.{ins.bybit_symbol}"
              for ins in universe.instruments.values() if ins.tradeable]

    async def on_msg(msg: dict) -> None:
        rows = []
        for d in msg.get("data", []):
            rows.append((int(d.get("T") or now_ms()), d.get("s", ""),
                         d.get("S", ""), float(d.get("p") or 0), float(d.get("v") or 0)))
        if rows:
            journal.write_liq_prints(rows)

    return BybitWS(cfg.bybit.ws_public, topics, on_msg, name="liq_ws")


def make_private_ws(cfg: BotConfig, executor: Executor) -> BybitWS:
    return BybitWS(cfg.bybit.ws_private, ["order", "execution", "wallet"],
                   executor.on_private_message,
                   api_key=cfg.bybit.api_key, api_secret=cfg.bybit.api_secret,
                   on_reconnect=executor.reconcile, name="private_ws")


async def watchdog(journal: Journal, notify: Notifier) -> None:
    """Failure-only heartbeat: alarm when the snapshot clock stalls."""
    while True:
        await asyncio.sleep(600)
        last = journal.last_bar_ms()
        if last is None:
            continue
        age_min = (now_ms() - (last + H_MS)) / 60000
        if age_min > 70:
            await notify.alarm(f"no snapshot tick for {age_min:.0f} min")


async def daily_summary(cfg: BotConfig, journal: Journal, notify: Notifier,
                        rest: BybitRest) -> None:
    """21:00 UTC: balance, open slots, signals count (plan component 9)."""
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=21, minute=0, second=30, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        live_eq = journal.last_equity("live")
        paper_eq = journal.last_equity("paper") or cfg.paper_equity_usd
        n_open = len(journal.open_positions())
        day_ago = now_ms() - DAY_MS
        n_sig = journal.db.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE ts_ms>?", (day_ago,)).fetchone()["c"]
        await notify.send(
            f"daily: live=${live_eq:,.0f} paper=${paper_eq:,.0f} "
            f"open={n_open} signals24h={n_sig}"
            if live_eq is not None else
            f"daily: paper=${paper_eq:,.0f} open={n_open} signals24h={n_sig}")


async def unlock_refresh_loop(unlock_strategy, refresh_hour: int) -> None:
    while True:
        now = datetime.now(timezone.utc)
        target_s = ((refresh_hour - now.hour) % 24) * 3600 - now.minute * 60 - now.second
        if target_s <= 0:
            target_s += 86400
        await asyncio.sleep(target_s)
        try:
            await unlock_strategy.refresh_calendar()
        except Exception:
            log.exception("unlock calendar refresh failed")
