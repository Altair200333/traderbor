"""Bot entrypoint: single asyncio process, task supervisor, graceful shutdown.

Windows-dev / Linux-prod portability (plan): pure asyncio (no uvloop),
ProactorEventLoop-safe; shutdown via SIGTERM (systemd) AND KeyboardInterrupt
(Windows Ctrl+C); filelock single-instance guard; clock drift checked against
/v5/market/time; RotatingFileHandler logging.

Run: python -m bot
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import signal
import sys
import time

from filelock import FileLock, Timeout

from bot.bybit import BybitRest
from bot.config import BotConfig, load_config
from bot.execution import Executor, PaperBroker
from bot.feed import (Engine, SnapshotLoop, backfill, daily_summary,
                      make_liq_logger, make_private_ws, unlock_refresh_loop,
                      watchdog)
from bot.journal import Journal
from bot.notify import Notifier
from bot.risk import RiskManager
from bot.strategy.liqrev_v2 import LiqrevStrategy
from bot.strategy.unlock_watch import UnlockWatch
from bot.universe import Universe

log = logging.getLogger("bot")


def setup_logging(cfg: BotConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    fh = logging.handlers.RotatingFileHandler(cfg.data_dir / "bot.log",
                                              maxBytes=10_000_000, backupCount=5,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(cfg.log_level.upper())
    root.addHandler(fh)
    root.addHandler(sh)


async def check_clock(rest: BybitRest, max_drift_s: float) -> float:
    server_ms = await rest.server_time_ms()
    drift = abs(time.time() * 1000 - server_ms) / 1000.0
    if drift > max_drift_s:
        raise SystemExit(f"clock drift {drift:.1f}s > {max_drift_s}s — fix NTP, refusing to trade")
    return drift


async def async_main(cfg: BotConfig) -> None:
    journal = Journal(cfg.data_dir / "journal.sqlite")
    notify = Notifier(cfg.telegram.bot_token, cfg.telegram.chat_id)
    rest = BybitRest(cfg.bybit.rest_base, cfg.bybit.api_key, cfg.bybit.api_secret,
                     cfg.bybit.recv_window_ms)

    drift = await check_clock(rest, cfg.clock_max_drift_s)
    log.info("clock drift %.2fs vs %s", drift, cfg.bybit.rest_base)

    universe = Universe.load(journal, cfg.liqrev.liq_gate_usd)
    universe.apply_instruments_info(await rest.instruments_linear())
    log.info("universe: %d pairs, %d tradeable on Bybit",
             len(universe.instruments), len(universe.tradeable_pairs()))

    risk = RiskManager(cfg.risk, journal)
    executor = Executor(cfg, rest, journal, universe, risk, notify)
    paper = PaperBroker(cfg, rest, journal, universe, notify)
    strategies = []
    liqrev = LiqrevStrategy(cfg.liqrev, journal)
    strategies.append(liqrev)
    unlock = UnlockWatch(cfg.unlock, journal, universe.pairs(), notify)
    strategies.append(unlock)
    engine = Engine(cfg, journal, universe, risk, executor, paper, strategies, notify)
    snap = SnapshotLoop(cfg, rest, journal, universe, engine, notify)

    sha = cfg.public_sha256()
    journal.event("info", "startup", f"config sha256={sha}")
    await notify.send(f"bot started ({cfg.bybit.contour}; liqrev={cfg.liqrev.mode} "
                      f"unlock={cfg.unlock.mode}) config sha256={sha[:16]}")

    await backfill(cfg, rest, journal, universe)
    if cfg.bybit.api_key:
        await executor.reconcile()
        eq = await rest.wallet_equity_usd()
        if eq > 0:
            risk.check_kill_switch(eq, "live")

    tasks = [
        asyncio.create_task(notify.run(), name="notify"),
        asyncio.create_task(snap.run(), name="snapshot"),
        asyncio.create_task(make_liq_logger(cfg, journal, universe).run(), name="liq_ws"),
        asyncio.create_task(watchdog(journal, notify), name="watchdog"),
        asyncio.create_task(daily_summary(cfg, journal, notify, rest), name="summary"),
        asyncio.create_task(unlock_refresh_loop(unlock, cfg.unlock.refresh_utc_hour),
                            name="unlock_refresh"),
    ]
    if cfg.bybit.api_key:
        tasks.append(asyncio.create_task(make_private_ws(cfg, executor).run(),
                                         name="private_ws"))

    stop = asyncio.Event()
    try:  # SIGTERM for systemd; not available on Windows (KeyboardInterrupt instead)
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    except (NotImplementedError, AttributeError):
        pass

    try:
        done, _ = await asyncio.wait(
            [*tasks, asyncio.create_task(stop.wait(), name="stop")],
            return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.get_name() != "stop" and t.exception():
                log.error("task %s died: %r", t.get_name(), t.exception())
                journal.event("alarm", "task_died", f"{t.get_name()}: {t.exception()!r}")
    finally:
        log.info("shutting down")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await rest.close()
        journal.close()


def run() -> None:
    cfg = load_config()
    setup_logging(cfg)
    lock = FileLock(str(cfg.data_dir / "bot.lock"))
    try:
        lock.acquire(timeout=0.1)
    except Timeout:
        print("another bot instance holds the lock — refusing to start (double orders)",
              file=sys.stderr)
        raise SystemExit(2)
    try:
        asyncio.run(async_main(cfg))
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — clean exit (reconcile-on-start makes this safe)")
    finally:
        lock.release()


if __name__ == "__main__":
    run()
