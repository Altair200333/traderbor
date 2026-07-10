"""Execution defect-2 coverage: entry price = floor_to_tick(min(close, bid)),
bounded PostOnly retries within TTL, postonly_reject journaled per occurrence,
lost create ack resolved by orderLinkId query, reconnect entry-block, and
reservation release."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from bot.bybit import BybitError
from bot.config import BotConfig
from bot.execution import Executor
from bot.journal import link_id, now_ms
from bot.risk import Decision, RiskManager
from bot.strategy.base import EntryIntent
from bot.universe import Instrument, Universe


class FakeRest:
    def __init__(self, bids: list[float | None]):
        self.bids = list(bids)          # successive best_bid answers
        self.placed: list[dict] = []    # kwargs of each accepted place_order
        self.reject_places = 0          # BybitError for the next N places
        self.drop_places = 0            # TransportError (lost ack) for next N
        self.fail_open_orders = False
        self.history: list[dict] = []   # order_history answer

    async def best_bid(self, symbol: str) -> float | None:
        return self.bids.pop(0) if self.bids else None

    async def place_order(self, **kw):
        if self.drop_places > 0:
            self.drop_places -= 1
            raise httpx.ConnectError("timed out")
        if self.reject_places > 0:
            self.reject_places -= 1
            raise BybitError(110017, "rejected")
        self.placed.append(kw)
        return {"orderId": f"oid{len(self.placed)}"}

    async def cancel_order(self, symbol, order_link_id):
        return {}

    async def order_history(self, order_link_id):
        return self.history

    async def open_orders(self):
        if self.fail_open_orders:
            raise RuntimeError("exchange down")
        return []

    async def positions(self):
        return []


class FakeNotify:
    def __init__(self):
        self.sent: list[str] = []
        self.alarms: list[str] = []

    async def send(self, msg: str) -> None:
        self.sent.append(msg)

    async def alarm(self, msg: str) -> None:
        self.alarms.append(msg)


def make_executor(journal, bids):
    cfg = BotConfig()
    rest = FakeRest(bids)
    uni = Universe(journal=journal)
    uni.instruments["ADAUSDT"] = Instrument(pair="ADAUSDT", bybit_symbol="ADAUSDT",
                                            tick_size=0.0001, qty_step=1.0,
                                            min_qty=1.0, tradeable=True)
    notify = FakeNotify()
    ex = Executor(cfg, rest, journal, uni, RiskManager(cfg.risk, journal), notify)
    return ex, rest, notify


LK = link_id("liqrev_v2", "ADAUSDT", 1000)


def entry_intent(price: float = 1.0, ttl_s: int = 3600) -> EntryIntent:
    return EntryIntent(strategy="liqrev_v2", symbol="ADAUSDT", side="Buy",
                       limit_price=price, weight=1.0, ttl_s=ttl_s, stop_pct=0.20,
                       exit_at_ms=now_ms() + 86_400_000,
                       meta={"signal_ts_ms": 1000})


async def _submit(ex, mode: str = "live", ttl_s: int = 3600,
                  dec: Decision | None = None) -> None:
    dec = dec or Decision(True, "", qty=100.0, notional=100.0, weight=1.0)
    await ex.submit_entry(entry_intent(ttl_s=ttl_s), dec, mode)
    for t in ex._timers.values():
        t.cancel()


def cancel_msg(lk: str) -> dict:
    return {"topic": "order",
            "data": [{"orderLinkId": lk, "orderStatus": "Cancelled",
                      "cancelType": "CancelByPostOnly",
                      "cumExecQty": "0", "avgPrice": ""}]}


def po_reject_events(journal) -> int:
    row = journal.db.execute(
        "SELECT COUNT(*) AS c FROM events WHERE kind='postonly_reject'").fetchone()
    return row["c"]


# ------------------------------------------------------------ entry price ---
def test_entry_price_is_min_close_bid(journal):
    ex, rest, _ = make_executor(journal, bids=[0.98])
    asyncio.run(_submit(ex))
    assert rest.placed[0]["price"] == "0.98"
    o = journal.get_order(LK)
    assert o["price"] == pytest.approx(0.98) and o["status"] == "open"
    # deploy-spec: disaster stop = -20% FROM ENTRY (re-anchored at bid entry)
    assert float(rest.placed[0]["stopLoss"]) == pytest.approx(0.98 * 0.8)


def test_entry_price_keeps_close_when_bid_above(journal):
    ex, rest, _ = make_executor(journal, bids=[1.05])
    asyncio.run(_submit(ex))
    assert rest.placed[0]["price"] == "1"


def test_entry_price_falls_back_without_bid(journal):
    ex, rest, _ = make_executor(journal, bids=[None])
    asyncio.run(_submit(ex))
    assert rest.placed[0]["price"] == "1"


def test_entry_price_floored_to_tick(journal):
    ex, rest, _ = make_executor(journal, bids=[0.98763])
    ex.universe.instruments["ADAUSDT"].tick_size = 0.001
    asyncio.run(_submit(ex))
    assert rest.placed[0]["price"] == "0.987"


# --------------------------------------------------------- PostOnly retry ---
def test_postonly_reject_retries_twice_then_gives_up(journal):
    ex, rest, notify = make_executor(journal, bids=[0.98, 0.95, 0.94])
    asyncio.run(_submit(ex))
    # first PostOnly cancel: retry 1 at the fresh best bid
    asyncio.run(ex.on_private_message(cancel_msg(LK)))
    assert len(rest.placed) == 2 and rest.placed[1]["price"] == "0.95"
    o = journal.get_order(LK)
    assert o["status"] == "open" and o["price"] == pytest.approx(0.95)
    assert po_reject_events(journal) == 1
    # second PostOnly cancel: retry 2
    asyncio.run(ex.on_private_message(cancel_msg(LK)))
    assert len(rest.placed) == 3 and rest.placed[2]["price"] == "0.94"
    assert po_reject_events(journal) == 2
    # third PostOnly cancel: bounded — give up, rejected + alarm
    asyncio.run(ex.on_private_message(cancel_msg(LK)))
    assert len(rest.placed) == 3
    assert journal.get_order(LK)["status"] == "rejected"
    assert po_reject_events(journal) == 3
    assert notify.alarms


def test_postonly_reject_past_ttl_no_retry(journal):
    ex, rest, _ = make_executor(journal, bids=[0.98, 0.95])
    asyncio.run(_submit(ex, ttl_s=0))
    asyncio.run(ex.on_private_message(cancel_msg(LK)))
    assert len(rest.placed) == 1                    # no retry outside TTL window
    assert journal.get_order(LK)["status"] == "cancelled"
    assert po_reject_events(journal) == 1


def test_postonly_retry_place_error_gives_up(journal):
    ex, rest, notify = make_executor(journal, bids=[0.98, 0.95])
    asyncio.run(_submit(ex))
    rest.reject_places = 1                          # retry placement errors
    asyncio.run(ex.on_private_message(cancel_msg(LK)))
    assert len(rest.placed) == 1
    assert journal.get_order(LK)["status"] == "rejected"
    assert po_reject_events(journal) == 1
    assert notify.alarms


# ------------------------------------------------------- unknown create ack -
def test_unknown_create_ack_marks_rejected_when_absent(journal):
    ex, rest, notify = make_executor(journal, bids=[0.98])
    rest.drop_places = 1                            # ack lost, order not found
    asyncio.run(_submit(ex))
    assert rest.placed == []                        # never double-sent
    assert journal.get_order(LK)["status"] == "rejected"
    assert notify.alarms


def test_unknown_create_ack_adopts_live_order(journal):
    ex, rest, _ = make_executor(journal, bids=[0.98])
    rest.drop_places = 1
    rest.history = [{"orderStatus": "New", "cumExecQty": "0", "avgPrice": ""}]
    asyncio.run(_submit(ex))
    assert rest.placed == []                        # never double-sent
    assert journal.get_order(LK)["status"] == "open"


# -------------------------------------------------- reservation / reconnect -
def test_reservation_released_after_submit(journal):
    ex, rest, _ = make_executor(journal, bids=[0.98])
    it = entry_intent()
    dec = ex.risk.evaluate(it, 5000, "live", 1.0, 1.0)
    assert dec.approved and dec.reserve_key == LK
    assert ex.risk._reserved                        # headroom held
    asyncio.run(_submit(ex, dec=dec))
    assert ex.risk._reserved == {}                  # journal row took over
    assert journal.get_order(LK)["status"] == "open"


def test_reconcile_blocks_entries_until_done(journal):
    ex, rest, _ = make_executor(journal, bids=[])
    ex.cfg.bybit.api_key = "k"
    rest.fail_open_orders = True
    with pytest.raises(RuntimeError):
        asyncio.run(ex.reconcile())
    # failed reconcile: entries stay blocked until the next successful run
    dec = ex.risk.evaluate(entry_intent(), 5000, "live", 1.0, 1.0)
    assert not dec.approved and "blocked" in dec.reason
    rest.fail_open_orders = False
    asyncio.run(ex.reconcile())
    assert ex.risk.evaluate(entry_intent(), 5000, "live", 1.0, 1.0).approved


def test_paper_entry_unchanged(journal):
    ex, rest, _ = make_executor(journal, bids=[0.98])
    asyncio.run(_submit(ex, mode="paper"))
    assert rest.placed == []                        # no exchange call in paper
    o = journal.get_order(LK)
    assert o["price"] == 1.0 and o["status"] == "open"
