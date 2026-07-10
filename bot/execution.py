"""Order lifecycle: PostOnly entry + server-side stop, TTL self-cancel,
time-based exit, reconcile-on-start, and the paper broker.

Live rules (deploy-spec section 1): entry = PostOnly limit BUY at
floor_to_tick(min(trigger-bar close, best bid)) — reduces but does NOT
eliminate marketable-PostOnly cancels (marketability is decided in the matching
engine vs best ask at arrival; the ask can fall while the request is in
flight); entry price improves vs the research trigger-close assumption only
CONDITIONAL on fill, and fill probability drops (audit defect 2). Stop stays
anchored at trigger_close - 20% (the research stop level), attached stopLoss
(slOrderType=Market, survives bot death). A PostOnly cancel retries up to 2x at
the fresh best bid within the original TTL window, then the entry dies
unfilled. Order-status truth is the private order stream; a lost create ack is
resolved by querying orderLinkId — never double-sent. TTL cancels the unfilled
remainder at +1h (partial fills keep their stop and exit on actual size);
exit = market reduce-only at trigger close + 24h.
Paper mode mirrors the research fill model: filled iff next 1h bar low < limit.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass

import httpx

from bot.bybit import BybitError, BybitRest
from bot.config import BotConfig
from bot.journal import Journal, link_id, now_ms
from bot.notify import Notifier
from bot.risk import Decision, RiskManager
from bot.strategy.base import EntryIntent
from bot.universe import Universe

log = logging.getLogger("bot.exec")

TAKER_MAKER_RT = 0.0010     # paper net-return convention (frozen research label)
MAX_PO_RETRIES = 2          # bounded PostOnly-cancel retries per entry


# ------------------------------------------------------------- reconcile ----
@dataclass(frozen=True)
class ReconcileAction:
    kind: str                  # resolve_order | adopt_order | close_position | adopt_position | fix_qty
    symbol: str
    order_link_id: str = ""
    position_id: int = -1
    detail: str = ""


def plan_reconcile(journal_orders: list[dict], journal_positions: list[dict],
                   exch_orders: list[dict], exch_positions: list[dict]
                   ) -> list[ReconcileAction]:
    """Pure diff of journal state vs exchange state (live mode only).

    journal_orders/positions: open rows as dicts (order_link_id, symbol, qty...).
    exch_orders: [{orderLinkId, symbol, qty}], exch_positions: [{symbol, size}].
    """
    actions: list[ReconcileAction] = []
    ex_links = {o["orderLinkId"] for o in exch_orders if o.get("orderLinkId")}
    j_links = {o["order_link_id"] for o in journal_orders}
    for o in journal_orders:
        if o["order_link_id"] not in ex_links:
            actions.append(ReconcileAction("resolve_order", o["symbol"],
                                           order_link_id=o["order_link_id"],
                                           detail="journal-open order absent on exchange"))
    for o in exch_orders:
        lk = o.get("orderLinkId") or ""
        if lk not in j_links:
            actions.append(ReconcileAction("adopt_order", o["symbol"], order_link_id=lk,
                                           detail="exchange order unknown to journal"))
    ex_size = {p["symbol"]: float(p.get("size") or 0) for p in exch_positions}
    j_by_sym = {p["symbol"]: p for p in journal_positions}
    for p in journal_positions:
        size = ex_size.get(p["symbol"], 0.0)
        if size == 0.0:
            actions.append(ReconcileAction("close_position", p["symbol"],
                                           position_id=p["id"],
                                           detail="journal-open position absent on exchange"))
        elif abs(size - p["qty"]) > 1e-12:
            actions.append(ReconcileAction("fix_qty", p["symbol"], position_id=p["id"],
                                           detail=f"journal qty {p['qty']} != exchange {size}"))
    for sym, size in ex_size.items():
        if size > 0 and sym not in j_by_sym:
            actions.append(ReconcileAction("adopt_position", sym,
                                           detail=f"exchange position size {size} unknown"))
    return actions


# ------------------------------------------------------------ live executor -
class Executor:
    def __init__(self, cfg: BotConfig, rest: BybitRest, journal: Journal,
                 universe: Universe, risk: RiskManager, notify: Notifier):
        self.cfg = cfg
        self.rest = rest
        self.journal = journal
        self.universe = universe
        self.risk = risk
        self.notify = notify
        self._timers: dict[str, asyncio.Task] = {}
        self._po_retries: dict[str, int] = {}   # entry link id -> retries used

    # ------------------------------------------------------------- entry --
    async def _entry_price(self, trigger_close: float, ins) -> float:
        """floor_to_tick(min(trigger close, best bid)): reduces (does not
        eliminate) marketable-PostOnly cancels; entry improves vs the research
        trigger-close assumption only conditional on fill."""
        try:
            bid = await self.rest.best_bid(ins.bybit_symbol)
        except (BybitError, httpx.TransportError) as exc:
            log.warning("best_bid %s failed (%s); using trigger close",
                        ins.bybit_symbol, exc)
            bid = None
        px = min(trigger_close, bid) if bid else trigger_close
        return self._floor_tick(px, ins.tick_size)

    async def submit_entry(self, intent: EntryIntent, dec: Decision, mode: str) -> None:
        try:
            await self._submit_entry(intent, dec, mode)
        finally:
            # journal row (or terminal status) now carries the accounting
            self.risk.release(dec.reserve_key)

    async def _submit_entry(self, intent: EntryIntent, dec: Decision, mode: str) -> None:
        ts = intent.meta.get("signal_ts_ms", now_ms())
        lk = link_id(intent.strategy, intent.symbol, int(ts))
        # deploy-spec: disaster stop is -20% FROM ENTRY — anchored at the
        # actual order price (paper entry == trigger close; live entry may
        # sit at best bid and is re-anchored below once px is known)
        stop_px = intent.limit_price * (1.0 - intent.stop_pct)
        row = dict(order_link_id=lk, strategy=intent.strategy, mode=mode,
                   symbol=intent.symbol, side=intent.side, order_type="Limit",
                   price=intent.limit_price, qty=dec.qty, status="pending",
                   ttl_deadline_ms=now_ms() + intent.ttl_s * 1000,
                   meta={"exit_at_ms": intent.exit_at_ms, "stop_px": stop_px,
                         "stop_pct": intent.stop_pct,
                         "weight": dec.weight, **intent.meta})
        if mode == "paper":
            self.journal.upsert_order(**row)
            self.journal.set_order_status(lk, "open")
            await self.notify.send(f"[paper] entry placed {intent.symbol} "
                                   f"@{intent.limit_price} w={dec.weight:.2f}")
            return
        ins = self.universe.instruments[intent.symbol]
        px = await self._entry_price(intent.limit_price, ins)
        stop_px = px * (1.0 - intent.stop_pct)
        row["price"] = px
        row["meta"]["stop_px"] = stop_px
        self.journal.upsert_order(**row)
        try:
            r = await self.rest.place_order(
                symbol=ins.bybit_symbol, side=intent.side, orderType="Limit",
                qty=self._fmt(dec.qty), price=self._fmt(px),
                timeInForce="PostOnly", orderLinkId=lk,
                stopLoss=self._fmt(self._round_tick(stop_px, ins.tick_size)),
                slOrderType="Market", slTriggerBy="LastPrice")
            self.journal.upsert_order(**{**row, "status": "open",
                                         "exchange_order_id": r.get("orderId")})
            await self.notify.send(f"entry placed {intent.symbol} @{px} "
                                   f"qty={dec.qty} stop={stop_px:.6g} w={dec.weight:.2f}")
            self._timers[lk] = asyncio.create_task(self._ttl_task(lk, intent.ttl_s))
        except BybitError as exc:
            self.journal.set_order_status(lk, "rejected")
            self.journal.event("alarm", "order_reject", f"{intent.symbol}: {exc}")
            await self.notify.alarm(f"order REJECTED {intent.symbol}: {exc}")
        except httpx.TransportError as exc:
            # create ack lost: the order may or may not exist on the exchange —
            # query by orderLinkId before assuming anything; never double-send
            self.journal.event("warn", "order_unknown", f"{intent.symbol}: {exc}")
            await self._adopt_unknown(lk, intent.ttl_s)

    async def _adopt_unknown(self, lk: str, ttl_s: int) -> None:
        """Resolve a lost create ack from the exchange's view of orderLinkId."""
        status, filled, avg = None, 0.0, None
        try:
            hist = await self.rest.order_history(lk)
            if hist:
                h = hist[0]
                status = h.get("orderStatus")
                filled = float(h.get("cumExecQty") or 0)
                avg = float(h.get("avgPrice") or 0) or None
        except (BybitError, httpx.TransportError):
            pass
        if status in ("New", "PartiallyFilled"):
            self.journal.set_order_status(lk, "open", filled, avg)
            self._timers[lk] = asyncio.create_task(self._ttl_task(lk, ttl_s))
        elif status == "Filled":
            self.journal.set_order_status(lk, "filled", filled, avg)
        else:   # Cancelled/Rejected/not found: entry died unfilled
            self.journal.set_order_status(lk, "rejected")
            await self.notify.alarm(f"order create unresolved {lk}: marked "
                                    "rejected (reconcile will adopt strays)")

    @staticmethod
    def _fmt(x: float) -> str:
        return f"{x:.10f}".rstrip("0").rstrip(".")

    @staticmethod
    def _round_tick(px: float, tick: float) -> float:
        if tick <= 0:
            return px
        return round(px / tick) * tick

    @staticmethod
    def _floor_tick(px: float, tick: float) -> float:
        if tick <= 0:
            return px
        return math.floor(px / tick + 1e-9) * tick

    # --------------------------------------------------------------- TTL --
    async def _ttl_task(self, lk: str, ttl_s: int) -> None:
        await asyncio.sleep(ttl_s)
        o = self.journal.get_order(lk)
        if o is None or o["status"] not in ("pending", "open"):
            return
        sym = self.universe.bybit_symbol(o["symbol"])
        try:
            await self.rest.cancel_order(sym, lk)
        except BybitError as exc:
            log.info("TTL cancel %s: %s (likely already filled/cancelled)", lk, exc)
        o = self.journal.get_order(lk)
        if o and o["filled_qty"] > 0:
            self.journal.set_order_status(lk, "partial_ttl")
            await self.notify.send(f"TTL: partial fill kept {o['symbol']} "
                                   f"qty={o['filled_qty']}")
        else:
            self.journal.set_order_status(lk, "cancelled")
            await self.notify.send(f"TTL cancel (no fill) {o['symbol'] if o else lk}")

    # ------------------------------------------------------ private feed --
    async def on_private_message(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        if topic == "order":
            for d in msg.get("data", []):
                await self._on_order(d)
        elif topic == "execution":
            for d in msg.get("data", []):
                await self._on_execution(d)
        elif topic == "wallet":
            for d in msg.get("data", []):
                eq = float(d.get("totalEquity") or 0)
                if eq > 0:
                    self.risk.check_kill_switch(eq, "live")

    async def _on_order(self, d: dict) -> None:
        lk = d.get("orderLinkId") or ""
        o = self.journal.get_order(lk)
        if o is None:
            return
        status = d.get("orderStatus", "")
        filled = float(d.get("cumExecQty") or 0)
        avg = float(d.get("avgPrice") or 0) or None
        if status in ("New", "PartiallyFilled"):
            self.journal.set_order_status(lk, "open", filled, avg)
        elif status == "Filled":
            self.journal.set_order_status(lk, "filled", filled, avg)
            t = self._timers.pop(lk, None)
            if t:
                t.cancel()
        elif status in ("Cancelled", "Rejected", "Deactivated"):
            if (d.get("cancelType") == "CancelByPostOnly" and filled == 0
                    and o["side"] == "Buy" and not lk.endswith("-x")):
                await self._on_postonly_reject(o)
                return
            new = "partial_ttl" if filled > 0 else "cancelled"
            self.journal.set_order_status(lk, new, filled, avg)

    async def _on_postonly_reject(self, o) -> None:
        """Entry PostOnly cancelled as marketable (the ask fell into our price
        while the request was in flight). Bounded retries at the fresh best
        bid, all within the original TTL window; then the entry dies unfilled
        ('rejected' — an unfilled signal for the fill-rate gate)."""
        lk = o["order_link_id"]
        self.journal.event("warn", "postonly_reject", f"{o['symbol']} @{o['price']}")
        tries = self._po_retries.get(lk, 0)
        if tries >= MAX_PO_RETRIES:
            self.journal.set_order_status(lk, "rejected")
            await self.notify.alarm(f"PostOnly rejected {tries + 1}x "
                                    f"{o['symbol']} — giving up")
            return
        if (o["ttl_deadline_ms"] or 0) <= now_ms():
            self.journal.set_order_status(lk, "cancelled")   # TTL window over
            await self.notify.send(f"PostOnly reject past TTL {o['symbol']} — no retry")
            return
        self._po_retries[lk] = tries + 1
        ins = self.universe.instruments[o["symbol"]]
        try:
            bid = await self.rest.best_bid(ins.bybit_symbol)
        except (BybitError, httpx.TransportError):
            bid = None
        if not bid:
            self.journal.set_order_status(lk, "rejected")
            await self.notify.alarm(f"PostOnly reject {o['symbol']}: "
                                    "no bid for retry — giving up")
            return
        px = self._floor_tick(min(o["price"], bid), ins.tick_size)
        meta = json.loads(o["meta"]) if o["meta"] else {}
        stop_pct = meta.get("stop_pct")
        # re-anchor the -20%-from-entry stop at the repriced entry
        stop_px = px * (1.0 - stop_pct) if stop_pct else meta.get("stop_px")
        meta["stop_px"] = stop_px
        try:
            r = await self.rest.place_order(
                symbol=ins.bybit_symbol, side="Buy", orderType="Limit",
                qty=self._fmt(o["qty"]), price=self._fmt(px),
                timeInForce="PostOnly", orderLinkId=lk,
                stopLoss=(self._fmt(self._round_tick(stop_px, ins.tick_size))
                          if stop_px else None),
                slOrderType="Market", slTriggerBy="LastPrice")
            self.journal.upsert_order(
                order_link_id=lk, strategy=o["strategy"], mode=o["mode"],
                symbol=o["symbol"], side="Buy", order_type="Limit", price=px,
                qty=o["qty"], status="open", created_ms=o["created_ms"],
                ttl_deadline_ms=o["ttl_deadline_ms"], meta=meta,
                exchange_order_id=r.get("orderId"))
            await self.notify.send(f"PostOnly retry {o['symbol']} @{px}")
        except BybitError as exc:
            self.journal.set_order_status(lk, "rejected")
            self.journal.event("alarm", "order_reject", f"{o['symbol']}: {exc}")
            await self.notify.alarm(f"PostOnly retry REJECTED {o['symbol']}: {exc}")

    async def _on_execution(self, d: dict) -> None:
        lk = d.get("orderLinkId") or ""
        sym = d.get("symbol", "")
        exec_id = d.get("execId", "")
        px = float(d.get("execPrice") or 0)
        qty = float(d.get("execQty") or 0)
        fee = float(d.get("execFee") or 0)
        ts = int(d.get("execTime") or now_ms())
        if not self.journal.write_execution(exec_id, lk or None, sym, d.get("side", ""),
                                            px, qty, fee, ts):
            return  # duplicate
        o = self.journal.get_order(lk) if lk else None
        if o is not None and not lk.endswith("-x") and o["side"] == "Buy":
            await self._grow_position(o, px, qty, ts)
        elif o is not None and lk.endswith("-x"):
            await self._settle_exit(o, px, ts, reason="exit_24h")
        else:
            # no link: system-generated (stop-loss market) close
            await self._maybe_stop_close(sym, px, ts)

    async def _grow_position(self, o, px: float, qty: float, ts: int) -> None:
        pair = o["symbol"]
        meta = json.loads(o["meta"]) if o["meta"] else {}
        open_pos = [p for p in self.journal.open_positions(mode=o["mode"],
                                                           strategy=o["strategy"])
                    if p["symbol"] == pair]
        if not open_pos:
            pos_id = self.journal.open_position(
                strategy=o["strategy"], mode=o["mode"], symbol=pair, side="Buy",
                qty=qty, entry_px=px, entry_ms=ts, stop_px=meta.get("stop_px"),
                exit_due_ms=meta.get("exit_at_ms"), weight=meta.get("weight"),
                meta={"order_link_id": o["order_link_id"]})
            await self.notify.send(f"FILLED {pair} qty={qty} @{px}")
            delay = max(0.0, (meta.get("exit_at_ms", now_ms()) - now_ms()) / 1000)
            self._timers[f"exit-{pos_id}"] = asyncio.create_task(
                self._exit_task(pos_id, delay))
        else:
            p = open_pos[0]
            new_qty = p["qty"] + qty
            new_px = (p["entry_px"] * p["qty"] + px * qty) / new_qty
            self.journal.update_position_qty(p["id"], new_qty, new_px)

    async def _exit_task(self, pos_id: int, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        rows = [p for p in self.journal.open_positions(mode="live") if p["id"] == pos_id]
        if not rows:
            return
        p = rows[0]
        ins = self.universe.instruments[p["symbol"]]
        lk = (json.loads(p["meta"]).get("order_link_id", f"pos{pos_id}")
              if p["meta"] else f"pos{pos_id}") + "-x"
        self.journal.upsert_order(order_link_id=lk, strategy=p["strategy"], mode="live",
                                  symbol=p["symbol"], side="Sell", order_type="Market",
                                  price=None, qty=p["qty"], status="open",
                                  meta={"position_id": pos_id})
        try:
            await self.rest.place_order(symbol=ins.bybit_symbol, side="Sell",
                                        orderType="Market", qty=self._fmt(p["qty"]),
                                        reduceOnly=True, orderLinkId=lk)
        except BybitError as exc:
            self.journal.event("alarm", "exit_fail", f"{p['symbol']}: {exc}")
            await self.notify.alarm(f"EXIT FAILED {p['symbol']}: {exc}")

    async def _settle_exit(self, o, px: float, ts: int, reason: str) -> None:
        meta = json.loads(o["meta"]) if o["meta"] else {}
        pos_id = meta.get("position_id")
        rows = [p for p in self.journal.open_positions(mode=o["mode"])
                if p["id"] == pos_id]
        if not rows:
            return
        p = rows[0]
        pnl = px / p["entry_px"] - 1.0 - TAKER_MAKER_RT
        self.journal.close_position(p["id"], px, ts, reason, pnl)
        await self.notify.send(f"EXIT {p['symbol']} @{px} net={pnl:+.2%} ({reason})")
        self.risk.check_review_gate(p["strategy"])

    async def _maybe_stop_close(self, bybit_symbol: str, px: float, ts: int) -> None:
        pair = self.universe.pair_for_bybit(bybit_symbol)
        if pair is None:
            return
        for p in self.journal.open_positions(mode="live"):
            if p["symbol"] == pair:
                pnl = px / p["entry_px"] - 1.0 - TAKER_MAKER_RT
                self.journal.close_position(p["id"], px, ts, "stop", pnl)
                t = self._timers.pop(f"exit-{p['id']}", None)
                if t:
                    t.cancel()
                await self.notify.alarm(f"STOP fired {pair} @{px} net={pnl:+.2%}")
                self.risk.check_review_gate(p["strategy"])
                return

    # ---------------------------------------------------------- reconcile --
    async def reconcile(self) -> None:
        """On start / WS reconnect: adopt or repair journal vs exchange.
        New entries stay blocked until this completes (unblock only on
        success; a failed reconcile re-runs on the next reconnect)."""
        if not self.cfg.bybit.api_key:
            log.info("reconcile skipped: no API key (paper-only run)")
            return
        self.risk.block_entries("reconcile after WS (re)connect")
        exch_orders = await self.rest.open_orders()
        exch_positions = [p for p in await self.rest.positions()
                          if float(p.get("size") or 0) > 0]
        j_orders = [dict(o) for o in self.journal.open_orders(mode="live")]
        j_positions = []
        for p in self.journal.open_positions(mode="live"):
            d = dict(p)
            d["symbol"] = self.universe.bybit_symbol(d["symbol"])
            j_positions.append(d)
        actions = plan_reconcile(j_orders, j_positions, exch_orders, exch_positions)
        for a in actions:
            log.warning("reconcile: %s %s %s", a.kind, a.symbol, a.detail)
            self.journal.event("warn", "reconcile", f"{a.kind} {a.symbol}: {a.detail}")
            if a.kind == "resolve_order":
                await self._resolve_order(a.order_link_id)
            elif a.kind == "close_position":
                await self._close_from_history(a.position_id)
            elif a.kind in ("adopt_order", "adopt_position", "fix_qty"):
                await self.notify.alarm(f"reconcile {a.kind}: {a.symbol} {a.detail} "
                                        "— manual check required")
        # re-arm timers for surviving open state
        for o in self.journal.open_orders(mode="live"):
            ttl = max(1.0, ((o["ttl_deadline_ms"] or now_ms()) - now_ms()) / 1000)
            self._timers[o["order_link_id"]] = asyncio.create_task(
                self._ttl_task(o["order_link_id"], int(ttl)))
        for p in self.journal.open_positions(mode="live"):
            delay = max(0.0, ((p["exit_due_ms"] or now_ms()) - now_ms()) / 1000)
            self._timers[f"exit-{p['id']}"] = asyncio.create_task(
                self._exit_task(p["id"], delay))
        self.risk.unblock_entries()

    async def _resolve_order(self, lk: str) -> None:
        try:
            hist = await self.rest.order_history(lk)
        except BybitError:
            hist = []
        if not hist:
            self.journal.set_order_status(lk, "cancelled")
            return
        h = hist[0]
        filled = float(h.get("cumExecQty") or 0)
        avg = float(h.get("avgPrice") or 0) or None
        if h.get("orderStatus") == "Filled":
            self.journal.set_order_status(lk, "filled", filled, avg)
            o = self.journal.get_order(lk)
            if o and avg and not [p for p in self.journal.open_positions(mode="live")
                                  if p["symbol"] == o["symbol"]]:
                await self._grow_position(o, avg, filled, now_ms())
        else:
            self.journal.set_order_status(lk, "partial_ttl" if filled > 0 else "cancelled",
                                          filled, avg)

    async def _close_from_history(self, pos_id: int) -> None:
        rows = [p for p in self.journal.open_positions(mode="live") if p["id"] == pos_id]
        if not rows:
            return
        p = rows[0]
        # position is gone on the exchange: stop fired or manual close while down
        self.journal.close_position(p["id"], p["stop_px"] or p["entry_px"], now_ms(),
                                    "reconciled_gone", None)
        await self.notify.alarm(f"reconcile: {p['symbol']} position closed while bot "
                                "was down (stop or manual) — journal updated, verify PnL")


# ------------------------------------------------------------ paper broker --
class PaperBroker:
    """Mirrors the frozen research execution on hourly bars, zero capital.
    Fill rule: pending limit fills iff the next 1h bar's low < limit (research
    maker semantics). Stop/exit checked bar-by-bar, gap-aware."""

    def __init__(self, cfg: BotConfig, rest: BybitRest, journal: Journal,
                 universe: Universe, notify: Notifier):
        self.cfg = cfg
        self.rest = rest
        self.journal = journal
        self.universe = universe
        self.notify = notify

    async def _last_bar(self, pair: str, bar_open_ms: int) -> dict | None:
        """OHLC of the completed bar [bar_open_ms, +1h) from REST kline."""
        sym = self.universe.bybit_symbol(pair)
        rows = await self.rest.kline(sym, "60", limit=3)
        for r in rows:                          # newest first
            if int(r[0]) == bar_open_ms:
                return {"open": float(r[1]), "high": float(r[2]),
                        "low": float(r[3]), "close": float(r[4])}
        return None

    async def on_bar(self, bar_open_ms: int) -> None:
        bar_close_ms = bar_open_ms + 3_600_000
        # 1) pending paper entries: fill or TTL-cancel on the completed bar
        for o in self.journal.open_orders(mode="paper"):
            if (o["ttl_deadline_ms"] or 0) > bar_close_ms:
                continue                        # TTL window not elapsed yet
            bar = await self._last_bar(o["symbol"], bar_open_ms)
            meta = json.loads(o["meta"]) if o["meta"] else {}
            if bar and bar["low"] < o["price"]:
                self.journal.set_order_status(o["order_link_id"], "filled",
                                              o["qty"], o["price"])
                self.journal.open_position(
                    strategy=o["strategy"], mode="paper", symbol=o["symbol"],
                    side="Buy", qty=o["qty"], entry_px=o["price"], entry_ms=bar_close_ms,
                    stop_px=meta.get("stop_px"), exit_due_ms=meta.get("exit_at_ms"),
                    weight=meta.get("weight"),
                    meta={"order_link_id": o["order_link_id"]})
                await self.notify.send(f"[paper] FILLED {o['symbol']} @{o['price']}")
            else:
                self.journal.set_order_status(o["order_link_id"], "cancelled")
                await self.notify.send(f"[paper] TTL cancel {o['symbol']}")
        # 2) open paper positions: stop / time exit
        for p in self.journal.open_positions(mode="paper"):
            bar = await self._last_bar(p["symbol"], bar_open_ms)
            if bar is None:
                continue
            exit_px, reason = None, ""
            stop = p["stop_px"]
            if stop and bar["open"] <= stop:
                exit_px, reason = bar["open"], "stop"
            elif stop and bar["low"] <= stop:
                exit_px, reason = stop, "stop"
            elif p["exit_due_ms"] and bar_close_ms >= p["exit_due_ms"]:
                exit_px, reason = bar["close"], "exit_24h"
            if exit_px is None:
                continue
            ret = exit_px / p["entry_px"] - 1.0 - TAKER_MAKER_RT
            self.journal.close_position(p["id"], exit_px, bar_close_ms, reason, ret)
            slots = self.cfg.risk.slots
            eq = self.journal.last_equity("paper") or self.cfg.paper_equity_usd
            eq *= 1.0 + (p["weight"] or 1.0) * ret / slots
            self.journal.write_equity(bar_close_ms, eq, "paper")
            await self.notify.send(f"[paper] EXIT {p['symbol']} @{exit_px:.6g} "
                                   f"net={ret:+.2%} ({reason}) eq=${eq:,.0f}")
