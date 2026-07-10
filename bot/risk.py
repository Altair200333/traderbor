"""Risk manager — the only component allowed to turn intents into orders.

Gates (deploy-spec section 3, DRAFT until frozen with Mike):
  - 15 slots x equity/15 x weight (weight cap 1.0 default), slot busy until exit
  - portfolio gross cap: abs(open) + abs(pending) notional across ALL
    strategies + in-flight reservations <= gross_cap_mult * (1 -
    gross_safety_buffer) * equity (downsize to headroom; veto if too small);
    admission control only — a breach blocks new entries, never force-closes
  - per-symbol 24h cooldown (double-check; strategy also enforces)
  - account kill-switch at -10% cumulative drawdown -> halt, manual restart
  - fill-rate gate: < 85% fills over the last 30 live signals -> pause
  - 50-trade review gate: running mean net < 0 -> pause + report
  - per-trade notional cap and Bybit $5 minimum notional
Pure math is in module functions for unit tests; RiskManager wires the journal.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from bot.config import RiskConfig
from bot.journal import Journal, link_id, now_ms
from bot.strategy.base import EntryIntent

log = logging.getLogger("bot.risk")

HALT_KEY = "risk_halt"          # kv flag: set by kill-switch, cleared manually
PAUSE_KEY = "risk_pause"        # kv flag: set by fill-rate / review gates


@dataclass(frozen=True)
class Decision:
    approved: bool
    reason: str = ""
    qty: float = 0.0
    notional: float = 0.0
    weight: float = 0.0
    reserve_key: str = ""       # headroom reservation; executor releases it


def round_qty(qty: float, qty_step: float) -> float:
    """Round DOWN to the instrument quantity step."""
    if qty_step <= 0:
        return qty
    return math.floor(qty / qty_step + 1e-9) * qty_step


def size_position(equity: float, slots: int, weight: float, max_weight: float,
                  price: float, qty_step: float, min_qty: float,
                  max_notional: float, min_notional: float) -> tuple[float, float, str]:
    """Return (qty, notional, veto_reason). veto_reason == '' means ok."""
    w = min(weight, max_weight)
    notional = min(equity / slots * w, max_notional)
    if price <= 0 or equity <= 0:
        return 0.0, 0.0, "bad price/equity"
    qty = round_qty(notional / price, qty_step)
    if qty < min_qty or qty <= 0:
        return 0.0, 0.0, f"qty {qty} below min {min_qty}"
    actual = qty * price
    if actual < min_notional:
        return 0.0, 0.0, f"notional {actual:.2f} below min {min_notional}"
    return qty, actual, ""


def committed_notional(open_positions: list, pending_orders: list) -> float:
    """Total committed gross by ABSOLUTE value (a hedge does not reduce gross):
    abs(open position notional) + abs(remaining unfilled limit notional)
    (qty - filled_qty = leavesQty, so partial fills are not double-counted).
    Reduce-only market exits (price None) are not new exposure."""
    total = sum(abs(p["qty"] * p["entry_px"]) for p in open_positions)
    for o in pending_orders:
        if o["price"] is None:
            continue
        total += abs(max(0.0, o["qty"] - (o["filled_qty"] or 0.0)) * o["price"])
    return total


def apply_gross_cap(qty: float, notional: float, headroom: float, price: float,
                    qty_step: float, min_qty: float, min_notional: float
                    ) -> tuple[float, float, str]:
    """Downsize a sized trade to the remaining gross headroom (audit defect 1).
    Veto if the downsized trade falls below min notional or below 25% of the
    originally-sized notional. Returns (qty, notional, veto_reason)."""
    if notional <= headroom:
        return qty, notional, ""
    if headroom < min_notional or headroom < 0.25 * notional:
        return 0.0, 0.0, (f"gross cap: headroom {headroom:.2f} < "
                          f"min({min_notional:.0f}, 25% of sized {notional:.2f})")
    new_qty = round_qty(headroom / price, qty_step)
    actual = new_qty * price
    if new_qty < min_qty or actual < min_notional or actual < 0.25 * notional:
        return 0.0, 0.0, (f"gross cap: downsized notional {actual:.2f} below "
                          f"min({min_notional:.0f}, 25% of sized {notional:.2f})")
    return new_qty, actual, ""


def drawdown(peak: float, current: float) -> float:
    if peak <= 0:
        return 0.0
    return max(0.0, 1.0 - current / peak)


class RiskManager:
    def __init__(self, cfg: RiskConfig, journal: Journal):
        self.cfg = cfg
        self.journal = journal
        self._reserved: dict[str, float] = {}   # in-flight headroom reservations
        self._block_reason: str | None = None   # e.g. reconcile after WS reconnect

    # ------------------------------------------------------ reservations --
    def release(self, key: str) -> None:
        """Release a headroom reservation (once the order row is journaled,
        or on reject/cancel before it was)."""
        self._reserved.pop(key, None)

    def block_entries(self, reason: str) -> None:
        self._block_reason = reason

    def unblock_entries(self) -> None:
        self._block_reason = None

    # ------------------------------------------------------------- gates --
    def halted(self) -> bool:
        return self.journal.kv_get(HALT_KEY) is not None

    def paused(self) -> bool:
        return self.journal.kv_get(PAUSE_KEY) is not None

    def check_kill_switch(self, equity: float, source: str) -> bool:
        """Record equity; trip the kill-switch on cumulative DD. Returns True
        if halted (already or newly)."""
        self.journal.write_equity(now_ms(), equity, source)
        if self.halted():
            return True
        # monotonic peak in kv (same-ms equity rows overwrite; kv never regresses)
        peak_key = f"equity_peak_{source}"
        peak = max(float(self.journal.kv_get(peak_key, "0") or 0), equity)
        self.journal.kv_set(peak_key, str(peak))
        dd = drawdown(peak, equity)
        if dd >= self.cfg.kill_dd:
            self.journal.kv_set(HALT_KEY, f"kill-switch dd={dd:.4f} peak={peak:.2f}")
            self.journal.event("alarm", "kill_switch",
                               f"account DD {dd:.1%} >= {self.cfg.kill_dd:.0%}: HALT "
                               "(manual restart required)")
            log.error("KILL SWITCH: dd=%.4f", dd)
            return True
        return False

    def check_fill_rate_gate(self, strategy: str) -> None:
        """< 85% fills over the last 30 live signals -> pause (spec s3)."""
        if self.paused():
            return
        sigs = self.journal.recent_signals(strategy, "live", self.cfg.fill_rate_window)
        if len(sigs) < self.cfg.fill_rate_window:
            return
        resolved, filled = 0, 0
        for s in sigs:
            o = self.journal.get_order(link_id(strategy, s["symbol"], s["ts_ms"]))
            if o is not None and o["status"] in ("pending", "open"):
                continue                # in flight — outcome not knowable yet
            # every signal is an observation: no order at all, cancelled (TTL)
            # and rejected (PostOnly death) all count as unfilled misses
            resolved += 1
            if o is not None and o["status"] in ("filled", "partial_ttl"):
                filled += 1
        if resolved >= self.cfg.fill_rate_window and \
                filled / resolved < self.cfg.fill_rate_min:
            self.journal.kv_set(PAUSE_KEY, f"fill-rate {filled}/{resolved}")
            self.journal.event("alarm", "fill_rate_gate",
                               f"fill rate {filled}/{resolved} < "
                               f"{self.cfg.fill_rate_min:.0%}: PAUSED")

    def check_review_gate(self, strategy: str) -> None:
        """50 filled live trades with mean net < 0 -> pause + report (spec s3)."""
        if self.paused():
            return
        trades = self.journal.closed_trades(strategy, "live", self.cfg.review_trades)
        if len(trades) < self.cfg.review_trades:
            return
        rets = [t["pnl_net"] for t in trades if t["pnl_net"] is not None]
        if rets and sum(rets) / len(rets) < 0:
            self.journal.kv_set(PAUSE_KEY, f"review mean={sum(rets)/len(rets):.4f}")
            self.journal.event("alarm", "review_gate",
                               f"{len(rets)}-trade running mean net < 0: PAUSED")

    # ---------------------------------------------------------- evaluate --
    def evaluate(self, intent: EntryIntent, equity: float, mode: str,
                 qty_step: float, min_qty: float) -> Decision:
        if self.halted():
            return Decision(False, "halted (kill-switch)")
        if self._block_reason:
            return Decision(False, f"entries blocked ({self._block_reason})")
        if mode == "live" and self.paused():
            return Decision(False, f"paused ({self.journal.kv_get(PAUSE_KEY)})")
        open_all = self.journal.open_positions(mode=mode)
        pending_all = self.journal.open_orders(mode=mode)
        open_pos = [p for p in open_all if p["strategy"] == intent.strategy]
        pending = [o for o in pending_all if o["strategy"] == intent.strategy]
        if len(open_pos) + len(pending) >= self.cfg.slots:
            return Decision(False, f"no free slot ({len(open_pos)} open, "
                                   f"{len(pending)} pending)")
        for p in open_pos:
            if p["symbol"] == intent.symbol:
                return Decision(False, "symbol already held")
        qty, notional, veto = size_position(
            equity, self.cfg.slots, intent.weight, self.cfg.max_weight,
            intent.limit_price, qty_step, min_qty,
            self.cfg.max_trade_notional_usd, self.cfg.min_notional_usd)
        if veto:
            return Decision(False, veto)
        # portfolio gross cap across ALL strategies (audit defect 1): journal
        # state + in-flight reservations, vs buffered equity cap
        cap = (self.cfg.gross_cap_mult
               * (1.0 - self.cfg.gross_safety_buffer) * equity)
        committed = (committed_notional(open_all, pending_all)
                     + sum(self._reserved.values()))
        if committed >= cap:
            # admission control only: on a breach (e.g. equity fell under the
            # existing gross) block new entries + alarm — never force-close
            self.journal.event("alarm", "gross_cap",
                               f"gross {committed:.2f} >= cap {cap:.2f}: "
                               "blocking new entries")
            return Decision(False, f"gross cap: committed {committed:.2f} "
                                   f">= cap {cap:.2f}")
        qty, notional, veto = apply_gross_cap(
            qty, notional, cap - committed, intent.limit_price, qty_step,
            min_qty, self.cfg.min_notional_usd)
        if veto:
            return Decision(False, veto)
        # reserve the headroom ATOMICALLY (before the order is sent): no two
        # concurrent intents may both pass on the same headroom
        key = link_id(intent.strategy, intent.symbol,
                      int(intent.meta.get("signal_ts_ms", now_ms())))
        self._reserved[key] = notional
        return Decision(True, "", qty=qty, notional=notional,
                        weight=min(intent.weight, self.cfg.max_weight),
                        reserve_key=key)
