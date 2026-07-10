"""Risk gating: slots, kill-switch, notional caps, gross cap, fill-rate gate."""
from __future__ import annotations

import pytest

from bot.config import RiskConfig, load_config
from bot.journal import link_id, now_ms
from bot.risk import (Decision, RiskManager, apply_gross_cap,
                      committed_notional, drawdown, round_qty, size_position)
from bot.strategy.base import EntryIntent


def intent(symbol: str = "ADAUSDT", price: float = 0.5, weight: float = 1.0,
           strategy: str = "liqrev_v2") -> EntryIntent:
    return EntryIntent(strategy=strategy, symbol=symbol, side="Buy",
                       limit_price=price, weight=weight, ttl_s=3600,
                       stop_pct=0.20, exit_at_ms=now_ms() + 86_400_000,
                       meta={"signal_ts_ms": now_ms()})


def test_round_qty_down_to_step():
    assert round_qty(123.456, 1.0) == 123.0
    assert round_qty(123.456, 0.1) == pytest.approx(123.4)
    assert round_qty(0.999, 0.01) == pytest.approx(0.99)
    assert round_qty(5.0, 0.0) == 5.0


def test_size_position_slot_math():
    qty, notional, veto = size_position(
        equity=5000, slots=15, weight=1.0, max_weight=2.0, price=0.5,
        qty_step=1.0, min_qty=1.0, max_notional=1000, min_notional=5)
    assert veto == ""
    assert notional == pytest.approx(qty * 0.5)
    assert qty == 666.0                      # floor(5000/15/0.5)


def test_size_position_weight_and_cap():
    qty2, _, _ = size_position(5000, 15, 2.0, 2.0, 0.5, 1.0, 1.0, 1000, 5)
    qty3, _, _ = size_position(5000, 15, 3.0, 2.0, 0.5, 1.0, 1.0, 1000, 5)
    assert qty2 == qty3                      # weight capped at 2.0
    _, notional, _ = size_position(100000, 15, 2.0, 2.0, 0.5, 1.0, 1.0, 1000, 5)
    assert notional <= 1000                  # per-trade notional cap


def test_size_position_min_notional_veto():
    _, _, veto = size_position(50, 15, 0.3, 2.0, 0.5, 1.0, 1.0, 1000, 5)
    assert "below min" in veto


def test_drawdown():
    assert drawdown(100, 90) == pytest.approx(0.10)
    assert drawdown(100, 110) == 0.0
    assert drawdown(0, 50) == 0.0


def test_kill_switch_trips_and_blocks(journal):
    rm = RiskManager(RiskConfig(), journal)
    assert not rm.check_kill_switch(5000, "live")
    assert not rm.check_kill_switch(4600, "live")    # -8%: fine
    assert rm.check_kill_switch(4400, "live")        # -12%: HALT
    assert rm.halted()
    dec = rm.evaluate(intent(), 4400, "live", 1.0, 1.0)
    assert not dec.approved and "halted" in dec.reason
    # manual restart only: another good equity print does NOT clear it
    assert rm.check_kill_switch(5100, "live")


def test_slots_full_veto(journal):
    rm = RiskManager(RiskConfig(slots=2), journal)
    for sym in ("AAAUSDT", "BBBUSDT"):
        journal.open_position(strategy="liqrev_v2", mode="paper", symbol=sym,
                              side="Buy", qty=1, entry_px=1.0, entry_ms=now_ms())
    dec = rm.evaluate(intent("CCCUSDT", price=1.0), 5000, "paper", 0.1, 0.1)
    assert not dec.approved and "no free slot" in dec.reason


def test_pending_orders_occupy_slots(journal):
    rm = RiskManager(RiskConfig(slots=1), journal)
    journal.upsert_order(order_link_id="liqrev_v2-AAAUSDT-1", strategy="liqrev_v2",
                         mode="paper", symbol="AAAUSDT", side="Buy", qty=1,
                         status="open", price=1.0)
    dec = rm.evaluate(intent("BBBUSDT", price=1.0), 5000, "paper", 0.1, 0.1)
    assert not dec.approved


def test_duplicate_symbol_veto(journal):
    rm = RiskManager(RiskConfig(), journal)
    journal.open_position(strategy="liqrev_v2", mode="paper", symbol="ADAUSDT",
                          side="Buy", qty=1, entry_px=1.0, entry_ms=now_ms())
    dec = rm.evaluate(intent("ADAUSDT"), 5000, "paper", 0.1, 0.1)
    assert not dec.approved and "already held" in dec.reason


def test_approval_happy_path(journal):
    # uncapped-overlay variant needs explicit max_weight (default is 1.0)
    rm = RiskManager(RiskConfig(max_weight=2.0), journal)
    dec = rm.evaluate(intent(weight=1.5), 5000, "paper", 1.0, 1.0)
    assert dec.approved
    assert dec.qty == 1000.0                 # floor(5000/15*1.5/0.5)
    assert dec.weight == 1.5


def test_max_weight_default_clips_to_one(journal):
    assert RiskConfig().max_weight == 1.0    # audit 2026-07-10 safety default
    assert RiskConfig().gross_cap_mult == 1.0
    assert RiskConfig().gross_safety_buffer == 0.05
    rm = RiskManager(RiskConfig(), journal)
    dec = rm.evaluate(intent(weight=2.0), 5000, "paper", 1.0, 1.0)
    assert dec.approved
    assert dec.weight == 1.0                 # overlay weight clipped
    assert dec.qty == 666.0                  # floor(5000/15*1.0/0.5)


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("BOT_MAX_WEIGHT", "2.0")
    monkeypatch.setenv("BOT_GROSS_CAP_MULT", "1.5")
    monkeypatch.setenv("BOT_GROSS_SAFETY_BUFFER", "0.1")
    cfg = load_config()
    assert cfg.risk.max_weight == 2.0
    assert cfg.risk.gross_cap_mult == 1.5
    assert cfg.risk.gross_safety_buffer == 0.1


# ------------------------------------------------------------- gross cap ----
def test_committed_notional_math():
    pos = [{"qty": 100.0, "entry_px": 2.0}]                                # 200
    orders = [
        {"side": "Buy", "price": 1.0, "qty": 50.0, "filled_qty": 10.0},    # 40 unfilled (leaves)
        {"side": "Sell", "price": 1.0, "qty": 30.0, "filled_qty": 0.0},    # short limit: abs 30
        {"side": "Sell", "price": None, "qty": 100.0, "filled_qty": 0.0},  # reduce-only exit: ignored
    ]
    assert committed_notional(pos, orders) == pytest.approx(270.0)
    assert committed_notional([], []) == 0.0


def test_apply_gross_cap_paths():
    # fits within headroom: untouched
    assert apply_gross_cap(100.0, 50.0, 60.0, 0.5, 1.0, 1.0, 5.0) == (100.0, 50.0, "")
    # downsized to headroom
    qty, notional, veto = apply_gross_cap(666.0, 333.0, 200.0, 0.5, 1.0, 1.0, 5.0)
    assert veto == ""
    assert qty == 400.0 and notional == pytest.approx(200.0)
    # veto: headroom below $5 min notional
    _, _, veto = apply_gross_cap(666.0, 333.0, 3.0, 0.5, 1.0, 1.0, 5.0)
    assert "gross cap" in veto
    # veto: headroom below 25% of the originally-sized notional
    _, _, veto = apply_gross_cap(666.0, 333.0, 50.0, 0.5, 1.0, 1.0, 5.0)
    assert "gross cap" in veto
    # veto: already over the cap (negative headroom)
    _, _, veto = apply_gross_cap(666.0, 333.0, -10.0, 0.5, 1.0, 1.0, 5.0)
    assert "gross cap" in veto


def test_gross_cap_across_strategies(journal):
    """Committed = open positions + pending entries across ALL strategies;
    cap = 1.0 * (1 - 0.05) * 5000 = 4750."""
    rm = RiskManager(RiskConfig(), journal)
    journal.open_position(strategy="unlock_s2", mode="paper", symbol="XUSDT",
                          side="Buy", qty=2300, entry_px=1.0, entry_ms=now_ms())
    journal.upsert_order(order_link_id="liqrev_v2-YUSDT-1", strategy="liqrev_v2",
                         mode="paper", symbol="YUSDT", side="Buy", qty=2250,
                         price=1.0, status="open")
    # committed 4550 of cap 4750 -> headroom 200; sized 333 -> downsized to 200
    dec = rm.evaluate(intent("CCCUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert dec.approved
    assert dec.qty == 400.0
    assert dec.notional == pytest.approx(200.0)


def test_gross_cap_veto_below_min_notional(journal):
    rm = RiskManager(RiskConfig(), journal)
    journal.open_position(strategy="unlock_s2", mode="paper", symbol="XUSDT",
                          side="Buy", qty=4747, entry_px=1.0, entry_ms=now_ms())
    dec = rm.evaluate(intent("CCCUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert not dec.approved and "gross cap" in dec.reason  # headroom 3 < $5


def test_gross_cap_veto_below_quarter_of_sized(journal):
    rm = RiskManager(RiskConfig(), journal)
    journal.open_position(strategy="unlock_s2", mode="paper", symbol="XUSDT",
                          side="Buy", qty=4700, entry_px=1.0, entry_ms=now_ms())
    # headroom 50 >= $5 but < 25% of sized 333 -> veto, no dust entries
    dec = rm.evaluate(intent("CCCUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert not dec.approved and "gross cap" in dec.reason


def test_gross_cap_breach_blocks_and_alarms(journal):
    """Equity fell under existing gross: admission control only — new entries
    blocked + alarm, nothing force-closed."""
    rm = RiskManager(RiskConfig(), journal)
    journal.open_position(strategy="unlock_s2", mode="paper", symbol="XUSDT",
                          side="Buy", qty=4800, entry_px=1.0, entry_ms=now_ms())
    dec = rm.evaluate(intent("CCCUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert not dec.approved and "gross cap" in dec.reason  # 4800 >= cap 4750
    row = journal.db.execute(
        "SELECT COUNT(*) AS c FROM events WHERE kind='gross_cap'").fetchone()
    assert row["c"] == 1
    assert len(journal.open_positions(mode="paper")) == 1  # nothing closed


def test_reservation_blocks_concurrent_headroom(journal):
    """Atomic reservation: a second intent evaluated before the first order is
    journaled must NOT pass on the same headroom."""
    rm = RiskManager(RiskConfig(), journal)
    journal.open_position(strategy="unlock_s2", mode="paper", symbol="XUSDT",
                          side="Buy", qty=4300, entry_px=1.0, entry_ms=now_ms())
    # cap 4750, committed 4300 -> headroom 450
    d1 = rm.evaluate(intent("AAAUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert d1.approved and d1.notional == pytest.approx(333.0)
    assert d1.reserve_key
    # reservation shrinks the visible headroom to 117 -> downsized
    d2 = rm.evaluate(intent("BBBUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert d2.approved and d2.notional == pytest.approx(117.0)
    # third intent: committed 4750 >= cap -> blocked
    d3 = rm.evaluate(intent("DDDUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert not d3.approved and "gross cap" in d3.reason
    # release (reject/cancel path) restores the headroom
    rm.release(d2.reserve_key)
    d4 = rm.evaluate(intent("EEEUSDT", price=0.5), 5000, "paper", 1.0, 1.0)
    assert d4.approved and d4.notional == pytest.approx(117.0)


def test_block_entries_gate(journal):
    """New entries blocked while reconcile-after-reconnect is in progress."""
    rm = RiskManager(RiskConfig(), journal)
    rm.block_entries("reconcile after WS (re)connect")
    dec = rm.evaluate(intent(), 5000, "live", 1.0, 1.0)
    assert not dec.approved and "blocked" in dec.reason
    rm.unblock_entries()
    assert rm.evaluate(intent(), 5000, "live", 1.0, 1.0).approved


def test_fill_rate_gate_pauses(journal):
    cfg = RiskConfig(fill_rate_window=10, fill_rate_min=0.85)
    rm = RiskManager(cfg, journal)
    for i in range(10):
        ts = 1000 + i
        journal.write_signal(strategy="liqrev_v2", mode="live", symbol=f"S{i}USDT",
                             ts_ms=ts, approved=True)
        journal.upsert_order(order_link_id=link_id("liqrev_v2", f"S{i}USDT", ts),
                             strategy="liqrev_v2", mode="live", symbol=f"S{i}USDT",
                             side="Buy", qty=1, price=1.0,
                             status="filled" if i < 8 else "cancelled")
    rm.check_fill_rate_gate("liqrev_v2")     # 8/10 = 80% < 85%
    assert rm.paused()
    dec = rm.evaluate(intent(), 5000, "live", 0.1, 0.1)
    assert not dec.approved and "paused" in dec.reason
    # paper entries are NOT blocked by the live pause
    dec_paper = rm.evaluate(intent(), 5000, "paper", 0.1, 0.1)
    assert isinstance(dec_paper, Decision)


def test_fill_rate_gate_counts_postonly_rejects(journal):
    """An entry that died by PostOnly reject is an unfilled signal (defect 2)."""
    cfg = RiskConfig(fill_rate_window=10, fill_rate_min=0.85)
    rm = RiskManager(cfg, journal)
    for i in range(10):
        ts = 1000 + i
        journal.write_signal(strategy="liqrev_v2", mode="live", symbol=f"S{i}USDT",
                             ts_ms=ts, approved=True)
        journal.upsert_order(order_link_id=link_id("liqrev_v2", f"S{i}USDT", ts),
                             strategy="liqrev_v2", mode="live", symbol=f"S{i}USDT",
                             side="Buy", qty=1, price=1.0,
                             status="filled" if i < 8 else "rejected")
    rm.check_fill_rate_gate("liqrev_v2")     # 8 filled + 2 rejects = 80% < 85%
    assert rm.paused()


def test_review_gate_pauses_on_negative_mean(journal):
    cfg = RiskConfig(review_trades=5)
    rm = RiskManager(cfg, journal)
    for i in range(5):
        pid = journal.open_position(strategy="liqrev_v2", mode="live",
                                    symbol=f"S{i}USDT", side="Buy", qty=1,
                                    entry_px=1.0, entry_ms=1000 + i)
        journal.close_position(pid, 0.99, 2000 + i, "exit_24h", -0.011)
    rm.check_review_gate("liqrev_v2")
    assert rm.paused()
