"""Journal-vs-exchange reconcile planning (pure logic)."""
from __future__ import annotations

from bot.execution import plan_reconcile


def j_order(link: str, symbol: str = "ADAUSDT", qty: float = 10.0) -> dict:
    return {"order_link_id": link, "symbol": symbol, "qty": qty}


def j_pos(pid: int, symbol: str = "ADAUSDT", qty: float = 10.0) -> dict:
    return {"id": pid, "symbol": symbol, "qty": qty}


def test_clean_state_no_actions():
    a = plan_reconcile(
        [j_order("lk1")], [j_pos(1, "SOLUSDT", 5.0)],
        [{"orderLinkId": "lk1", "symbol": "ADAUSDT", "qty": "10"}],
        [{"symbol": "SOLUSDT", "size": "5.0"}])
    assert a == []


def test_journal_order_gone_on_exchange():
    a = plan_reconcile([j_order("lk1")], [], [], [])
    assert len(a) == 1 and a[0].kind == "resolve_order" and a[0].order_link_id == "lk1"


def test_unknown_exchange_order_adopted():
    a = plan_reconcile([], [], [{"orderLinkId": "alien", "symbol": "XRPUSDT"}], [])
    assert len(a) == 1 and a[0].kind == "adopt_order"


def test_journal_position_gone_stop_fired():
    a = plan_reconcile([], [j_pos(7, "ADAUSDT")], [], [])
    assert len(a) == 1
    assert a[0].kind == "close_position" and a[0].position_id == 7


def test_unknown_exchange_position_adopted():
    a = plan_reconcile([], [], [], [{"symbol": "DOGEUSDT", "size": "100"}])
    assert len(a) == 1 and a[0].kind == "adopt_position"


def test_qty_mismatch_flagged():
    a = plan_reconcile([], [j_pos(3, "ADAUSDT", 10.0)], [],
                       [{"symbol": "ADAUSDT", "size": "6.0"}])
    assert len(a) == 1 and a[0].kind == "fix_qty"


def test_zero_size_exchange_position_is_flat():
    a = plan_reconcile([], [], [], [{"symbol": "ADAUSDT", "size": "0"}])
    assert a == []


def test_combined_crash_recovery_scene():
    """kill -9 with one open order, one live position; exchange shows the
    order filled->gone and the position still there but partially closed."""
    a = plan_reconcile(
        [j_order("lk-entry", "ADAUSDT")],
        [j_pos(1, "SOLUSDT", 10.0)],
        [],
        [{"symbol": "SOLUSDT", "size": "4.0"}])
    kinds = sorted(x.kind for x in a)
    assert kinds == ["fix_qty", "resolve_order"]
