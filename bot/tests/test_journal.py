"""Journal storage semantics used by the detector and gates."""
from __future__ import annotations

from bot.journal import link_id


def test_link_id_format_and_length():
    lk = link_id("liqrev_v2", "PEPEUSDT", 1751234400000)
    assert lk == "liqrev_v2-PEPEUSDT-1751234400000"
    assert len(lk) <= 45
    assert all(c.isalnum() or c in "-_" for c in lk)


def test_snapshot_series_ascending(journal):
    for i, px in enumerate([3.0, 1.0, 2.0]):
        journal.write_snapshots(1000 + i * 3_600_000,
                                [{"symbol": "ADAUSDT", "close": px, "oi": 10.0 * i}])
    s = journal.get_series("ADAUSDT", 10)
    assert [r["bar_ms"] for r in s] == [1000, 3601000, 7201000]
    assert [r["close"] for r in s] == [3.0, 1.0, 2.0]
    s2 = journal.get_series("ADAUSDT", 2)
    assert [r["close"] for r in s2] == [1.0, 2.0]


def test_snapshot_upsert_idempotent(journal):
    journal.write_snapshots(1000, [{"symbol": "ADAUSDT", "close": 1.0}])
    journal.write_snapshots(1000, [{"symbol": "ADAUSDT", "close": 1.5}])
    s = journal.get_series("ADAUSDT", 10)
    assert len(s) == 1 and s[0]["close"] == 1.5


DAY = 86_400_000


def test_median_daily_volume_needs_30_completed_days(journal):
    journal.write_daily_volume("ADAUSDT", [(i * DAY, 2e6) for i in range(29)])
    assert journal.median_daily_volume("ADAUSDT", 29 * DAY) is None
    journal.write_daily_volume("ADAUSDT", [(29 * DAY, 2e6)])
    assert journal.median_daily_volume("ADAUSDT", 30 * DAY) == 2e6


def test_median_daily_volume_excludes_today(journal):
    rows = [(i * DAY, 1e6) for i in range(30)] + [(30 * DAY, 9e9)]
    journal.write_daily_volume("ADAUSDT", rows)
    # day 30 is "today" (not completed) when now is inside day 30
    assert journal.median_daily_volume("ADAUSDT", 30 * DAY) == 1e6


def test_median_is_true_median(journal):
    vols = [float(v) for v in range(1, 31)]          # 1..30
    journal.write_daily_volume("ADAUSDT", [(i * DAY, vols[i]) for i in range(30)])
    assert journal.median_daily_volume("ADAUSDT", 30 * DAY) == 15.5


def test_signal_dedupe_and_cooldown_lookup(journal):
    journal.write_signal(strategy="liqrev_v2", mode="paper", symbol="ADAUSDT",
                         ts_ms=5000, approved=True)
    journal.write_signal(strategy="liqrev_v2", mode="paper", symbol="ADAUSDT",
                         ts_ms=5000, approved=True)      # duplicate ignored
    assert journal.last_signal_ms("liqrev_v2", "ADAUSDT") == 5000
    assert journal.last_signal_ms("liqrev_v2", "OTHERUSDT") is None


def test_signal_result_update(journal):
    journal.write_signal(strategy="liqrev_v2", mode="live", symbol="ADAUSDT",
                         ts_ms=5000, approved=False)
    journal.set_signal_result("liqrev_v2", "ADAUSDT", 5000, True, None)
    row = journal.db.execute("SELECT approved, veto_reason FROM signals").fetchone()
    assert row["approved"] == 1 and row["veto_reason"] is None
    journal.set_signal_result("liqrev_v2", "ADAUSDT", 5000, False, "no free slot")
    row = journal.db.execute("SELECT approved, veto_reason FROM signals").fetchone()
    assert row["approved"] == 0 and row["veto_reason"] == "no free slot"


def test_equity_peak_tracking(journal):
    journal.write_equity(1, 5000, "live")
    journal.write_equity(2, 5500, "live")
    journal.write_equity(3, 5200, "live")
    assert journal.equity_peak("live") == 5500
    assert journal.last_equity("live") == 5200


def test_execution_dedupe(journal):
    assert journal.write_execution("e1", "lk", "ADAUSDT", "Buy", 1.0, 2.0, 0.0, 10)
    assert not journal.write_execution("e1", "lk", "ADAUSDT", "Buy", 1.0, 2.0, 0.0, 10)
