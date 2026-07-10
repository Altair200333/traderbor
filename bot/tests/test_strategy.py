"""LiqrevStrategy live wrapper + unlock cliff extraction."""
from __future__ import annotations

import pytest

from bot.config import LiqrevConfig
from bot.strategy.base import EntryIntent, SymbolSeries
from bot.strategy.liqrev_v2 import H_MS, LiqrevStrategy, Overlay
from bot.strategy.unlock_watch import extract_cliffs

BAR0 = 1_767_225_600_000  # 2026-01-01 00:00 UTC


class FakeMarket:
    """Crash on the last bar for ADAUSDT; BTC flat."""

    def __init__(self, n: int = 10, bar_open_ms: int | None = None):
        self.n = n
        self.bar_open_ms = bar_open_ms or (BAR0 + (n - 1) * H_MS)
        self.bar_close_ms = self.bar_open_ms + H_MS
        self.equity_usd = 5000.0

    def symbols(self):
        return ["ADAUSDT", "BTCUSDT"]

    def series(self, symbol: str, n_bars: int):
        bars = [BAR0 + i * H_MS for i in range(self.n)][-n_bars:]
        n = len(bars)
        if symbol == "ADAUSDT":
            close = [1.0] * (n - 1) + [0.88]          # -12% vs 6 bars ago
            oi = [1000.0] * (n - 1) + [850.0]         # -15%
        else:
            close = [50_000.0] * n
            oi = [1.0] * n
        return SymbolSeries(bar_ms=bars, close=close, oi=oi)

    def liquidity_ok(self, symbol: str) -> bool:
        return True


def make_strategy(journal) -> LiqrevStrategy:
    return LiqrevStrategy(LiqrevConfig(mode="paper"), journal,
                          overlay=Overlay([-1.0, 0.0, 1.0]))


def test_signal_emitted_with_frozen_execution_params(journal):
    strat = make_strategy(journal)
    mkt = FakeMarket()
    intents = strat.on_bar(mkt)
    assert len(intents) == 1
    it = intents[0]
    assert isinstance(it, EntryIntent)
    assert it.symbol == "ADAUSDT"
    assert it.side == "Buy"
    assert it.limit_price == pytest.approx(0.88)      # trigger-bar close
    assert it.ttl_s == 3600
    assert it.stop_pct == 0.20
    assert it.exit_at_ms == mkt.bar_close_ms + 24 * H_MS
    # BTC flat -> score 0 -> P = 2/3 with the toy distribution -> w = 4/3
    assert it.weight == pytest.approx(4.0 / 3.0)
    assert journal.last_signal_ms("liqrev_v2", "ADAUSDT") == mkt.bar_close_ms


def test_journal_cooldown_blocks_repeat(journal):
    strat = make_strategy(journal)
    assert len(strat.on_bar(FakeMarket())) == 1
    # 3 hours later, still crashed -> raw trigger, but journal cooldown holds
    later = FakeMarket(n=13)
    assert strat.on_bar(later) == []


def test_mode_off_is_silent(journal):
    strat = LiqrevStrategy(LiqrevConfig(mode="off"), journal,
                           overlay=Overlay([0.0]))
    assert strat.on_bar(FakeMarket()) == []


def test_illiquid_symbol_skipped(journal):
    strat = make_strategy(journal)

    class M(FakeMarket):
        def liquidity_ok(self, symbol):
            return False

    assert strat.on_bar(M()) == []


# ------------------------------------------------------------- unlocks ------
def test_extract_cliffs_threshold_and_denominator():
    payload = {
        "supplyMetrics": {"maxSupply": 1_000_000.0},
        "metadata": {"unlockEvents": [
            {"timestamp": 1_790_000_000, "summary": {"totalTokensCliff": 50_000}},
            {"timestamp": 1_791_000_000, "summary": {"totalTokensCliff": 10_000}},
            {"timestamp": 1_792_000_000,
             "cliffAllocations": [{"amount": 20_000}, {"amount": 15_000}]},
        ]},
    }
    rows = extract_cliffs(payload, "ARBUSDT", "arbitrum", min_frac=0.03)
    assert len(rows) == 2                    # 5% and 3.5% pass; 1% filtered
    fracs = sorted(r[3] for r in rows)
    assert fracs == [pytest.approx(0.035), pytest.approx(0.05)]
    assert all(r[0] == "arbitrum" and r[1] == "ARBUSDT" for r in rows)


def test_extract_cliffs_no_denominator():
    assert extract_cliffs({"metadata": {"unlockEvents": [
        {"timestamp": 1, "summary": {"totalTokensCliff": 1}}]}},
        "XUSDT", "x", 0.03) == []
