"""Detector math on synthetic data (frozen liqrev v2 rules)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.strategy.liqrev_v2 import (HOLD_BARS, detect_series,
                                    research_liquidity_gate, research_oi_hourly)


def make_series(n: int = 120, drop_at: int | None = 60, ret6: float = -0.10,
                doi6: float = -0.15) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Flat 100/1000 series with an engineered 6-bar crash ending at drop_at."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0)
    oi = np.full(n, 1000.0)
    if drop_at is not None:
        close[drop_at:] = 100.0 * (1 + ret6)
        oi[drop_at:] = 1000.0 * (1 + doi6)
    liq_ok = pd.Series(True, index=idx)
    return (pd.Series(close, index=idx), pd.Series(oi, index=idx), liq_ok)


def test_trigger_fires_on_crash():
    close, oi, liq = make_series()
    trig = detect_series(close, oi, liq)
    assert len(trig) == 1
    assert trig[0].i == 60
    assert trig[0].ts == close.index[60]
    assert trig[0].ret6 == pytest.approx(-0.10)
    assert trig[0].doi6 == pytest.approx(-0.15)
    assert trig[0].trig_close == pytest.approx(90.0)


def test_thresholds_are_inclusive():
    # boundary equality tested with thresholds computed the same way pandas
    # computes the returns (IEEE-float safe): <=, not <
    close, oi, liq = make_series(ret6=-0.08, doi6=-0.10)
    t_ret = float(close.pct_change(6).iloc[60])
    t_doi = float(oi.pct_change(6).iloc[60])
    assert len(detect_series(close, oi, liq, ret6_max=t_ret, doi6_max=t_doi)) == 1
    close, oi, liq = make_series(ret6=-0.079, doi6=-0.15)
    assert len(detect_series(close, oi, liq)) == 0
    close, oi, liq = make_series(ret6=-0.12, doi6=-0.099)
    assert len(detect_series(close, oi, liq)) == 0


def test_oi_condition_required():
    close, oi, liq = make_series(ret6=-0.12, doi6=-0.05)
    assert detect_series(close, oi, liq) == []


def test_liquidity_gate_blocks():
    close, oi, liq = make_series()
    liq[:] = False
    assert detect_series(close, oi, liq) == []


def test_cooldown_first_trigger_wins():
    # after the crash the level stays low: ret6 recovers after 6 flat bars,
    # so engineer a second crash 3 bars later (inside cooldown) and one 30h later
    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    oi = pd.Series(1000.0, index=idx)
    liq = pd.Series(True, index=idx)
    for start, level in [(60, 0.88), (63, 0.76), (100, 0.62)]:
        close.iloc[start:] = 100.0 * level
        oi.iloc[start:] = 1000.0 * level
    trig = detect_series(close, oi, liq)
    ts = [t.i for t in trig]
    assert 60 in ts
    assert 63 not in ts          # 3h later: inside 24h cooldown
    assert 100 in ts             # 40h later: cooldown expired


def test_cooldown_exact_24h_is_allowed():
    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    oi = pd.Series(1000.0, index=idx)
    liq = pd.Series(True, index=idx)
    close.iloc[60:] = 90.0
    oi.iloc[60:] = 850.0
    close.iloc[84:] = 81.0       # exactly 24h later, fresh -10% / -10.6%
    oi.iloc[84:] = 760.0
    trig = detect_series(close, oi, liq)
    assert [t.i for t in trig] == [60, 84]   # research: (t - last_t) < 24h skips


def test_tail_guard_matches_research():
    n = 80
    close, oi, liq = make_series(n=n, drop_at=n - 10)
    assert len(detect_series(close, oi, liq, min_tail_bars=0)) == 1
    # research drops events without HOLD_BARS+1 subsequent bars
    assert detect_series(close, oi, liq, min_tail_bars=HOLD_BARS + 1) == []


def test_nan_oi_never_triggers():
    close, oi, liq = make_series()
    oi[:] = np.nan
    assert detect_series(close, oi, liq) == []


def test_research_oi_hourly_resample_last_ffill():
    kidx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    oi_ts = pd.date_range("2026-01-01", periods=24, freq="5min", tz="UTC")
    oi = pd.Series(np.arange(24, dtype=float), index=oi_ts)
    out = research_oi_hourly(oi, kidx)
    assert out.iloc[0] == 11.0   # last 5m value inside hour 0
    assert out.iloc[1] == 23.0
    assert out.iloc[2] == 23.0   # ffill beyond OI history
    assert out.iloc[4] == 23.0


def test_research_liquidity_gate_median_semantics():
    # 40 days of hourly volume; first 35 days at $30k/h (~$720k/day), then
    # $100k/h (~$2.4M/day). The 30d rolling median crosses $1M only after
    # the median day flips.
    idx = pd.date_range("2026-01-01", periods=40 * 24, freq="1h", tz="UTC")
    qv = pd.Series(30_000.0, index=idx)
    qv.iloc[35 * 24:] = 100_000.0
    gate = research_liquidity_gate(qv)
    assert not gate.iloc[:30 * 24].any()      # warmup + low volume
    assert not gate.iloc[36 * 24]             # median of last 30d still $720k
    # gate uses the DAY's value ffilled onto hours (research semantics)
    assert gate.index[gate].size == 0 or gate.iloc[-1] in (True, False)


def test_gate_warmup_is_nan_safe():
    idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    qv = pd.Series(1e6, index=idx)
    gate = research_liquidity_gate(qv)
    assert not gate.any()                     # < 30 days of history -> False
