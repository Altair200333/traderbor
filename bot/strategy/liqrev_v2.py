"""Liqrev v2 strategy: the frozen detector + BTC-context sizing overlay.

Detector math reproduces research/scanner_lab/liqrev_v2.py:detect_events
EXACTLY (P1 signal-parity requirement): on a positional 1h grid,
  trigger:   ret_6h <= -8%  AND  d_oi_6h <= -10%   (pct_change over 6 BARS)
  liquidity: 30d-median daily quote volume > $1M
  cooldown:  24h per symbol, first trigger wins
Execution intent (deploy-spec section 1): post-only limit BUY at trigger-bar
close, TTL 1h, disaster stop -20% server-side, exit at trigger close + 24h
(= close of research bar i+24), slot weight = min(2, 2*P(-btc_ret_6h)) against
the frozen DEV distribution (bot/artifacts/liqrev_overlay.json — exported
verbatim from research, never re-derived).

detect_series() is the single detector code path used by BOTH the live on_bar
loop and the P1 replay harness (bot/replay.py).
"""
from __future__ import annotations

import json
import logging
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bot.config import ARTIFACTS_DIR, LiqrevConfig
from bot.journal import Journal
from bot.strategy.base import EntryIntent, Intent, MarketState

log = logging.getLogger("bot.liqrev")

HOLD_BARS = 24
H_MS = 3_600_000


# --------------------------------------------------------------- detector ---
@dataclass(frozen=True)
class Trigger:
    ts: pd.Timestamp           # trigger bar OPEN time (research convention)
    i: int                     # positional index into the series
    ret6: float
    doi6: float
    trig_close: float


def research_oi_hourly(oi: pd.Series, kline_index: pd.DatetimeIndex) -> pd.Series:
    """5m (or finer) OI series -> hourly-last aligned to the kline index,
    forward-filled. Identical to the research pipeline."""
    return oi.resample("1h").last().reindex(kline_index).ffill()


def research_liquidity_gate(quote_volume_1h: pd.Series,
                            gate_usd: float = 1_000_000.0) -> pd.Series:
    """Research-pipeline liquidity gate: 30d rolling median of daily quote
    volume, day value forward-filled onto the hourly grid. NOTE: within a day
    this uses that day's FULL volume (the research pipeline's semantics; the
    live bot instead uses the last 30 COMPLETED days — causal, documented)."""
    dvol30 = quote_volume_1h.resample("1D").sum().rolling(30).median()
    return dvol30.reindex(quote_volume_1h.index, method="ffill") > gate_usd


def detect_series(close: pd.Series, oi_h: pd.Series, liq_ok: pd.Series,
                  ret6_max: float = -0.08, doi6_max: float = -0.10,
                  cooldown_h: int = 24, min_tail_bars: int = 0) -> list[Trigger]:
    """The frozen detector over an aligned hourly series (positional grid).

    min_tail_bars: research parity guard — skip triggers with fewer than that
    many bars after them (research uses 25 = HOLD_BARS+1 so forward returns
    exist). Live runs with 0 (the newest bar must be able to trigger).
    Cooldown-skipped and tail-skipped triggers do NOT start a cooldown
    (identical to research detect_events)."""
    import warnings
    with warnings.catch_warnings():
        # research liqrev_v2.py uses the default pct_change fill_method ('pad');
        # we keep IDENTICAL semantics (parity) and just silence the deprecation
        warnings.simplefilter("ignore", FutureWarning)
        ret6 = close.pct_change(6)
        doi6 = oi_h.pct_change(6)
    mask = ((ret6 <= ret6_max) & (doi6 <= doi6_max) & liq_ok).fillna(False)
    out: list[Trigger] = []
    last_t: pd.Timestamp | None = None
    n = len(close)
    cooldown = pd.Timedelta(hours=cooldown_h)
    for t in close.index[mask]:
        if last_t is not None and (t - last_t) < cooldown:
            continue
        i = int(close.index.get_loc(t))
        if min_tail_bars and i + min_tail_bars >= n:
            continue
        last_t = t
        out.append(Trigger(ts=t, i=i, ret6=float(ret6.loc[t]), doi6=float(doi6.loc[t]),
                           trig_close=float(close.iloc[i])))
    return out


# ---------------------------------------------------------------- overlay ---
class Overlay:
    """Frozen DEV-percentile map: weight = min(2, 2 * P(-btc_ret_6h))."""

    def __init__(self, scores_sorted: list[float]):
        if not scores_sorted:
            raise ValueError("empty overlay score distribution")
        self.scores = sorted(scores_sorted)

    @classmethod
    def load(cls, path: Path | None = None) -> "Overlay":
        p = path or (ARTIFACTS_DIR / "liqrev_overlay.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data["scores_sorted"])

    def percentile(self, btc_ret_6h: float) -> float:
        return bisect_right(self.scores, -btc_ret_6h) / len(self.scores)

    def weight(self, btc_ret_6h: float | None) -> float:
        """Live rule. btc_ret_6h missing => weight 1.0 (neutral) + caller
        should alarm; cannot happen while the BTC feed is healthy."""
        if btc_ret_6h is None or not math.isfinite(btc_ret_6h):
            return 1.0
        return min(2.0, 2.0 * self.percentile(btc_ret_6h))


def session_tag(hour_utc: int) -> str:
    return "ASIA" if hour_utc < 8 else ("EU" if hour_utc < 16 else "US")


# --------------------------------------------------------------- strategy ---
class LiqrevStrategy:
    """Live wrapper: evaluates the frozen detector on the newest completed bar."""

    name = "liqrev_v2"
    WINDOW_BARS = 64           # detector needs 7; headroom for gap diagnostics

    def __init__(self, cfg: LiqrevConfig, journal: Journal,
                 overlay: Overlay | None = None):
        self.cfg = cfg
        self.mode = cfg.mode
        self.journal = journal
        self.overlay = overlay or Overlay.load()

    def _btc_ret6(self, market: MarketState) -> float | None:
        s = market.series("BTCUSDT", 8)
        if s is None or len(s.close) < 7:
            return None
        c = list(s.close)
        return c[-1] / c[-7] - 1.0

    def on_bar(self, market: MarketState) -> list[Intent]:
        if self.mode == "off":
            return []
        intents: list[Intent] = []
        btc_ret6 = self._btc_ret6(market)
        if btc_ret6 is None:
            self.journal.event("warn", "liqrev", "btc_ret_6h unavailable; weight=1.0")
        bar_close_ms = market.bar_close_ms
        last_ts = pd.Timestamp(market.bar_open_ms, unit="ms", tz="UTC")
        for sym in market.symbols():
            s = market.series(sym, self.WINDOW_BARS)
            if s is None or len(s.close) < 7 or s.bar_ms[-1] != market.bar_open_ms:
                continue
            if not market.liquidity_ok(sym):
                continue
            close, oi = s.to_pandas()
            liq_ok = pd.Series(True, index=close.index)  # gated above (causal, live)
            trig = [t for t in detect_series(close, oi, liq_ok,
                                             self.cfg.ret6_max, self.cfg.doi6_max,
                                             self.cfg.cooldown_h)
                    if t.ts == last_ts]
            if not trig:
                continue
            t = trig[0]
            # journal cooldown is authoritative (window may not span 24h fully)
            prev_ms = self.journal.last_signal_ms(self.name, sym)
            if prev_ms is not None and bar_close_ms - prev_ms < self.cfg.cooldown_h * H_MS:
                continue
            weight = self.overlay.weight(btc_ret6)
            self.journal.write_signal(
                strategy=self.name, mode=self.mode, symbol=sym, ts_ms=bar_close_ms,
                ret6=t.ret6, doi6=t.doi6, btc_ret6=btc_ret6, weight=weight,
                mw_tag="mw" if weight >= 1.0 else "idio",
                session=session_tag(last_ts.hour), limit_px=t.trig_close)
            log.info("SIGNAL %s ret6=%.4f doi6=%.4f w=%.2f px=%s",
                     sym, t.ret6, t.doi6, weight, t.trig_close)
            intents.append(EntryIntent(
                strategy=self.name, symbol=sym, side="Buy",
                limit_price=t.trig_close, weight=weight, ttl_s=self.cfg.ttl_s,
                stop_pct=self.cfg.stop_pct,
                exit_at_ms=bar_close_ms + self.cfg.hold_h * H_MS,
                meta={"ret6": t.ret6, "doi6": t.doi6, "btc_ret6": btc_ret6,
                      "trigger_bar_ms": market.bar_open_ms,
                      "signal_ts_ms": bar_close_ms}))
        return intents
