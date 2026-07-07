"""Scanner provider seam: resolution, config plumbing, replay routing, and the
scanner-v2 contract on a synthetic cache.

Fast tests only; wide-data and e2e tests live in test_scanner_v2_wide.py
(env-gated TRADERBOT_RUN_SLOW_SCANNER_V2_TESTS=1).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from traderbot_ai.screener import ScreenerConfig, ScreenerStateStore
from traderbot_ai.screener.plan import PlanPrimitives
from traderbot_ai.screener.providers import (
    _DEFAULT_ARTIFACTS,
    resolve_scanner_provider,
)
from traderbot_ai.screener.render import to_markdown_table
from traderbot_ai.screener.screener import ScanResult, SymbolRow
from traderbot_ai.screener.state import OpenPosition, TradingState, candidate_cooldown_key
from traderbot_ai.simulator.exchange_replay import (
    _build_deterministic_scan_context,
    build_exchange_replay_config,
)
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache

BTC = "BTCUSDT"
ALT = "ALTUSDT"
ONE_HOUR = 3_600_000
BASE_MS = 1_749_999_600_000  # hour-aligned (3_600_000 * 486_111)


def _v2_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        import pandas  # noqa: F401
    except Exception:
        return False
    return (_DEFAULT_ARTIFACTS / "model.txt").exists()


def _mk_result(symbol: str, candidate: str | None = "long") -> ScanResult:
    plan = PlanPrimitives(
        pattern_used="P1", boundary_price=100.0, trigger_age_bars=0,
        invalidation_price=99.0, d_atr=0.02, d_struct=0.025, d_noise=0.015,
        d_final=0.025, stop_feasible=True, tp_rr_default=3.0, ref_entry=103.0)
    row = SymbolRow(symbol=symbol, status="ok", close=103.0, candidate=candidate,
                    candidate_quality="hard" if candidate else None,
                    plan=plan if candidate else None)
    return ScanResult(as_of_ms=BASE_MS, as_of_iso="x", screener_version="stub",
                      config_hash="c", symbols=[row],
                      candidates=[symbol] if candidate else [])


def _synthetic_cache(tmp: Path, btc_up: bool = True, extra_alt_bar: bool = False) -> LocalMarketCache:
    cache = LocalMarketCache(path=tmp / "market_cache.sqlite3")
    candles: list[Candle] = []
    n = 1300
    for i in range(n):
        t = BASE_MS + i * ONE_HOUR
        drift = 1.0005 if btc_up else 0.9995
        px = 50_000.0 * (drift ** i)
        candles.append(Candle(symbol=BTC, interval="1h", open_time=t,
                              close_time=t + ONE_HOUR - 1, open=px, high=px * 1.001,
                              low=px * 0.999, close=px, volume=100.0,
                              quote_volume=px * 100, trade_count=100,
                              taker_buy_base_volume=50.0))
        if i < n - 1:
            alt_close = 100.0 + (i % 5) * 0.1
        else:
            alt_close = 103.0  # trigger bar: P1 breakout, vol spike
        vol = 900.0 if i == n - 1 else 100.0
        hi = alt_close + (0.2 if i == n - 1 else 0.5)
        lo = alt_close - (1.2 if i == n - 1 else 0.5)
        candles.append(Candle(symbol=ALT, interval="1h", open_time=t,
                              close_time=t + ONE_HOUR - 1, open=alt_close, high=hi,
                              low=lo, close=alt_close, volume=vol,
                              quote_volume=alt_close * vol, trade_count=int(vol),
                              taker_buy_base_volume=vol * 0.5))
    if extra_alt_bar:
        t = BASE_MS + n * ONE_HOUR
        candles.append(Candle(symbol=ALT, interval="1h", open_time=t,
                              close_time=t + ONE_HOUR - 1, open=50.0, high=50.0,
                              low=50.0, close=50.0, volume=1.0,
                              quote_volume=50.0, trade_count=1,
                              taker_buy_base_volume=0.5))
    cache.upsert_candles(candles)
    return cache


SYNTH_AS_OF = BASE_MS + 1300 * ONE_HOUR  # last synthetic bar closed


class ResolveProviderTests(unittest.TestCase):
    def test_default_is_legacy(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRADERBOT_SCANNER_PROVIDER", None)
            self.assertEqual(resolve_scanner_provider(None), "legacy")

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"TRADERBOT_SCANNER_PROVIDER": "v2"}):
            self.assertEqual(resolve_scanner_provider(None), "v2")

    def test_explicit_beats_env(self) -> None:
        with patch.dict(os.environ, {"TRADERBOT_SCANNER_PROVIDER": "v2"}):
            self.assertEqual(resolve_scanner_provider("legacy"), "legacy")

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_scanner_provider("nope")

    def test_config_carries_provider(self) -> None:
        cfg = build_exchange_replay_config(
            symbols=BTC, start_time=BASE_MS, end_time=BASE_MS + ONE_HOUR,
            decision_interval="1h", screener_mode="deterministic",
            scanner_provider="v2")
        self.assertEqual(cfg.scanner_provider, "v2")

    def test_config_invalid_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_exchange_replay_config(
                symbols=BTC, start_time=BASE_MS, end_time=BASE_MS + ONE_HOUR,
                scanner_provider="bogus")


class ReplayRoutingTests(unittest.TestCase):
    def _context(self, provider: str, legacy_stub: ScanResult, v2_stub: ScanResult) -> dict:
        with tempfile.TemporaryDirectory() as td:
            events = Path(td) / "events.jsonl"
            events.write_text("")
            config = build_exchange_replay_config(
                symbols=[BTC, ALT], start_time=BASE_MS, end_time=BASE_MS + ONE_HOUR,
                decision_interval="1h", screener_mode="deterministic",
                events_path=events, state_path=Path(td) / "s.json",
                replay_path=Path(td) / "r.jsonl", scanner_provider=provider)
            with patch("traderbot_ai.simulator.exchange_replay.state_from_wallet_and_events",
                       return_value=TradingState()), \
                 patch("traderbot_ai.simulator.exchange_replay.run_screener",
                       return_value=legacy_stub) as legacy_mock, \
                 patch("traderbot_ai.screener.providers.scan_v2",
                       return_value=v2_stub) as v2_mock, \
                 patch("traderbot_ai.simulator.exchange_replay.write_scan_artifacts",
                       return_value={"artifact_path": "a", "sha256": "h"}):
                context = _build_deterministic_scan_context(
                    config, SimpleNamespace(path=Path(td) / "cache.sqlite3"),
                    {}, BASE_MS, ScreenerConfig(), ScreenerStateStore())
            context["_legacy_called"] = legacy_mock.called
            context["_v2_called"] = v2_mock.called
            return context

    def test_legacy_route(self) -> None:
        ctx = self._context("legacy", _mk_result(BTC), _mk_result(ALT))
        self.assertTrue(ctx["_legacy_called"])
        self.assertFalse(ctx["_v2_called"])
        self.assertEqual(ctx["scanner_provider"], "legacy")
        self.assertEqual(ctx["screener_candidates"], [BTC])

    def test_v2_route(self) -> None:
        ctx = self._context("v2", _mk_result(BTC), _mk_result(ALT))
        self.assertFalse(ctx["_legacy_called"])
        self.assertTrue(ctx["_v2_called"])
        self.assertEqual(ctx["scanner_provider"], "v2")
        self.assertEqual(ctx["screener_candidates"], [ALT])
        # plan flows into internal primitives, agent view stays fact-only
        self.assertIsNotNone(ctx["candidate_primitives"][0]["plan"])
        self.assertNotIn("plan", ctx["candidate_view"][0])
        self.assertNotIn("score", ctx["candidate_view"][0])


@unittest.skipUnless(_v2_available(), "scanner-v2 artifacts or deps missing")
class ScannerV2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _scan(self, cache: LocalMarketCache, state: TradingState | None = None,
              min_ev: float = -100.0, as_of: int = SYNTH_AS_OF) -> ScanResult:
        from traderbot_ai.screener.providers import scan_v2
        return scan_v2(symbols=[ALT, BTC], as_of_ms=as_of, cfg=ScreenerConfig(),
                       state=state or TradingState(), cache_path=cache.path,
                       min_ev=min_ev)

    def test_emits_long_p1_candidate_with_feasible_plan(self) -> None:
        result = self._scan(_synthetic_cache(self.tmp))
        self.assertEqual(result.candidates, [ALT])
        row = next(r for r in result.symbols if r.symbol == ALT)
        self.assertEqual(row.candidate, "long")
        self.assertEqual(row.candidate_quality, "hard")
        self.assertEqual(row.plan.pattern_used, "P1")
        self.assertTrue(0.01 <= row.plan.d_final <= 0.04)
        self.assertEqual(row.plan.tp_rr_default, 3.0)
        self.assertAlmostEqual(row.plan.ref_entry, 103.0, places=6)
        self.assertLess(row.plan.invalidation_price, row.plan.ref_entry)
        self.assertIsNotNone(row.candidate_score)          # EV, internal only
        self.assertEqual(result.screener_version, "scanner-v2-0.1.0")
        self.assertTrue(to_markdown_table(result))

    def test_regime_gate_blocks_in_btc_downtrend(self) -> None:
        result = self._scan(_synthetic_cache(self.tmp, btc_up=False))
        self.assertEqual(result.candidates, [])
        self.assertIn("btc_regime_bear", result.global_blocks)

    def test_symbol_day_dedup(self) -> None:
        state = TradingState(last_candidate_ts={
            candidate_cooldown_key(ALT, "long"): SYNTH_AS_OF - ONE_HOUR})
        result = self._scan(_synthetic_cache(self.tmp), state=state)
        self.assertEqual(result.candidates, [])

    def test_open_position_blocks(self) -> None:
        state = TradingState(open_positions=[
            OpenPosition(symbol=ALT, side="long", opened_at_ms=SYNTH_AS_OF - ONE_HOUR)])
        result = self._scan(_synthetic_cache(self.tmp), state=state)
        self.assertEqual(result.candidates, [])
        row = next(r for r in result.symbols if r.symbol == ALT)
        self.assertTrue(row.blocked_by)
        self.assertEqual(row.signal_candidate_before_state, "long")

    def test_deterministic(self) -> None:
        cache = _synthetic_cache(self.tmp)
        a = self._scan(cache).model_dump(mode="json")
        b = self._scan(cache).model_dump(mode="json")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_causal_future_bars_ignored(self) -> None:
        plain = self._scan(_synthetic_cache(Path(tempfile.mkdtemp())))
        with_future = self._scan(_synthetic_cache(self.tmp, extra_alt_bar=True))
        self.assertEqual(
            json.dumps(plain.model_dump(mode="json"), sort_keys=True),
            json.dumps(with_future.model_dump(mode="json"), sort_keys=True))

    def test_min_ev_threshold_filters(self) -> None:
        result = self._scan(_synthetic_cache(self.tmp), min_ev=1e9)
        self.assertEqual(result.candidates, [])

    def test_missing_symbol_reports_insufficient_data(self) -> None:
        from traderbot_ai.screener.providers import scan_v2
        cache = _synthetic_cache(self.tmp)
        result = scan_v2(symbols=["GHOSTUSDT", BTC], as_of_ms=SYNTH_AS_OF,
                         cfg=ScreenerConfig(), state=TradingState(),
                         cache_path=cache.path, min_ev=-100.0)
        row = next(r for r in result.symbols if r.symbol == "GHOSTUSDT")
        self.assertEqual(row.status, "insufficient_data")


if __name__ == "__main__":
    unittest.main()
