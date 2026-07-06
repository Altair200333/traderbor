from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from traderbot_ai.screener.artifacts import write_scan_artifacts
from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.data import CandleFrame, load_closed, validate_frame
from traderbot_ai.screener.gates import evaluate_signal_gates, gates_pass, scan_global_blocks, state_blocks
from traderbot_ai.screener.indicators import atr_wilder, ema, roc, rolling_median_previous, rsi_wilder
from traderbot_ai.screener.patterns import PatternHit, detect_patterns
from traderbot_ai.screener.plan import build_plan_primitives
from traderbot_ai.screener.render import to_canonical_json, to_markdown_table
from traderbot_ai.screener.screener import scan
from traderbot_ai.screener.state import OpenPosition, ScreenerStateStore, TradingState, state_from_wallet_and_events
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.screener.market import INTERVAL_MS


BTC = "BTCUSDT"
BASE_MS = 1_704_067_200_000
ONE_HOUR = INTERVAL_MS["1h"]


def candle(symbol: str, open_time: int, open_price: float, high: float, low: float, close: float, volume: float = 1.0, interval: str = "1h") -> Candle:
    interval_ms = INTERVAL_MS[interval]
    return Candle(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=open_time + interval_ms - 1,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class ScreenerIndicatorTests(unittest.TestCase):
    def test_ema_uses_sma_seed_and_same_length_output(self) -> None:
        self.assertEqual(ema([1, 2, 3, 4, 5], 3), [None, None, 2.0, 3.0, 4.0])
        result = ema(list(range(1, 22)), 20)
        self.assertIsNone(result[18])
        self.assertEqual(result[19], 10.5)

    def test_rsi_wilder_seed_edges_and_mixed_vector(self) -> None:
        self.assertEqual(rsi_wilder([1, 2, 3, 4, 5], 3)[3], 100.0)
        self.assertEqual(rsi_wilder([5, 4, 3, 2, 1], 3)[3], 0.0)
        self.assertEqual(rsi_wilder([5, 5, 5, 5, 5], 3)[3], 100.0)
        mixed = rsi_wilder([10, 12, 11, 13, 12, 14], 3)
        self.assertAlmostEqual(mixed[3], 80.0)
        self.assertAlmostEqual(mixed[4], 61.5384615385)
        self.assertAlmostEqual(mixed[5], 77.2727272727)

    def test_atr_wilder_uses_gap_true_range_and_seed_index(self) -> None:
        highs = [10, 13, 12, 16, 15]
        lows = [9, 10, 9, 13, 13]
        closes = [9.5, 11, 10, 14, 14]
        result = atr_wilder(highs, lows, closes, 3)
        self.assertEqual(result[:3], [None, None, None])
        self.assertAlmostEqual(result[3], (3.5 + 3.0 + 6.0) / 3.0)

    def test_roc_and_volume_median_previous(self) -> None:
        result = roc([100, 110, 121], 1)
        self.assertIsNone(result[0])
        self.assertAlmostEqual(result[1], 0.1)
        self.assertAlmostEqual(result[2], 0.1)
        volumes = [1.0] * 24 + [100.0]
        med = rolling_median_previous(volumes, 24)
        self.assertEqual(med[-1], 1.0)


class ScreenerDataTests(unittest.TestCase):
    def test_load_closed_uses_open_plus_interval_boundary_not_close_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(BTC, BASE_MS, 100, 101, 99, 100)])

            before_close = load_closed(cache.path, BTC, ONE_HOUR, BASE_MS + ONE_HOUR - 1, limit=10)
            at_close = load_closed(cache.path, BTC, ONE_HOUR, BASE_MS + ONE_HOUR, limit=10)

        self.assertEqual(before_close.length, 0)
        self.assertEqual(at_close.length, 1)
        self.assertEqual(at_close.close[0], 100.0)

    def test_validate_frame_detects_missing_tail(self) -> None:
        frame = CandleFrame(
            symbol=BTC,
            interval_ms=ONE_HOUR,
            open_time=[BASE_MS],
            open=[100.0],
            high=[101.0],
            low=[99.0],
            close=[100.0],
            volume=[1.0],
        )
        issue = validate_frame(frame, BASE_MS + 3 * ONE_HOUR, min_bars=1)
        self.assertIsNotNone(issue)
        self.assertEqual(issue.reason, "missing_tail")

    def test_validate_frame_detects_bad_ohlcv_and_gaps(self) -> None:
        bad = CandleFrame(BTC, ONE_HOUR, [BASE_MS], [100.0], [99.0], [98.0], [100.0], [1.0])
        self.assertEqual(validate_frame(bad, BASE_MS + ONE_HOUR, 1).status, "bad_data")
        gap = CandleFrame(BTC, ONE_HOUR, [BASE_MS, BASE_MS + 2 * ONE_HOUR], [1, 1], [2, 2], [0, 0], [1, 1], [1, 1])
        self.assertEqual(validate_frame(gap, BASE_MS + 3 * ONE_HOUR, 1).reason, "gap")
        unaligned = CandleFrame(BTC, ONE_HOUR, [BASE_MS + 1], [1], [2], [0], [1], [1])
        self.assertEqual(validate_frame(unaligned, BASE_MS + ONE_HOUR + 1, 1).reason, "unaligned_open_time")


class ScreenerPatternGatePlanTests(unittest.TestCase):
    def test_p1_boundary_excludes_current_bar(self) -> None:
        frame = _frame_from_prices([99.0] * 20 + [101.0], highs=[100.0] * 20 + [150.0])
        hits = detect_patterns(frame, [None] * frame.length, [None] * frame.length, [None] * frame.length, ScreenerConfig())
        self.assertIn("P1", [hit.id for hit in hits if hit.side == "long"])

        no_hit_frame = _frame_from_prices([99.0] * 20 + [100.0], highs=[100.0] * 20 + [150.0])
        no_hits = detect_patterns(no_hit_frame, [None] * no_hit_frame.length, [None] * no_hit_frame.length, [None] * no_hit_frame.length, ScreenerConfig())
        self.assertNotIn("P1", [hit.id for hit in no_hits if hit.side == "long"])

    def test_p2_uses_per_bar_ema_and_loose_touch_semantics(self) -> None:
        closes = [100.0] * 23 + [103.0]
        highs = [101.0] * 23 + [104.0]
        lows = [99.0] * 21 + [98.0, 98.0, 102.0]
        frame = _frame_from_prices(closes, highs=highs, lows=lows)
        ema20 = [100.0] * frame.length
        ema50 = [90.0] * frame.length
        hits = detect_patterns(frame, ema20, ema50, [2.0] * frame.length, ScreenerConfig())
        self.assertIn("P2", [hit.id for hit in hits if hit.side == "long"])

    def test_p3_detects_long_and_short_compression_breakouts(self) -> None:
        cfg = ScreenerConfig()
        atr = [2.0] * 80
        atr[-1] = 1.0
        long_frame = _frame_from_prices(
            [100.0] * 79 + [101.0],
            highs=[100.0] * 79 + [101.2],
            lows=[99.0] * 80,
        )
        short_frame = _frame_from_prices(
            [100.0] * 79 + [99.0],
            highs=[101.0] * 80,
            lows=[100.0] * 79 + [98.8],
        )

        long_hits = detect_patterns(long_frame, [100.0] * 80, [100.0] * 80, atr, cfg)
        short_hits = detect_patterns(short_frame, [100.0] * 80, [100.0] * 80, atr, cfg)

        self.assertIn("P3", [hit.id for hit in long_hits if hit.side == "long"])
        self.assertIn("P3", [hit.id for hit in short_hits if hit.side == "short"])

    def test_short_signal_gates_accept_valid_p3_breakout(self) -> None:
        cfg = ScreenerConfig()
        gates = evaluate_signal_gates(
            "short",
            close=99.0,
            roc_4h=-0.03,
            roc_24h=-0.04,
            roc_1h_last=-0.01,
            vol_ratio=2.5,
            rsi=35.0,
            atr=2.0,
            atr_pct=0.02,
            ema20=100.0,
            ema50=101.0,
            btc_roc_4h=0.0,
            funding=None,
            patterns=[PatternHit(id="P3", side="short", boundary_price=100.0)],
            cfg=cfg,
        )

        self.assertTrue(gates_pass(gates))
        self.assertIsNone(gates["S8"].passed)

    def test_s9a_spike_and_s9c_overextended_breakout_fail(self) -> None:
        cfg = ScreenerConfig()
        p1 = PatternHit(id="P1", side="long", boundary_price=100.0)
        gates = evaluate_signal_gates(
            "long",
            close=103.0,
            roc_4h=0.0395,
            roc_24h=0.05,
            roc_1h_last=0.038,
            vol_ratio=3.0,
            rsi=65.0,
            atr=2.0,
            atr_pct=0.02,
            ema20=102.0,
            ema50=99.0,
            btc_roc_4h=0.0,
            funding=None,
            patterns=[p1],
            cfg=cfg,
        )
        self.assertFalse(gates["S9a"].passed)
        self.assertFalse(gates["S9c"].passed)
        self.assertIsNone(gates["S8"].passed)

    def test_plan_primitives_prioritize_p2_and_keep_stop_feasibility(self) -> None:
        cfg = ScreenerConfig()
        frame = _frame_from_prices([100.0] * 30, lows=[99.0] * 27 + [97.0, 98.0, 99.0])
        patterns = [
            PatternHit(id="P1", side="long", boundary_price=99.0),
            PatternHit(id="P2", side="long", boundary_price=97.0),
            PatternHit(id="P3", side="long", boundary_price=99.0),
        ]
        plan = build_plan_primitives("long", patterns, frame, atr_value=2.0, cfg=cfg)
        self.assertEqual(plan.pattern_used, "P2")
        self.assertAlmostEqual(plan.invalidation_price, 96.5)
        self.assertAlmostEqual(plan.d_struct, 0.035)
        self.assertAlmostEqual(plan.d_atr, 0.024)
        self.assertTrue(plan.stop_feasible)


class ScreenerScanArtifactTests(unittest.TestCase):
    def test_scan_is_deterministic_and_artifact_filename_is_windows_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            candles = []
            for index in range(168):
                open_time = BASE_MS + index * ONE_HOUR
                price = 100.0 + index * 0.01
                candles.append(candle(BTC, open_time, price, price + 1.0, price - 1.0, price, volume=10.0))
            cache.upsert_candles(candles)
            as_of = BASE_MS + 168 * ONE_HOUR

            first = scan([BTC], as_of, cache_path=cache.path)
            second = scan([BTC], as_of, cache_path=cache.path)
            artifacts = write_scan_artifacts(first, "run:bad/name", root=Path(tmp) / "artifacts")
            self.assertTrue(Path(artifacts["artifact_path"]).exists())
            self.assertNotIn(":", Path(artifacts["artifact_path"]).name)
            payload = json.loads(Path(artifacts["artifact_path"]).read_text(encoding="utf-8"))

        self.assertEqual(to_canonical_json(first), to_canonical_json(second))
        self.assertEqual(payload["as_of_ms"], as_of)
        self.assertIn("as_of_iso", payload)

    def test_markdown_renders_missing_as_dash(self) -> None:
        result = scan([BTC], BASE_MS + ONE_HOUR, cache_path=Path(tempfile.gettempdir()) / "missing-screener-cache.sqlite3")
        markdown = to_markdown_table(result)
        self.assertIn("| sym |", markdown)
        self.assertNotIn("None", markdown)

    def test_state_store_updates_from_signal_candidate_before_state(self) -> None:
        store = ScreenerStateStore()
        store.update_from_scan_rows([{"symbol": BTC, "signal_candidate_before_state": "long", "candidate": None}], BASE_MS)
        self.assertEqual(store.last_candidate_ts[BTC], BASE_MS)

    def test_trading_state_reconstructs_risk_counters_from_exchange_events(self) -> None:
        as_of = BASE_MS + 30 * ONE_HOUR
        wallet = {"totals": {"equity_usdt": 1000.0}, "open_positions": []}
        events = [
            {
                "type": "place_order",
                "payload": {
                    "category": "linear",
                    "status": "Filled",
                    "symbol": BTC,
                    "created_at_ms": as_of - ONE_HOUR,
                },
            },
            {
                "type": "position_closed",
                "payload": {
                    "symbol": BTC,
                    "exit_time_ms": as_of - 50 * 60_000,
                    "exit_reason": "StopLoss",
                    "realized_pnl_usdt": -10.0,
                },
            },
            {
                "type": "position_closed",
                "payload": {
                    "symbol": "ETHUSDT",
                    "exit_time_ms": as_of - 40 * 60_000,
                    "exit_reason": "TakeProfit",
                    "realized_pnl_usdt": 5.0,
                },
            },
            {
                "type": "position_closed",
                "payload": {
                    "symbol": "SOLUSDT",
                    "exit_time_ms": as_of - 30 * 60_000,
                    "exit_reason": "StopLoss",
                    "realized_pnl_usdt": -15.0,
                },
            },
            {
                "type": "position_closed",
                "payload": {
                    "symbol": "ADAUSDT",
                    "exit_time_ms": as_of - 20 * 60_000,
                    "exit_reason": "StopLoss",
                    "realized_pnl_usdt": -5.0,
                },
            },
        ]

        state = state_from_wallet_and_events(wallet, events, as_of, last_candidate_ts={BTC: BASE_MS})

        self.assertEqual(state.trades_opened_today, 1)
        self.assertEqual(state.last_candidate_ts[BTC], BASE_MS)
        self.assertEqual(state.last_stopout_ts[BTC], as_of - 50 * 60_000)
        self.assertEqual(state.last_stopout_ts["SOLUSDT"], as_of - 30 * 60_000)
        self.assertEqual(state.consecutive_stopouts, 2)
        self.assertAlmostEqual(state.daily_realized_pnl_pct, -0.025)
        self.assertAlmostEqual(state.weekly_realized_pnl_pct, -0.025)

    def test_take_profit_with_negative_net_pnl_is_not_stopout(self) -> None:
        as_of = BASE_MS + 30 * ONE_HOUR
        wallet = {"totals": {"equity_usdt": 1000.0}, "open_positions": []}
        events = [
            {
                "type": "position_closed",
                "payload": {
                    "symbol": BTC,
                    "exit_time_ms": as_of - ONE_HOUR,
                    "exit_reason": "TakeProfit",
                    "realized_pnl_usdt": -0.01,
                },
            }
        ]

        state = state_from_wallet_and_events(wallet, events, as_of)

        self.assertEqual(state.last_stopout_ts, {})
        self.assertEqual(state.consecutive_stopouts, 0)

    def test_max_same_direction_is_side_specific_not_scan_global(self) -> None:
        cfg = ScreenerConfig(max_same_direction=2)
        state = TradingState(
            open_positions=[
                OpenPosition(symbol="ETHUSDT", side="long"),
                OpenPosition(symbol="SOLUSDT", side="long"),
            ]
        )

        self.assertNotIn("max_same_direction", scan_global_blocks(state, cfg))
        long_blocks, _ = state_blocks("XRPUSDT", "long", state, BASE_MS, cfg)
        short_blocks, _ = state_blocks("XRPUSDT", "short", state, BASE_MS, cfg)

        self.assertIn("max_same_direction", long_blocks)
        self.assertNotIn("max_same_direction", short_blocks)

    def test_state_cooldown_boundaries_are_strictly_less_than_duration(self) -> None:
        cfg = ScreenerConfig()
        state = TradingState(last_candidate_ts={BTC: BASE_MS}, last_stopout_ts={BTC: BASE_MS})

        before_candidate, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_candidate_ms - 1, cfg)
        at_candidate, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_candidate_ms, cfg)
        before_stopout, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_stopout_ms - 1, cfg)
        at_stopout, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_stopout_ms, cfg)

        self.assertIn("cooldown_candidate", before_candidate)
        self.assertNotIn("cooldown_candidate", at_candidate)
        self.assertIn("cooldown_stopout", before_stopout)
        self.assertNotIn("cooldown_stopout", at_stopout)

    def test_scan_reports_global_blocks_without_per_symbol_global_block_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            candles = []
            for index in range(168):
                open_time = BASE_MS + index * ONE_HOUR
                price = 100.0 + index * 0.01
                candles.append(candle(BTC, open_time, price, price + 1.0, price - 1.0, price, volume=10.0))
            cache.upsert_candles(candles)
            result = scan([BTC], BASE_MS + 168 * ONE_HOUR, state=TradingState(halt=True), cache_path=cache.path)

        self.assertIn("halt_active", result.global_blocks)
        self.assertFalse(any(block.startswith("global:") for row in result.symbols for block in row.blocked_by))

    def test_scan_reports_implicit_btc_data_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            eth = "ETHUSDT"
            candles = []
            for index in range(168):
                open_time = BASE_MS + index * ONE_HOUR
                price = 100.0 + index * 0.01
                candles.append(candle(eth, open_time, price, price + 1.0, price - 1.0, price, volume=10.0))
            cache.upsert_candles(candles)
            result = scan([eth], BASE_MS + 168 * ONE_HOUR, cache_path=cache.path)

        self.assertIn("BTCUSDT:insufficient_data", result.data_warnings)


def _frame_from_prices(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> CandleFrame:
    highs = highs or [close + 1.0 for close in closes]
    lows = lows or [close - 1.0 for close in closes]
    return CandleFrame(
        symbol=BTC,
        interval_ms=ONE_HOUR,
        open_time=[BASE_MS + index * ONE_HOUR for index in range(len(closes))],
        open=list(closes),
        high=list(highs),
        low=list(lows),
        close=list(closes),
        volume=[1.0] * len(closes),
    )


if __name__ == "__main__":
    unittest.main()
