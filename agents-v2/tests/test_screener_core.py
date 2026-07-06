from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from traderbot_ai.screener.artifacts import write_scan_artifacts
from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.data import CandleFrame, load_closed, validate_frame
from traderbot_ai.screener.gates import evaluate_signal_gates, gates_pass, scan_global_blocks, state_blocks, stoploss_guard_locked_sides
from traderbot_ai.screener.indicators import atr_wilder, chande_kroll_stop, ema, median_range_pct, range_expansion_last, roc, rolling_median_previous, rolling_std, rsi_wilder, sma, zscore_last
from traderbot_ai.screener.patterns import PatternHit, detect_patterns
from traderbot_ai.screener.plan import build_plan_primitives
from traderbot_ai.screener.render import to_canonical_json, to_markdown_table
from traderbot_ai.screener.score import score_candidate
from traderbot_ai.screener.screener import SymbolRow, _apply_candidate_rank_cap, _retest_seen, get_setup_digest, scan
from traderbot_ai.screener.state import OpenPosition, ScreenerStateStore, StopoutEvent, TradingState, candidate_cooldown_key, state_from_wallet_and_events
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

    def test_p1h_detects_recent_breakout_hold_without_current_bar_breakout(self) -> None:
        closes = [99.0] * 20 + [101.0, 101.2, 101.1]
        highs = [100.0] * 20 + [101.5, 101.4, 101.3]
        frame = _frame_from_prices(closes, highs=highs)
        hits = detect_patterns(frame, [None] * frame.length, [None] * frame.length, [None] * frame.length, ScreenerConfig())
        long_hits = [hit for hit in hits if hit.side == "long"]

        self.assertNotIn("P1", [hit.id for hit in long_hits])
        p1h = next(hit for hit in long_hits if hit.id == "P1H")
        self.assertEqual(p1h.boundary_price, 100.0)
        self.assertEqual(p1h.detail["age_bars"], 2)
        plan = build_plan_primitives("long", long_hits, frame, atr_value=2.0, cfg=ScreenerConfig())
        self.assertEqual(plan.trigger_age_bars, 2)

    def test_retest_seen_requires_touch_after_breakout_trigger(self) -> None:
        false_frame = _frame_from_prices(
            [99.0] * 20 + [101.0, 101.2, 101.1],
            highs=[100.0] * 20 + [101.5, 101.4, 101.3],
            lows=[98.0] * 20 + [99.7, 100.8, 100.9],
        )
        true_frame = _frame_from_prices(
            [99.0] * 20 + [101.0, 101.2, 101.1],
            highs=[100.0] * 20 + [101.5, 101.4, 101.3],
            lows=[98.0] * 20 + [99.7, 100.3, 100.9],
        )

        self.assertFalse(_retest_seen(false_frame, "long", boundary=100.0, atr_value=2.0, trigger_age_bars=2))
        self.assertTrue(_retest_seen(true_frame, "long", boundary=100.0, atr_value=2.0, trigger_age_bars=2))
        self.assertFalse(_retest_seen(true_frame, "long", boundary=100.0, atr_value=2.0, trigger_age_bars=0))
        self.assertIsNone(_retest_seen(true_frame, "long", boundary=100.0, atr_value=2.0, trigger_age_bars=None))

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

    def test_p1h_breakout_hold_still_uses_s9c_breakout_distance(self) -> None:
        cfg = ScreenerConfig()
        p1h = PatternHit(id="P1H", side="long", boundary_price=100.0)
        gates = evaluate_signal_gates(
            "long",
            close=103.0,
            roc_4h=0.04,
            roc_24h=0.05,
            roc_1h_last=0.01,
            vol_ratio=3.0,
            rsi=65.0,
            atr=2.0,
            atr_pct=0.02,
            ema20=102.5,
            ema50=99.0,
            btc_roc_4h=0.0,
            funding=None,
            patterns=[p1h],
            cfg=cfg,
        )

        self.assertFalse(gates["S9c"].passed)
        self.assertEqual(gates["S9c"].value, 1.5)
        self.assertEqual(gates["S9c"].reason, "S9c_breakout_extension")

    def test_plan_primitives_prioritize_p2_and_keep_stop_feasibility(self) -> None:
        cfg = ScreenerConfig()
        frame = _frame_from_prices([100.0] * 30, lows=[99.0] * 27 + [97.0, 98.0, 99.0])
        patterns = [
            PatternHit(id="P1", side="long", boundary_price=99.0),
            PatternHit(id="P2", side="long", boundary_price=97.0),
            PatternHit(id="P3", side="long", boundary_price=99.0),
            PatternHit(id="P1H", side="long", boundary_price=99.0),
        ]
        plan = build_plan_primitives("long", patterns, frame, atr_value=2.0, cfg=cfg)
        self.assertEqual(plan.pattern_used, "P2")
        self.assertAlmostEqual(plan.invalidation_price, 96.0)
        self.assertAlmostEqual(plan.d_struct, 0.04)
        self.assertAlmostEqual(plan.d_atr, 0.04)
        self.assertAlmostEqual(plan.d_noise, 0.03)
        self.assertAlmostEqual(plan.d_final, 0.04)
        self.assertTrue(plan.stop_feasible)

    def test_plan_noise_floor_binds_when_atr_and_structure_are_tight(self) -> None:
        cfg = ScreenerConfig()
        frame = _frame_from_prices([100.0] * 30)
        patterns = [PatternHit(id="P1", side="long", boundary_price=99.8)]

        plan = build_plan_primitives("long", patterns, frame, atr_value=0.5, cfg=cfg)

        self.assertAlmostEqual(plan.d_atr, 0.01)
        self.assertAlmostEqual(plan.d_struct, (100.0 - (99.8 - 0.5 * 0.5)) / 100.0)
        self.assertAlmostEqual(plan.d_noise, 0.03)
        self.assertAlmostEqual(plan.d_final, 0.03)
        self.assertTrue(plan.stop_feasible)

    def test_chande_kroll_stop_hand_computed_fixture(self) -> None:
        highs = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        lows = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        closes = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]

        long_stop, short_stop = chande_kroll_stop(highs, lows, closes, p=3, x=1.0, q=2)

        self.assertIsNone(long_stop[3])
        self.assertIsNone(short_stop[3])
        self.assertAlmostEqual(long_stop[4], 12.5)
        self.assertAlmostEqual(long_stop[5], 13.5)
        self.assertAlmostEqual(short_stop[4], 11.5)
        self.assertAlmostEqual(short_stop[5], 12.5)

    def test_median_range_pct_fixture(self) -> None:
        self.assertAlmostEqual(median_range_pct([10.0, 11.0], [9.0, 9.0], [10.0, 10.0], 2), 0.15)
        self.assertAlmostEqual(median_range_pct([10.0, 11.0], [9.0, 9.0], [10.0, 10.0], 1), 0.2)
        self.assertIsNone(median_range_pct([], [], [], 5))

    def test_setup_digest_returns_facts_without_advisory_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cfg = ScreenerConfig(
                min_1h_bars=50,
                min_4h_bars=10,
                roc_4h_long=0.0,
                roc_24h_long=0.0,
                vol_ratio_min=0.1,
                rsi_long=(0.0, 100.0),
                atr_pct_min=0.0,
                atr_pct_max=1.0,
                last_hour_share_max=100.0,
                ext_atr_max=100.0,
                breakout_dist_atr_max=100.0,
            )
            closes = [100.0, 106.0, 109.0, 106.0, 100.0] + ([100.0, 102.0, 104.0, 102.0, 100.0, 98.0, 96.0, 98.0] * 5) + [100.0, 101.0, 102.0, 103.0, 105.0]
            candles = []
            for index, price in enumerate(closes):
                candles.append(candle(BTC, BASE_MS + index * ONE_HOUR, price, price + 0.8, price - 0.8, price, volume=10.0))
            cache.upsert_candles(candles)
            as_of = BASE_MS + len(closes) * ONE_HOUR

            digest = get_setup_digest(BTC, "long", as_of, cfg=cfg, cache_path=cache.path)
            opposite_digest = get_setup_digest(BTC, "short", as_of, cfg=cfg, cache_path=cache.path)

        self.assertTrue(digest["ok"])
        self.assertIn("support_levels", digest)
        self.assertIn("resistance_levels", digest)
        self.assertGreater(len(digest["support_levels"]), 0)
        self.assertGreater(len(digest["resistance_levels"]), 0)
        self.assertLessEqual(digest["support_levels"][0]["price"], digest["row"]["close"])
        self.assertGreaterEqual(digest["resistance_levels"][0]["price"], digest["row"]["close"])
        for level in [*digest["support_levels"], *digest["resistance_levels"]]:
            self.assertGreaterEqual(level["touches"], 1)
            self.assertIn(level["timeframe"], {"1h", "4h"})
            self.assertGreaterEqual(level["age_bars"], 0)
        self.assertIn("trigger_age_bars", digest)
        self.assertIn("retest_seen", digest)
        self.assertEqual(digest["row"]["candidate"], "long")
        self.assertNotIn("plan", digest["row"])
        self.assertNotIn("candidate_score", digest["row"])
        self.assertNotIn("marginal_score", digest["row"])
        self.assertNotIn("signal_marginal_score_before_state", digest["row"])
        for advisory_key in (
            "noise_floor_stop_pct",
            "structural_stop_pct",
            "recommended_stop_pct",
            "recommended_stop_feasible",
            "recommended_stop_reason",
            "stop_is_inside_noise",
            "tp_side_level",
            "tp_price_ref",
            "tp_room_pct",
            "tp_beyond_first_level",
            # derivable facts stripped 2026-07-07: the agent computes these itself from candles/indicators
            "recent_1h_csv",
            "range_20_high",
            "range_20_low",
            "range_48h_high",
            "range_48h_low",
            "last_3_high",
            "last_3_low",
            "median_1h_range_pct",
            "nearest_support",
            "nearest_resistance",
            "support_levels_1h",
            "resistance_levels_1h",
            "support_levels_4h",
            "resistance_levels_4h",
        ):
            self.assertNotIn(advisory_key, digest)
            self.assertNotIn(advisory_key, opposite_digest)
        self.assertIsNone(opposite_digest["trigger_age_bars"])
        self.assertIsNone(opposite_digest["retest_seen"])


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

    def test_state_store_updates_from_surfaced_candidate_only(self) -> None:
        store = ScreenerStateStore()
        store.update_from_scan_rows([{"symbol": BTC, "signal_candidate_before_state": "long", "candidate": None}], BASE_MS)
        self.assertEqual(store.last_candidate_ts, {})

        store.update_from_scan_rows([{"symbol": BTC, "signal_candidate_before_state": "long", "candidate": "long"}], BASE_MS + ONE_HOUR)
        self.assertEqual(store.last_candidate_ts[candidate_cooldown_key(BTC, "long")], BASE_MS + ONE_HOUR)
        self.assertEqual(store.last_candidate_quality[candidate_cooldown_key(BTC, "long")], "hard")
        self.assertNotIn(candidate_cooldown_key(BTC, "short"), store.last_candidate_ts)

        store.update_from_scan_rows([{"symbol": BTC, "candidate": "short", "candidate_quality": "marginal_extension"}], BASE_MS + 2 * ONE_HOUR)
        self.assertEqual(store.last_candidate_quality[candidate_cooldown_key(BTC, "short")], "marginal_extension")

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

        state = state_from_wallet_and_events(wallet, events, as_of, last_candidate_ts={candidate_cooldown_key(BTC, "long"): BASE_MS})

        self.assertEqual(state.trades_opened_today, 1)
        self.assertEqual(state.last_candidate_ts[candidate_cooldown_key(BTC, "long")], BASE_MS)
        self.assertEqual(state.last_stopout_ts[BTC], as_of - 50 * 60_000)
        self.assertEqual(state.last_stopout_ts["SOLUSDT"], as_of - 30 * 60_000)
        self.assertEqual([event.ts_ms for event in state.stopout_events], [as_of - 50 * 60_000, as_of - 30 * 60_000, as_of - 20 * 60_000])
        self.assertEqual([event.side for event in state.stopout_events], [None, None, None])
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
        self.assertEqual(state.stopout_events, [])

    def test_max_same_direction_is_side_specific_not_scan_global(self) -> None:
        cfg = ScreenerConfig(max_same_direction=2)
        state = TradingState(
            open_positions=[
                OpenPosition(symbol="ETHUSDT", side="long"),
                OpenPosition(symbol="SOLUSDT", side="long"),
            ]
        )

        self.assertNotIn("max_same_direction", scan_global_blocks(state, cfg, BASE_MS))
        long_blocks, _ = state_blocks("XRPUSDT", "long", state, BASE_MS, cfg)
        short_blocks, _ = state_blocks("XRPUSDT", "short", state, BASE_MS, cfg)

        self.assertIn("max_same_direction", long_blocks)
        self.assertNotIn("max_same_direction", short_blocks)

    def test_score_candidate_exact_margins(self) -> None:
        from traderbot_ai.screener.gates import GateResult

        cfg = ScreenerConfig()
        gates = {
            "S1": GateResult(passed=True, value=0.05, threshold=0.025),
            "S2": GateResult(passed=True, value=0.03, threshold=0.02),
            "S3": GateResult(passed=True, value=4.0, threshold=2.0),
            "S9a": GateResult(passed=True, value=0.2, threshold=0.45),
            "S9b": GateResult(passed=True, value=0.5, threshold=1.5),
            "S9c": GateResult(passed=True, value=0.4, threshold=0.8),
            "S10": GateResult(passed=True, value=0.1, threshold=0.35),
            "S11": GateResult(passed=True, value=1.0, threshold=3.0),
        }

        expected = 1.0 + 0.5 + 1.0 + (0.25 / 0.45) + (1.0 / 1.5) + 0.5 + (0.25 / 0.35) + (2.0 / 3.0)
        self.assertAlmostEqual(score_candidate(gates, "hard", cfg), expected)
        self.assertAlmostEqual(score_candidate(gates, "marginal_extension", cfg), expected - 1.0)

        capped = dict(gates)
        capped["S1"] = GateResult(passed=True, value=0.2, threshold=0.025)
        self.assertAlmostEqual(score_candidate(capped, "hard", cfg), expected - 1.0 + 2.0)

        sparse = {"S1": GateResult(passed=True, value=0.05, threshold=0.025), "S8": GateResult(passed=None, reason="funding_missing")}
        self.assertAlmostEqual(score_candidate(sparse, "hard", cfg), 1.0)

    def test_candidate_rank_cap_demotes_weakest(self) -> None:
        cfg = ScreenerConfig(max_candidates_per_scan=2)
        rows = [
            SymbolRow(symbol="AAAUSDT", status="ok", candidate="long", candidate_quality="hard", candidate_score=3.0),
            SymbolRow(symbol="BBBUSDT", status="ok", candidate="long", candidate_quality="hard", candidate_score=1.0),
            SymbolRow(symbol="CCCUSDT", status="ok", candidate="short", candidate_quality="marginal_extension", candidate_score=2.0, marginal_reasons=["S9b_extension=2.00<=marginal_2.50"]),
        ]

        _apply_candidate_rank_cap(rows, cfg)

        survivors = [row.symbol for row in rows if row.candidate in {"long", "short"}]
        self.assertEqual(survivors, ["AAAUSDT", "CCCUSDT"])
        demoted = rows[1]
        self.assertIsNone(demoted.candidate)
        self.assertIsNone(demoted.candidate_quality)
        self.assertIsNone(demoted.plan)
        self.assertIn("candidate_rank_cap", demoted.blocked_by)
        self.assertEqual(rows[2].marginal_reasons, ["S9b_extension=2.00<=marginal_2.50"])

    def test_sma_rolling_std_and_zscore_fixtures(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        sma_series = sma(values, 3)
        std_series = rolling_std(values, 3)

        self.assertIsNone(sma_series[1])
        self.assertAlmostEqual(sma_series[2], 2.0)
        self.assertAlmostEqual(sma_series[4], 4.0)
        self.assertAlmostEqual(std_series[4], (2.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(zscore_last([1.0, 1.0, 1.0, 1.0, 3.0], 5), 2.0)
        self.assertIsNone(zscore_last([2.0, 2.0, 2.0], 3))
        self.assertIsNone(zscore_last([1.0, 2.0], 5))

    def test_range_expansion_last_fixture(self) -> None:
        self.assertAlmostEqual(range_expansion_last([10.0, 12.0, 15.0], [9.0, 8.0, 10.0], 3), 0.875)
        self.assertAlmostEqual(range_expansion_last([10.0, 12.0, 15.0], [9.0, 8.0, 10.0], 2), 0.875)
        self.assertIsNone(range_expansion_last([10.0], [9.0], 3))
        self.assertIsNone(range_expansion_last([10.0, 11.0], [0.0, -1.0], 2))

    def test_s10_and_s11_gates_block_extended_moves(self) -> None:
        cfg = ScreenerConfig()

        def gates_for(side: str, range_expansion: float | None, zscore: float | None) -> dict:
            return evaluate_signal_gates(
                side,
                close=100.0,
                roc_4h=0.03,
                roc_24h=0.03 if side == "long" else -0.03,
                roc_1h_last=0.005,
                vol_ratio=3.0,
                rsi=60.0 if side == "long" else 40.0,
                atr=2.0,
                atr_pct=0.02,
                ema20=99.0,
                ema50=95.0 if side == "long" else 105.0,
                btc_roc_4h=0.0,
                funding=None,
                patterns=[],
                cfg=cfg,
                range_expansion=range_expansion,
                zscore=zscore,
            )

        blocked = gates_for("long", 0.5, None)
        self.assertFalse(blocked["S10"].passed)
        self.assertEqual(blocked["S10"].reason, "S10_range_expansion")
        self.assertTrue(gates_for("long", 0.3, None)["S10"].passed)
        self.assertTrue(gates_for("long", None, None)["S10"].passed)
        self.assertEqual(gates_for("long", None, None)["S10"].reason, "not_applicable")
        self.assertFalse(gates_for("short", 0.5, None)["S10"].passed)

        self.assertFalse(gates_for("long", None, 3.5)["S11"].passed)
        self.assertEqual(gates_for("long", None, 3.5)["S11"].reason, "S11_zscore")
        self.assertTrue(gates_for("long", None, 2.5)["S11"].passed)
        self.assertFalse(gates_for("short", None, -3.5)["S11"].passed)
        self.assertTrue(gates_for("short", None, 3.5)["S11"].passed)
        self.assertTrue(gates_for("long", None, None)["S11"].passed)

    def test_stoploss_guard_locks_side_within_window_and_expires(self) -> None:
        cfg = ScreenerConfig()
        as_of = BASE_MS + 48 * ONE_HOUR
        stopouts = [
            StopoutEvent(ts_ms=as_of - 5 * ONE_HOUR, side="long"),
            StopoutEvent(ts_ms=as_of - 3 * ONE_HOUR, side="long"),
            StopoutEvent(ts_ms=as_of - 2 * ONE_HOUR, side="long"),
        ]
        state = TradingState(stopout_events=stopouts)

        self.assertEqual(stoploss_guard_locked_sides(state, as_of, cfg), {"long"})
        long_blocks, _ = state_blocks("XRPUSDT", "long", state, as_of, cfg)
        short_blocks, _ = state_blocks("XRPUSDT", "short", state, as_of, cfg)
        self.assertIn("stoploss_guard_long", long_blocks)
        self.assertNotIn("stoploss_guard_short", short_blocks)
        self.assertNotIn("stoploss_guard", scan_global_blocks(state, cfg, as_of))

        expired_as_of = as_of - 2 * ONE_HOUR + cfg.stoploss_guard_stop_duration_ms
        self.assertEqual(stoploss_guard_locked_sides(state, expired_as_of, cfg), set())
        long_blocks_after, _ = state_blocks("XRPUSDT", "long", state, expired_as_of, cfg)
        self.assertNotIn("stoploss_guard_long", long_blocks_after)

    def test_stoploss_guard_ignores_stopouts_outside_lookback(self) -> None:
        cfg = ScreenerConfig()
        as_of = BASE_MS + 72 * ONE_HOUR
        stopouts = [
            StopoutEvent(ts_ms=as_of - cfg.stoploss_guard_lookback_ms - ONE_HOUR, side="long"),
            StopoutEvent(ts_ms=as_of - 3 * ONE_HOUR, side="long"),
            StopoutEvent(ts_ms=as_of - 2 * ONE_HOUR, side="long"),
        ]
        state = TradingState(stopout_events=stopouts)

        self.assertEqual(stoploss_guard_locked_sides(state, as_of, cfg), set())

    def test_stoploss_guard_unknown_side_counts_for_both_and_blocks_globally(self) -> None:
        cfg = ScreenerConfig()
        as_of = BASE_MS + 48 * ONE_HOUR
        stopouts = [
            StopoutEvent(ts_ms=as_of - 4 * ONE_HOUR),
            StopoutEvent(ts_ms=as_of - 3 * ONE_HOUR),
            StopoutEvent(ts_ms=as_of - 2 * ONE_HOUR),
        ]
        state = TradingState(stopout_events=stopouts)

        self.assertEqual(stoploss_guard_locked_sides(state, as_of, cfg), {"long", "short"})
        self.assertIn("stoploss_guard", scan_global_blocks(state, cfg, as_of))

    def test_stoploss_guard_global_mode_mixes_sides(self) -> None:
        cfg = ScreenerConfig(stoploss_guard_only_per_side=False)
        as_of = BASE_MS + 48 * ONE_HOUR
        stopouts = [
            StopoutEvent(ts_ms=as_of - 4 * ONE_HOUR, side="long"),
            StopoutEvent(ts_ms=as_of - 3 * ONE_HOUR, side="short"),
            StopoutEvent(ts_ms=as_of - 2 * ONE_HOUR, side="long"),
        ]
        state = TradingState(stopout_events=stopouts)

        self.assertEqual(stoploss_guard_locked_sides(state, as_of, cfg), {"long", "short"})
        per_side_cfg = ScreenerConfig(stoploss_guard_only_per_side=True)
        self.assertEqual(stoploss_guard_locked_sides(state, as_of, per_side_cfg), set())

    def test_state_reconstruction_extracts_stopout_sides(self) -> None:
        as_of = BASE_MS + 30 * ONE_HOUR
        wallet = {"totals": {"equity_usdt": 1000.0}, "open_positions": []}
        events = [
            {
                "type": "position_closed",
                "payload": {
                    "symbol": BTC,
                    "side": "Sell",
                    "exit_time_ms": as_of - ONE_HOUR,
                    "exit_reason": "StopLoss",
                    "realized_pnl_usdt": -10.0,
                },
            },
            {
                "type": "position_closed",
                "payload": {
                    "symbol": "ETHUSDT",
                    "side": "Buy",
                    "exit_time_ms": as_of - 30 * 60_000,
                    "exit_reason": "StopLoss",
                    "realized_pnl_usdt": -10.0,
                },
            },
        ]

        state = state_from_wallet_and_events(wallet, events, as_of)

        self.assertEqual([event.side for event in state.stopout_events], ["short", "long"])

    def test_state_cooldown_boundaries_are_strictly_less_than_duration(self) -> None:
        cfg = ScreenerConfig()
        state = TradingState(last_candidate_ts={candidate_cooldown_key(BTC, "long"): BASE_MS}, last_stopout_ts={BTC: BASE_MS})

        before_candidate, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_candidate_ms - 1, cfg)
        at_candidate, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_candidate_ms, cfg)
        before_stopout, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_stopout_ms - 1, cfg)
        at_stopout, _ = state_blocks(BTC, "long", state, BASE_MS + cfg.cooldown_stopout_ms, cfg)

        self.assertIn("cooldown_candidate", before_candidate)
        self.assertNotIn("cooldown_candidate", at_candidate)
        self.assertIn("cooldown_stopout", before_stopout)
        self.assertNotIn("cooldown_stopout", at_stopout)

    def test_candidate_cooldown_is_side_specific(self) -> None:
        cfg = ScreenerConfig()
        state = TradingState(last_candidate_ts={candidate_cooldown_key(BTC, "long"): BASE_MS})

        long_blocks, _ = state_blocks(BTC, "long", state, BASE_MS + ONE_HOUR, cfg)
        short_blocks, _ = state_blocks(BTC, "short", state, BASE_MS + ONE_HOUR, cfg)

        self.assertIn("cooldown_candidate", long_blocks)
        self.assertNotIn("cooldown_candidate", short_blocks)

    def test_hard_signal_can_override_marginal_candidate_cooldown(self) -> None:
        cfg = ScreenerConfig()
        key = candidate_cooldown_key(BTC, "long")
        state = TradingState(last_candidate_ts={key: BASE_MS}, last_candidate_quality={key: "marginal_extension"})

        hard_blocks, _ = state_blocks(BTC, "long", state, BASE_MS + ONE_HOUR, cfg, signal_quality="hard")
        marginal_blocks, _ = state_blocks(BTC, "long", state, BASE_MS + ONE_HOUR, cfg, signal_quality="marginal_extension")

        self.assertNotIn("cooldown_candidate", hard_blocks)
        self.assertIn("cooldown_candidate", marginal_blocks)

    def test_legacy_symbol_only_candidate_cooldown_still_blocks_during_migration(self) -> None:
        cfg = ScreenerConfig()
        state = TradingState(last_candidate_ts={BTC: BASE_MS})

        long_blocks, _ = state_blocks(BTC, "long", state, BASE_MS + ONE_HOUR, cfg)
        short_blocks, _ = state_blocks(BTC, "short", state, BASE_MS + ONE_HOUR, cfg)

        self.assertIn("cooldown_candidate", long_blocks)
        self.assertIn("cooldown_candidate", short_blocks)

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
