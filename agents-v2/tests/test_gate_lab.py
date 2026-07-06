from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from traderbot_ai.screener.matrix import candidate_outcome_report, retest_sweep, stop_sweep
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache


BTC = "BTCUSDT"
BASE_MS = 1_704_067_200_000
ONE_MIN = 60_000


def _minute_candle(k: int, open_price: float, high: float, low: float, close: float, symbol: str = BTC) -> Candle:
    open_time = BASE_MS + k * ONE_MIN
    return Candle(
        symbol=symbol,
        interval="1m",
        open_time=open_time,
        close_time=open_time + ONE_MIN - 1,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class CandidateOutcomeReportTests(unittest.TestCase):
    def test_report_aggregates_by_tag_with_exact_counts(self) -> None:
        summary = {
            "candidates": [
                {
                    "as_of_ms": BASE_MS,
                    "symbol": "AAAUSDT",
                    "side": "long",
                    "quality": "hard",
                    "pattern": "P1",
                    "failed_gates": [],
                    "forward": {"tp_sl_first": "tp", "first_time_ms": BASE_MS + 120 * ONE_MIN},
                },
                {
                    "as_of_ms": BASE_MS,
                    "symbol": "BBBUSDT",
                    "side": "long",
                    "quality": "marginal_extension",
                    "pattern": "P1",
                    "failed_gates": ["S9b"],
                    "forward": {"tp_sl_first": "sl", "first_time_ms": BASE_MS + 60 * ONE_MIN},
                },
                {
                    "as_of_ms": BASE_MS,
                    "symbol": "CCCUSDT",
                    "side": "short",
                    "quality": "hard",
                    "pattern": "P3",
                    "failed_gates": [],
                    "forward": {"tp_sl_first": "ambiguous", "first_time_ms": BASE_MS + 10 * ONE_MIN},
                },
            ]
        }

        report = candidate_outcome_report(summary)

        self.assertEqual(report["total"]["n"], 3)
        self.assertEqual(report["total"]["tp"], 1)
        self.assertEqual(report["total"]["sl"], 1)
        self.assertEqual(report["total"]["ambiguous"], 1)
        self.assertAlmostEqual(report["total"]["tp_rate_resolved"], 0.5)
        self.assertAlmostEqual(report["total"]["median_time_to_exit_min"], 60.0)
        self.assertAlmostEqual(report["quality:hard"]["tp_rate_resolved"], 1.0)
        self.assertAlmostEqual(report["quality:marginal_extension"]["tp_rate_resolved"], 0.0)
        self.assertEqual(report["gates:S9b"]["sl"], 1)
        self.assertEqual(report["pattern:P1"]["n"], 2)
        self.assertEqual(report["side:short"]["ambiguous"], 1)
        self.assertIsNone(report["side:short"]["tp_rate_resolved"])


class StopSweepTests(unittest.TestCase):
    def test_stop_sweep_relabels_candidates_across_engines(self) -> None:
        candles = []
        for k in range(0, 30):
            candles.append(_minute_candle(k, 100.0, 100.2, 99.5, 100.0))
        candles.append(_minute_candle(30, 100.0, 100.0, 98.6, 99.0))
        for k in range(31, 120):
            candles.append(_minute_candle(k, 100.0, 100.5, 99.0, 100.0))
        candles.append(_minute_candle(120, 100.0, 103.0, 99.5, 103.0))
        for k in range(121, 180):
            candles.append(_minute_candle(k, 103.0, 103.5, 102.5, 103.0))
        candles.append(_minute_candle(180, 103.0, 104.0, 102.9, 104.0))
        for k in range(181, 240):
            candles.append(_minute_candle(k, 104.0, 104.5, 103.5, 104.0))
        candles.append(_minute_candle(240, 104.0, 106.0, 103.9, 106.0))
        for k in range(241, 300):
            candles.append(_minute_candle(k, 106.0, 106.2, 105.5, 106.0))

        summary = {
            "candidates": [
                {
                    "as_of_ms": BASE_MS,
                    "symbol": BTC,
                    "side": "long",
                    "quality": "hard",
                    "pattern": "P1",
                    "failed_gates": [],
                    "plan": {"ref_entry": 100.0, "d_final": 0.02, "tp_rr_default": 2.0, "d_cksp": 0.03},
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(candles)
            report = stop_sweep(summary, cache.path, horizon_hours=24)

        # stop 1.0% -> the 98.6 dip stops it out; stop 1.5% survives the dip and takes profit at 103.
        self.assertEqual(report["0.5x_d_final"]["sl"], 1)
        self.assertAlmostEqual(report["0.5x_d_final"]["mean_payoff_r"], -1.0)
        self.assertEqual(report["0.75x_d_final"]["tp"], 1)
        self.assertAlmostEqual(report["0.75x_d_final"]["mean_payoff_r"], 2.0)
        self.assertAlmostEqual(report["0.75x_d_final"]["median_time_to_exit_min"], 120.0)
        self.assertEqual(report["1x_d_final"]["tp"], 1)
        self.assertAlmostEqual(report["1x_d_final"]["median_time_to_exit_min"], 180.0)
        self.assertEqual(report["1.5x_d_final"]["tp"], 1)
        self.assertEqual(report["cksp"]["tp"], 1)
        self.assertAlmostEqual(report["cksp"]["median_time_to_exit_min"], 240.0)
        self.assertEqual(report["2x_d_final"]["none"], 1)
        self.assertIsNone(report["2x_d_final"]["tp_rate_resolved"])

    def test_stop_sweep_counts_missing_cksp_as_no_stop(self) -> None:
        summary = {
            "candidates": [
                {
                    "as_of_ms": BASE_MS,
                    "symbol": BTC,
                    "side": "long",
                    "quality": "hard",
                    "pattern": "P1",
                    "failed_gates": [],
                    "plan": {"ref_entry": 100.0, "d_final": 0.02, "tp_rr_default": 2.0, "d_cksp": None},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            report = stop_sweep(summary, cache.path, horizon_hours=1)

        self.assertEqual(report["cksp"]["no_stop"], 1)
        self.assertEqual(report["0.5x_d_final"]["missing_1m"], 1)


class RetestSweepTests(unittest.TestCase):
    """Five hand-computed 1m paths, entry=100, d_final=0.02 (stop 98, take 104), tp_rr=2.0.

    Limit prices: p=0.25 -> 99.5, p=0.4 -> 99.2.
    AAA pullback-then-TP: p25 fills minute 11, p40 fills minute 21, TP touch minute 61.
    BBB straight to TP at minute 31, never pulls back -> both p missed.
    CCC straight through limit to stop at minute 6 -> filled loser for both p.
    DDD pullback only at minute 130, no resolution, closes at 100 -> fill only at ttl>=130.
    EEE TP at minute 20, pullback to 99.4 at minute 40 -> p25 late_fill (0), p40 never fills.
    """

    @classmethod
    def setUpClass(cls) -> None:
        def flat(symbol: str, ks: range) -> list[Candle]:
            return [_minute_candle(k, 100.0, 100.2, 99.9, 100.0, symbol=symbol) for k in ks]

        candles: list[Candle] = []
        candles += flat("AAAUSDT", range(0, 10))
        candles.append(_minute_candle(10, 100.0, 100.1, 99.4, 100.0, symbol="AAAUSDT"))
        candles += flat("AAAUSDT", range(11, 20))
        candles.append(_minute_candle(20, 100.0, 100.1, 99.1, 100.0, symbol="AAAUSDT"))
        candles += flat("AAAUSDT", range(21, 60))
        candles.append(_minute_candle(60, 100.0, 104.5, 99.9, 104.0, symbol="AAAUSDT"))
        candles += flat("AAAUSDT", range(61, 240))

        candles += flat("BBBUSDT", range(0, 30))
        candles.append(_minute_candle(30, 100.0, 104.2, 99.9, 104.0, symbol="BBBUSDT"))
        candles += flat("BBBUSDT", range(31, 240))

        candles += flat("CCCUSDT", range(0, 5))
        candles.append(_minute_candle(5, 100.0, 100.1, 97.9, 98.5, symbol="CCCUSDT"))
        candles += [_minute_candle(k, 99.0, 99.2, 98.8, 99.0, symbol="CCCUSDT") for k in range(6, 240)]

        candles += flat("DDDUSDT", range(0, 129))
        candles.append(_minute_candle(129, 100.0, 100.1, 99.15, 100.0, symbol="DDDUSDT"))
        candles += flat("DDDUSDT", range(130, 240))

        candles += flat("EEEUSDT", range(0, 19))
        candles.append(_minute_candle(19, 100.0, 104.1, 99.9, 104.0, symbol="EEEUSDT"))
        candles += flat("EEEUSDT", range(20, 39))
        candles.append(_minute_candle(39, 100.0, 100.1, 99.4, 100.0, symbol="EEEUSDT"))
        candles += flat("EEEUSDT", range(40, 240))

        cls.tmp = tempfile.TemporaryDirectory()
        cache = LocalMarketCache(Path(cls.tmp.name) / "market.sqlite3")
        cache.upsert_candles(candles)
        plan = {"ref_entry": 100.0, "d_final": 0.02, "tp_rr_default": 2.0}
        summary = {
            "candidates": [
                {"as_of_ms": BASE_MS, "symbol": symbol, "side": "long", "quality": "hard", "pattern": "P1", "failed_gates": [], "plan": dict(plan)}
                for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")
            ]
        }
        cls.report = retest_sweep(summary, cache.path, pullbacks=(0.25, 0.4), ttls_min=(60, 240), horizon_hours=4)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_base_total_matches_hand_computation(self) -> None:
        # AAA tp +2, BBB tp +2, CCC sl -1, DDD none 0, EEE tp +2
        self.assertEqual(self.report["n"], 5)
        self.assertEqual(self.report["skipped_no_data"], 0)
        self.assertAlmostEqual(self.report["base_total_r"], 5.0)

    def test_p025_ttl60_cell(self) -> None:
        cell = self.report["cells"]["p0.25_ttl60m"]
        # AAA caught (+2.25), BBB missed, CCC filled loser (-0.75), DDD skipped none, EEE late fill
        self.assertEqual(cell["filled"], 2)
        self.assertEqual(cell["tp_caught"], 1)
        self.assertEqual(cell["tp_missed"], 1)
        self.assertEqual(cell["late_fill"], 1)
        self.assertEqual(cell["sl_avoided"], 0)
        self.assertEqual(cell["none_skipped"], 1)
        self.assertAlmostEqual(cell["total_r"], 1.5)
        self.assertAlmostEqual(cell["delta_r"], -3.5)

    def test_p025_ttl240_fills_the_late_pullback(self) -> None:
        cell = self.report["cells"]["p0.25_ttl240m"]
        # DDD now fills at minute 130: unresolved mark-to-horizon 0 + 0.25 head start
        self.assertEqual(cell["filled"], 3)
        self.assertAlmostEqual(cell["total_r"], 1.75)

    def test_p04_ttl60_cell(self) -> None:
        cell = self.report["cells"]["p0.4_ttl60m"]
        # AAA caught (+2.4), CCC filled loser (-0.6), BBB and EEE missed (99.4 never reaches 99.2)
        self.assertEqual(cell["filled"], 2)
        self.assertEqual(cell["tp_caught"], 1)
        self.assertEqual(cell["tp_missed"], 2)
        self.assertEqual(cell["late_fill"], 0)
        self.assertAlmostEqual(cell["total_r"], 1.8)

    def test_splits_cover_months_and_buckets(self) -> None:
        month_key = [key for key in self.report["splits"] if key.startswith("month:")]
        self.assertEqual(len(month_key), 1)
        self.assertEqual(self.report["splits"][month_key[0]]["n"], 5)
        self.assertEqual(self.report["splits"]["bucket:rest"]["n"], 5)
        self.assertEqual(self.report["splits"]["bucket:concentrated"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
