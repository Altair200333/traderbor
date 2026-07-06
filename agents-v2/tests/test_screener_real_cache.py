from __future__ import annotations

import sqlite3
import tempfile
import unittest
from collections import Counter
from contextlib import closing
from pathlib import Path

from traderbot_ai.screener import ScreenerStateStore, scan
from traderbot_ai.screener.render import to_canonical_json, to_markdown_table
from traderbot_ai.screener.market import parse_time_ms
from traderbot_ai.screener.state import TradingState
from traderbot_ai.simulator.market_cache import Candle, DEFAULT_CACHE_PATH, LocalMarketCache


BROAD_CACHE_PATH = DEFAULT_CACHE_PATH.parent / "broad-2w-20260706" / "market_cache.sqlite3"
UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "ADAUSDT",
    "LTCUSDT",
    "BNBUSDT",
    "ZECUSDT",
    "SUIUSDT",
    "PEPEUSDT",
]
MISSING_SYMBOL = "HYPEUSDT"
AVAX_CANDIDATE_AS_OF = 1_782_496_800_000
ONE_HOUR_MS = 60 * 60_000
FOUR_HOURS_MS = 4 * ONE_HOUR_MS
MONTH_START_MS = parse_time_ms("2026-05-30T00:00:00Z")
MONTH_END_MS = parse_time_ms("2026-07-05T16:00:00Z")


class ScreenerRealCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache_path = BROAD_CACHE_PATH if BROAD_CACHE_PATH.exists() else DEFAULT_CACHE_PATH
        if not self.cache_path.exists():
            self.skipTest(f"real market cache is missing: {self.cache_path}")
        missing = sorted(set(UNIVERSE) - _symbols_with_1h(self.cache_path))
        if missing:
            self.skipTest(f"real market cache is missing 1h symbols: {', '.join(missing)}")

    def test_real_cache_matrix_has_known_positive_and_negative_cases(self) -> None:
        as_of_values = [AVAX_CANDIDATE_AS_OF - index * FOUR_HOURS_MS for index in reversed(range(20))]
        checked_rows = 0
        candidate_hits = []

        for as_of_ms in as_of_values:
            result = scan(UNIVERSE, as_of_ms=as_of_ms, cache_path=self.cache_path)
            self.assertEqual(len(result.symbols), len(UNIVERSE))
            for row in result.symbols:
                checked_rows += 1
                self.assertEqual(row.status, "ok", f"{as_of_ms} {row.symbol} {row.data_issue}")
                if row.candidate in {"long", "short"}:
                    candidate_hits.append((as_of_ms, row.symbol, row.candidate, row.plan.pattern_used if row.plan else None))

        self.assertGreaterEqual(checked_rows, 200)
        self.assertIn((AVAX_CANDIDATE_AS_OF, "AVAXUSDT", "long", "P1"), candidate_hits)

        result = scan(UNIVERSE, as_of_ms=AVAX_CANDIDATE_AS_OF, cache_path=self.cache_path)
        rows = {row.symbol: row for row in result.symbols}
        self.assertEqual(rows["AVAXUSDT"].candidate, "long")
        self.assertEqual(rows["AVAXUSDT"].plan.pattern_used, "P1")
        self.assertEqual(rows["AVAXUSDT"].failed_gates, [])
        self.assertIsNone(rows["SOLUSDT"].candidate)
        self.assertIn("S3", rows["SOLUSDT"].failed_gates)
        self.assertIn("S9b", rows["SOLUSDT"].failed_gates)
        self.assertIn("PAT", rows["SOLUSDT"].failed_gates)
        self.assertIsNone(rows["BTCUSDT"].candidate)
        self.assertIn("S1", rows["BTCUSDT"].failed_gates)
        self.assertIn("S2", rows["BTCUSDT"].failed_gates)

    def test_real_cache_scan_is_stable_when_future_and_partial_bars_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "market.sqlite3"
            _copy_symbol_interval(self.cache_path, cache_path, ["BTCUSDT", "AVAXUSDT"], "1h", AVAX_CANDIDATE_AS_OF + 6 * ONE_HOUR_MS)
            cache = LocalMarketCache(cache_path)
            before = scan(["AVAXUSDT"], as_of_ms=AVAX_CANDIDATE_AS_OF + 30 * 60_000, cache_path=cache_path)
            cache.upsert_candles(
                [
                    Candle(
                        symbol="AVAXUSDT",
                        interval="1h",
                        open_time=AVAX_CANDIDATE_AS_OF,
                        close_time=AVAX_CANDIDATE_AS_OF + 30 * 60_000 - 1,
                        open=999.0,
                        high=1000.0,
                        low=998.0,
                        close=999.5,
                        volume=999999.0,
                    ),
                    Candle(
                        symbol="AVAXUSDT",
                        interval="1h",
                        open_time=AVAX_CANDIDATE_AS_OF + 5 * ONE_HOUR_MS,
                        close_time=AVAX_CANDIDATE_AS_OF + 6 * ONE_HOUR_MS - 1,
                        open=777.0,
                        high=778.0,
                        low=776.0,
                        close=777.5,
                        volume=777777.0,
                    ),
                ]
            )
            after = scan(["AVAXUSDT"], as_of_ms=AVAX_CANDIDATE_AS_OF + 30 * 60_000, cache_path=cache_path)

        self.assertEqual(to_canonical_json(before), to_canonical_json(after))

    def test_missing_hype_is_explicit_insufficient_data_case(self) -> None:
        result = scan([MISSING_SYMBOL, "BTCUSDT"], as_of_ms=AVAX_CANDIDATE_AS_OF, cache_path=self.cache_path)
        rows = {row.symbol: row for row in result.symbols}

        self.assertEqual(rows[MISSING_SYMBOL].status, "insufficient_data")
        self.assertIn(f"{MISSING_SYMBOL}:insufficient_data", result.data_warnings)

    def test_real_sui_spike_is_rejected_by_anti_chase(self) -> None:
        result = scan(["SUIUSDT", "BTCUSDT"], as_of_ms=parse_time_ms("2026-06-22T12:00:00Z"), cache_path=self.cache_path)
        row = {item.symbol: item for item in result.symbols}["SUIUSDT"]

        self.assertIsNone(row.candidate)
        self.assertGreater(row.roc_4h, 0.025)
        self.assertGreater(row.vol_ratio, 2.0)
        self.assertIn("P1", row.patterns_long)
        self.assertIn("S9a", row.failed_gates)
        self.assertIn("S9b", row.failed_gates)
        self.assertIn("S9c", row.failed_gates)

    def test_broad_month_cache_matrix_has_expected_candidates_and_no_data_leaks(self) -> None:
        if self.cache_path != BROAD_CACHE_PATH:
            self.skipTest("broad month cache is required for this regression")

        symbols = [*UNIVERSE, MISSING_SYMBOL]
        state_store = ScreenerStateStore()
        statuses: Counter[str] = Counter()
        warnings: Counter[str] = Counter()
        candidate_hits = []
        checked_rows = 0
        max_markdown_chars = 0

        as_of_ms = MONTH_START_MS
        while as_of_ms < MONTH_END_MS:
            state = TradingState(last_candidate_ts=dict(state_store.last_candidate_ts))
            result = scan(symbols, as_of_ms=as_of_ms, cache_path=self.cache_path, state=state)
            state_store.update_from_scan_rows([row.model_dump(mode="json") for row in result.symbols], as_of_ms)

            self.assertEqual(len(result.symbols), len(symbols))
            max_markdown_chars = max(max_markdown_chars, len(to_markdown_table(result)))
            warnings.update(result.data_warnings)
            for row in result.symbols:
                checked_rows += 1
                statuses[row.status] += 1
                if row.symbol != MISSING_SYMBOL:
                    self.assertEqual(row.status, "ok", f"{as_of_ms} {row.symbol} {row.data_issue}")
                if row.candidate in {"long", "short"}:
                    candidate_hits.append((as_of_ms, row.symbol, row.candidate, row.plan.pattern_used if row.plan else None))
            as_of_ms += FOUR_HOURS_MS

        self.assertEqual(checked_rows, 220 * len(symbols))
        self.assertEqual(statuses["ok"], 220 * len(UNIVERSE))
        self.assertEqual(statuses["insufficient_data"], 220)
        self.assertEqual(warnings[f"{MISSING_SYMBOL}:insufficient_data"], 220)
        self.assertLess(max_markdown_chars, 2_000)
        self.assertEqual(
            candidate_hits,
            [
                (parse_time_ms("2026-06-10T16:00:00Z"), "BTCUSDT", "long", "P1"),
                (parse_time_ms("2026-07-01T16:00:00Z"), "DOGEUSDT", "long", "P1"),
            ],
        )


def _symbols_with_1h(cache_path: Path) -> set[str]:
    with closing(sqlite3.connect(cache_path)) as conn:
        rows = conn.execute("select distinct symbol from candles where interval = '1h'").fetchall()
    return {str(row[0]) for row in rows}


def _copy_symbol_interval(source: Path, destination: Path, symbols: list[str], interval: str, through_open_time: int) -> None:
    destination_cache = LocalMarketCache(destination)
    with closing(sqlite3.connect(source)) as conn:
        placeholders = ",".join("?" for _ in symbols)
        rows = conn.execute(
            f"""
            select symbol, interval, open_time, close_time, open, high, low, close, volume
            from candles
            where interval = ?
              and symbol in ({placeholders})
              and open_time <= ?
            order by symbol, open_time
            """,
            [interval, *symbols, through_open_time],
        ).fetchall()
    destination_cache.upsert_candles(
        [
            Candle(
                symbol=str(row[0]),
                interval=str(row[1]),
                open_time=int(row[2]),
                close_time=int(row[3]),
                open=float(row[4]),
                high=float(row[5]),
                low=float(row[6]),
                close=float(row[7]),
                volume=float(row[8]),
            )
            for row in rows
        ]
    )


if __name__ == "__main__":
    unittest.main()
