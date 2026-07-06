from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from traderbot_ai.paths import AGENTS_V2_ROOT
from traderbot_ai.screener.indicators import atr_wilder, ema, roc, rsi_wilder
from traderbot_ai.simulator.clock import (
    clear_file_simulation_clock_state,
    clear_process_simulation_clock_state,
    file_simulation_clock_ms,
    process_simulation_clock_ms,
    set_file_simulation_clock_state,
    set_process_simulation_clock_state,
    set_simulation_clock_state,
)
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache
from traderbot_ai.tools.analysis import compute_indicators_impl, run_analysis_code_impl


BTC = "BTCUSDT"
ETH = "ETHUSDT"
BASE_MS = 1_704_067_200_000
ONE_HOUR_MS = 60 * 60_000
FOUR_HOURS_MS = 4 * ONE_HOUR_MS


class AnalysisToolTests(unittest.TestCase):
    def test_compute_indicators_matches_core_screener_math_and_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            candles = _trend_candles(BTC, "1h", BASE_MS, ONE_HOUR_MS, 200)
            cache.upsert_candles(candles)
            as_of = candles[-1].close_time
            with _cache_env(cache.path):
                result = compute_indicators_impl(BTC, "1h", as_of=as_of, limit=170, indicators="strategy", source="cache", tail=3)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["source"], "local_cache")
        self.assertEqual(result["symbol"], BTC)
        self.assertEqual(result["candle_summary"]["count"], 170)
        latest = result["latest"]
        loaded = candles[-170:]
        closes = [candle.close for candle in loaded]
        highs = [candle.high for candle in loaded]
        lows = [candle.low for candle in loaded]

        self.assertAlmostEqual(latest["sma_20"], sum(closes[-20:]) / 20)
        self.assertAlmostEqual(latest["roc_4"], roc(closes, 4)[-1])
        self.assertAlmostEqual(latest["ema_20"], ema(closes, 20)[-1])
        self.assertAlmostEqual(latest["rsi_14"], rsi_wilder(closes, 14)[-1])
        self.assertAlmostEqual(latest["atr_14"], atr_wilder(highs, lows, closes, 14)[-1])
        self.assertGreater(latest["plus_di_14"], latest["minus_di_14"])
        self.assertGreater(latest["adx_14"], 90.0)

        artifact_path = AGENTS_V2_ROOT / result["artifact_path"]
        self.assertTrue(artifact_path.exists())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["symbol"], BTC)
        self.assertEqual(len(artifact["rows"]), 170)
        self.assertEqual(len(result["tail"]), 3)
        json.dumps(result, allow_nan=False)

    def test_compute_indicators_handles_flat_zero_volume_data_and_renders_chart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            candles = [
                _candle(BTC, "1h", BASE_MS + index * ONE_HOUR_MS, ONE_HOUR_MS, 100.0, 100.0, 100.0, 100.0, 0.0)
                for index in range(80)
            ]
            cache.upsert_candles(candles)
            with _cache_env(cache.path):
                result = compute_indicators_impl(BTC, "1h", as_of=candles[-1].close_time, limit=80, indicators="all", source="cache", tail=2, render_chart=True)

        self.assertTrue(result["ok"], result)
        self.assertIsNone(result["latest"]["bb_percent_b_20_2"])
        self.assertIsNone(result["latest"]["stoch_k_14"])
        self.assertIsNone(result["latest"]["willr_14"])
        self.assertIsNone(result["latest"]["bop"])
        json.dumps(result, allow_nan=False)
        chart_path = AGENTS_V2_ROOT / result["chart_path"]
        self.assertTrue(chart_path.exists())
        self.assertGreater(chart_path.stat().st_size, 0)

    def test_deterministic_analysis_tools_share_candidate_anchor_and_future_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            btc_candles = _trend_candles(BTC, "1h", BASE_MS, ONE_HOUR_MS, 10)
            eth_candles = [_candle(ETH, "4h", btc_candles[-1].close_time, FOUR_HOURS_MS, 200.0, 201.0, 199.0, 200.0, 1000.0)]
            cache.upsert_candles([*btc_candles, *eth_candles])
            as_of = btc_candles[-1].close_time
            with _deterministic_env(cache.path, as_of, [{"symbol": BTC, "side": "long"}]):
                candidate = compute_indicators_impl(BTC, "1h", as_of=as_of, limit=10, indicators="sma", source="auto")
                anchor = compute_indicators_impl(ETH, "4h", as_of=as_of, limit=1, indicators="price", source="auto")
                wrong_symbol = compute_indicators_impl("XRPUSDT", "1h", as_of=as_of, limit=1, indicators="sma", source="auto")
                wrong_interval = compute_indicators_impl(BTC, "1m", as_of=as_of, limit=1, indicators="sma", source="auto")
                too_many = compute_indicators_impl(BTC, "1h", as_of=as_of, limit=171, indicators="sma", source="auto")
                missing_as_of = compute_indicators_impl(BTC, "1h", limit=1, indicators="sma", source="auto")
                stale = compute_indicators_impl(BTC, "1h", as_of=as_of - ONE_HOUR_MS, limit=1, indicators="sma", source="auto")
                future = compute_indicators_impl(BTC, "1h", as_of=as_of + 1, limit=1, indicators="sma", source="auto")
                live_bypass = compute_indicators_impl("XRPUSDT", "1m", as_of=as_of + 1, limit=999, indicators="sma", source="live")
                future_code = run_analysis_code_impl("result = len(df)", BTC, "1h", as_of=as_of + 1, limit=1, source="auto")
                live_code_bypass = run_analysis_code_impl("result = len(df)", "XRPUSDT", "1m", as_of=as_of + 1, limit=999, source="live")

        self.assertTrue(candidate["ok"], candidate)
        self.assertTrue(anchor["ok"], anchor)
        self.assertFalse(wrong_symbol["ok"])
        self.assertIn("runner candidates and BTC/ETH/SOL", wrong_symbol["error"])
        self.assertFalse(wrong_interval["ok"])
        self.assertIn("only allows 1h or 4h", wrong_interval["error"])
        self.assertFalse(too_many["ok"])
        self.assertIn("limit for 1h must be <= 170", too_many["error"])
        self.assertFalse(missing_as_of["ok"])
        self.assertIn("requires exact as_of", missing_as_of["error"])
        self.assertFalse(stale["ok"])
        self.assertIn("requires exact as_of equal to simulation clock", stale["error"])
        self.assertFalse(future["ok"])
        self.assertIn("exceeds simulation clock", future["error"])
        self.assertFalse(live_bypass["ok"])
        self.assertIn("source=live is not allowed", live_bypass["error"])
        self.assertFalse(future_code["ok"])
        self.assertIn("exceeds simulation clock", future_code["error"])
        self.assertFalse(live_code_bypass["ok"])
        self.assertIn("source=live is not allowed", live_code_bypass["error"])

    def test_compute_indicators_aggregates_4h_from_cached_1h_when_4h_rows_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            candles = [
                Candle(
                    symbol=BTC,
                    interval="1h",
                    open_time=BASE_MS + index * ONE_HOUR_MS,
                    close_time=BASE_MS + (index + 1) * ONE_HOUR_MS - 1,
                    open=100.0 + index,
                    high=101.0 + index,
                    low=99.0 + index,
                    close=100.5 + index,
                    volume=10.0 + index,
                )
                for index in range(14)
            ]
            cache.upsert_candles(candles)
            as_of = BASE_MS + 14 * ONE_HOUR_MS
            with _deterministic_env(cache.path, as_of, [{"symbol": BTC, "side": "long"}]):
                result = compute_indicators_impl(BTC, "4h", as_of=as_of, limit=3, indicators="price", source="auto", tail=3)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["interval"], "4h")
        self.assertEqual(result["candle_summary"]["count"], 3)
        self.assertEqual(result["candle_summary"]["last_timestamp"], BASE_MS + 8 * ONE_HOUR_MS)
        self.assertEqual(result["latest"]["c"], 111.5)
        self.assertIn("4h candles aggregated from cached 1h candles", result["warnings"])

    def test_compute_indicators_accepts_utf8_bom_file_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = LocalMarketCache(tmp_path / "market.sqlite3")
            candles = _trend_candles(BTC, "1h", BASE_MS, ONE_HOUR_MS, 10)
            cache.upsert_candles(candles)
            as_of = candles[-1].close_time
            clock_path = tmp_path / "clock.json"
            clock_path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"as_of_ms": as_of}).encode("utf-8"))
            old_clock_path = os.environ.get("TRADERBOT_SIMULATION_CLOCK_PATH")
            old_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
            old_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
            old_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
            previous_process_clock = process_simulation_clock_ms()
            previous_file_clock = file_simulation_clock_ms()
            try:
                clear_process_simulation_clock_state()
                os.environ["TRADERBOT_SIMULATION_CLOCK_PATH"] = str(clock_path)
                os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(cache.path)
                os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
                os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps([{"symbol": BTC, "side": "long"}])
                result = compute_indicators_impl(BTC, "1h", as_of=as_of, limit=10, indicators="sma", source="auto")
            finally:
                _restore_env("TRADERBOT_SIMULATION_CLOCK_PATH", old_clock_path)
                _restore_env("TRADERBOT_MARKET_CACHE_PATH", old_cache)
                _restore_env("TRADERBOT_SCREENER_MODE", old_mode)
                _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", old_candidates)
                if previous_file_clock is None:
                    clear_file_simulation_clock_state()
                else:
                    set_file_simulation_clock_state(previous_file_clock)
                if previous_process_clock is None:
                    clear_process_simulation_clock_state()
                else:
                    set_process_simulation_clock_state(previous_process_clock)

        self.assertTrue(result["ok"], result)

    def test_run_analysis_code_receives_guarded_dataframe_and_can_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            candles = _trend_candles(BTC, "1h", BASE_MS, ONE_HOUR_MS, 30)
            cache.upsert_candles(candles)
            code = """
import os
note = artifact_dir / "note.txt"
note.write_text("ok", encoding="utf-8")
result = {
    "rows": len(df),
    "last_close": float(df["c"].iloc[-1]),
    "close_volume_corr": float(df["c"].corr(df["v"])),
    "cache_env_visible": "TRADERBOT_MARKET_CACHE_PATH" in os.environ,
    "openai_env_visible": "OPENAI_API_KEY" in os.environ,
    "pythonpath_visible": "PYTHONPATH" in os.environ,
    "note_exists": note.exists(),
}
"""
            with _cache_env(cache.path):
                result = run_analysis_code_impl(code, BTC, "1h", as_of=candles[-1].close_time, limit=20, source="cache", timeout_seconds=10)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"]["rows"], 20)
        self.assertEqual(result["result"]["last_close"], candles[-1].close)
        self.assertTrue(math.isfinite(result["result"]["close_volume_corr"]))
        self.assertFalse(result["result"]["cache_env_visible"])
        self.assertFalse(result["result"]["openai_env_visible"])
        self.assertFalse(result["result"]["pythonpath_visible"])
        self.assertTrue(result["result"]["note_exists"])
        artifact_dir = AGENTS_V2_ROOT / result["artifact_dir"]
        self.assertTrue((artifact_dir / "note.txt").exists())

    def test_aroon_uses_most_recent_ties_and_psar_respects_initial_downtrend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            flat = [
                _candle(BTC, "1h", BASE_MS + index * ONE_HOUR_MS, ONE_HOUR_MS, 100.0, 101.0, 99.0, 100.0, 10.0)
                for index in range(30)
            ]
            down = [
                _candle(ETH, "1h", BASE_MS + index * ONE_HOUR_MS, ONE_HOUR_MS, 100.0 - index, 101.0 - index, 99.0 - index, 100.0 - index, 10.0)
                for index in range(30)
            ]
            cache.upsert_candles([*flat, *down])
            with _cache_env(cache.path):
                flat_result = compute_indicators_impl(BTC, "1h", as_of=flat[-1].close_time, limit=30, indicators="aroon", source="cache")
                down_result = compute_indicators_impl(ETH, "1h", as_of=down[-1].close_time, limit=30, indicators="psar", source="cache")

        self.assertTrue(flat_result["ok"], flat_result)
        self.assertEqual(flat_result["latest"]["aroon_up_25"], 100.0)
        self.assertEqual(flat_result["latest"]["aroon_down_25"], 100.0)
        self.assertTrue(down_result["ok"], down_result)
        self.assertGreater(down_result["tail"][1]["psar_0.02_0.2"], down_result["tail"][1]["c"])


def _trend_candles(symbol: str, interval: str, base_ms: int, duration_ms: int, count: int) -> list[Candle]:
    candles = []
    for index in range(count):
        close = 100.0 + index * 0.5
        candles.append(
            _candle(
                symbol,
                interval,
                base_ms + index * duration_ms,
                duration_ms,
                close - 0.2,
                close + 1.0,
                close - 1.0,
                close,
                100.0 + index,
            )
        )
    return candles


def _candle(
    symbol: str,
    interval: str,
    close_time: int,
    duration_ms: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> Candle:
    return Candle(
        symbol=symbol,
        interval=interval,
        open_time=close_time - duration_ms + 1,
        close_time=close_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class _cache_env:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.previous_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")

    def __enter__(self) -> None:
        os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(self.cache_path)

    def __exit__(self, exc_type, exc, tb) -> None:
        _restore_env("TRADERBOT_MARKET_CACHE_PATH", self.previous_cache)


class _deterministic_env:
    def __init__(self, cache_path: Path, as_of_ms: int, candidates: list[dict]) -> None:
        self.cache_path = cache_path
        self.as_of_ms = as_of_ms
        self.candidates = candidates
        self.previous_cache = os.environ.get("TRADERBOT_MARKET_CACHE_PATH")
        self.previous_mode = os.environ.get("TRADERBOT_SCREENER_MODE")
        self.previous_candidates = os.environ.get("TRADERBOT_DETERMINISTIC_CANDIDATES")
        self.previous_process_clock = process_simulation_clock_ms()
        self.previous_file_clock = file_simulation_clock_ms()

    def __enter__(self) -> None:
        os.environ["TRADERBOT_MARKET_CACHE_PATH"] = str(self.cache_path)
        os.environ["TRADERBOT_SCREENER_MODE"] = "deterministic"
        os.environ["TRADERBOT_DETERMINISTIC_CANDIDATES"] = json.dumps(self.candidates)
        set_simulation_clock_state(self.as_of_ms)

    def __exit__(self, exc_type, exc, tb) -> None:
        _restore_env("TRADERBOT_MARKET_CACHE_PATH", self.previous_cache)
        _restore_env("TRADERBOT_SCREENER_MODE", self.previous_mode)
        _restore_env("TRADERBOT_DETERMINISTIC_CANDIDATES", self.previous_candidates)
        if self.previous_file_clock is None:
            clear_file_simulation_clock_state()
        else:
            set_file_simulation_clock_state(self.previous_file_clock)
        if self.previous_process_clock is None:
            clear_process_simulation_clock_state()
        else:
            set_process_simulation_clock_state(self.previous_process_clock)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
