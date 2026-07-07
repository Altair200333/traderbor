"""Wide-data validation for BOTH scanner providers on the 2-year 149-coin
research dataset, plus an e2e deterministic replay smoke for each provider.

Env-gated: TRADERBOT_RUN_SLOW_SCANNER_V2_TESTS=1
Requires: research/data/klines parquet (download_klines.py) + scanner-v2
artifacts. The wide 1h SQLite cache is built once into agents-v2/data/wide-2y-1h/
(reused across runs; delete the dir to force a rebuild).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENTS_DIR.parent
WIDE_CACHE = AGENTS_DIR / "data" / "wide-2y-1h" / "market_cache.sqlite3"
KLINES = REPO_ROOT / "research" / "data" / "klines" / "1h"
ARTIFACTS = REPO_ROOT / "research" / "artifacts" / "scanner_v2"
ONE_HOUR = 3_600_000

SLOW = os.environ.get("TRADERBOT_RUN_SLOW_SCANNER_V2_TESTS") == "1"

# fixed, spread across regimes; hour-aligned UTC
SAMPLE_HOURS = [
    "2025-02-03T14:00:00Z", "2025-04-20T16:00:00Z", "2025-06-11T03:00:00Z",
    "2025-08-19T21:00:00Z", "2025-10-03T02:00:00Z", "2025-12-12T13:00:00Z",
    "2026-02-09T07:00:00Z", "2026-03-27T08:00:00Z", "2026-05-18T06:00:00Z",
    "2026-07-05T19:00:00Z",
]


def _ts(iso: str) -> int:
    import pandas as pd
    return int(pd.Timestamp(iso).timestamp() * 1000)


def _ensure_wide_cache() -> None:
    if WIDE_CACHE.exists():
        return
    script = REPO_ROOT / "research" / "scanner_lab" / "export_cache.py"
    subprocess.run(
        [sys.executable, str(script), "--out", str(WIDE_CACHE), "--intervals", "1h"],
        check=True, cwd=str(script.parent), timeout=1800)


@unittest.skipUnless(SLOW, "set TRADERBOT_RUN_SLOW_SCANNER_V2_TESTS=1")
@unittest.skipUnless(KLINES.exists() and (ARTIFACTS / "model.txt").exists(),
                     "research data or scanner-v2 artifacts missing")
class WideScannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_wide_cache()
        from traderbot_ai.screener.providers import _cached_1h_symbols
        cls.universe = _cached_1h_symbols(WIDE_CACHE)
        assert len(cls.universe) >= 140, f"wide cache has only {len(cls.universe)} symbols"

    def test_v2_sweep_invariants(self) -> None:
        """scan_v2 across sampled hours: no exceptions, long-P1-only, feasible
        plans, deterministic."""
        from traderbot_ai.screener.providers import scan_v2
        total_candidates = 0
        for iso in SAMPLE_HOURS:
            as_of = _ts(iso)
            result = scan_v2(symbols=self.universe, as_of_ms=as_of,
                             cache_path=WIDE_CACHE)
            self.assertEqual(result.as_of_ms, as_of)
            for symbol in result.candidates:
                row = next(r for r in result.symbols if r.symbol == symbol)
                self.assertEqual(row.candidate, "long")
                self.assertEqual(row.plan.pattern_used, "P1")
                self.assertTrue(0.01 <= row.plan.d_final <= 0.04, symbol)
                self.assertGreater(row.plan.ref_entry, 0)
                self.assertLess(row.plan.invalidation_price, row.plan.ref_entry)
                self.assertIsNotNone(row.candidate_score)
            total_candidates += len(result.candidates)
            again = scan_v2(symbols=self.universe, as_of_ms=as_of,
                            cache_path=WIDE_CACHE)
            self.assertEqual(result.model_dump(mode="json"),
                             again.model_dump(mode="json"), f"non-deterministic at {iso}")
        # EV>0 selectivity: most hours emit nothing; sanity band on the total
        self.assertLessEqual(total_candidates, 40)

    def test_v2_week_baseline(self) -> None:
        """Pinned regression baseline: hourly sweep of one active week.
        If this moves, the scanner behavior changed - re-baseline consciously
        (same discipline as the legacy month matrix)."""
        from concurrent.futures import ThreadPoolExecutor

        from traderbot_ai.screener.providers import scan_v2
        start = _ts("2026-06-22T00:00:00Z")

        def one(i: int) -> list[tuple[str, str]]:
            result = scan_v2(symbols=self.universe, as_of_ms=start + i * ONE_HOUR,
                             cache_path=WIDE_CACHE)
            return [(sym, result.as_of_iso) for sym in result.candidates]

        with ThreadPoolExecutor(max_workers=6) as ex:
            emitted = [item for chunk in ex.map(one, range(168)) for item in chunk]
        # stateless scans: symbol-day dedup is applied by the replay state, not here
        self.assertEqual(len(emitted), WEEK_BASELINE_COUNT,
                         f"scanner-v2 weekly flow changed (n={len(emitted)}): {emitted}")

    def test_legacy_wide_smoke(self) -> None:
        """Legacy screener runs on the 149-coin cache without exceptions and
        with sane status distribution."""
        from traderbot_ai.screener import scan
        result = scan(symbols=self.universe, as_of_ms=_ts("2026-06-25T12:00:00Z"),
                      cache_path=WIDE_CACHE)
        statuses = [r.status for r in result.symbols]
        self.assertEqual(len(statuses), len(self.universe))
        self.assertGreaterEqual(statuses.count("ok"), 120)
        for symbol in result.candidates:
            row = next(r for r in result.symbols if r.symbol == symbol)
            self.assertIsNotNone(row.plan)

    def test_v2_trigger_parity_vs_research_dataset(self) -> None:
        """Provider path (SQLite + 1100-bar warmup) must reproduce the research
        pool triggers (parquet, full history) at sampled hours: same long-P1
        stop-feasible trigger symbols."""
        import pandas as pd
        events = REPO_ROOT / "research" / "data" / "events.parquet"
        if not events.exists():
            self.skipTest("events.parquet missing")
        from traderbot_ai.screener.providers import scan_v2
        ev = pd.read_parquet(events, columns=[
            "symbol", "side", "pattern", "as_of", "stop_feasible"])
        ev = ev[(ev["side"] == "long") & (ev["pattern"] == "P1") & ev["stop_feasible"]]
        mismatches = []
        bear_hours = 0
        for iso in SAMPLE_HOURS:
            as_of = _ts(iso)
            expected = set(ev[ev["as_of"] == as_of]["symbol"])
            result = scan_v2(symbols=self.universe, as_of_ms=as_of,
                             cache_path=WIDE_CACHE, min_ev=-100.0)
            got = set(result.candidates)
            if "btc_regime_bear" in result.global_blocks:
                # regime gate suppresses everything by design; the research pool
                # (events.parquet) is regime-agnostic, so only assert suppression
                bear_hours += 1
                if got:
                    mismatches.append((iso, "bear regime must emit nothing", sorted(got)))
                continue
            if got != expected:
                mismatches.append((iso, sorted(expected - got), sorted(got - expected)))
        self.assertEqual(mismatches, [], f"trigger parity broken: {mismatches}")
        # sample must exercise both regimes to keep this test meaningful
        self.assertGreaterEqual(bear_hours, 1)
        self.assertLessEqual(bear_hours, len(SAMPLE_HOURS) - 3)


@unittest.skipUnless(SLOW, "set TRADERBOT_RUN_SLOW_SCANNER_V2_TESTS=1")
@unittest.skipUnless(KLINES.exists() and (ARTIFACTS / "model.txt").exists(),
                     "research data or scanner-v2 artifacts missing")
class ReplayE2ETests(unittest.TestCase):
    """Deterministic replay smoke on the wide cache for BOTH providers."""

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_wide_cache()

    def _run(self, provider: str) -> dict:
        import json as _json
        from traderbot_ai.simulator.exchange_replay import (
            build_exchange_replay_config, run_exchange_replay)
        from traderbot_ai.simulator.market_cache import LocalMarketCache
        td = Path(tempfile.mkdtemp(prefix=f"replay-{provider}-"))
        config = build_exchange_replay_config(
            symbols="ZECUSDT,PEPEUSDT,SOLUSDT,PENDLEUSDT,WIFUSDT,TAOUSDT",
            start_time="2026-06-24T00:00:00Z", end_time="2026-06-26T00:00:00Z",
            decision_interval="1h", execution_interval="1h",
            screener_mode="deterministic", scanner_provider=provider,
            run_id=f"wide-smoke-{provider}",
            state_path=td / "s.json", events_path=td / "e.jsonl",
            replay_path=td / "r.jsonl")
        result = run_exchange_replay(
            config=config, decide=lambda _ctx: {"final_decision": "hold"},
            cache=LocalMarketCache(path=WIDE_CACHE))
        steps = [_json.loads(line) for line in
                 (td / "r.jsonl").read_text(encoding="utf-8").splitlines()]
        return {"result": result, "steps": steps}

    @staticmethod
    def _scans(steps: list[dict]) -> list[dict]:
        return [s["payload"]["scan"] for s in steps
                if s.get("type") == "step_completed"
                and isinstance(s.get("payload", {}).get("scan"), dict)]

    def test_replay_legacy(self) -> None:
        scans = self._scans(self._run("legacy")["steps"])
        self.assertEqual(len(scans), 48)
        self.assertTrue(all(s["scanner_provider"] == "legacy" for s in scans))

    def test_replay_v2(self) -> None:
        scans = self._scans(self._run("v2")["steps"])
        self.assertEqual(len(scans), 48)
        self.assertTrue(all(s["scanner_provider"] == "v2" for s in scans))
        self.assertEqual({s["scanner_version"] for s in scans}, {"scanner-v2-0.1.0"})
        for s in scans:
            for item in s["candidate_primitives"]:
                self.assertEqual(item["side"], "long")
                self.assertIsNotNone(item["plan"])
                self.assertTrue(0.01 <= item["plan"]["d_final"] <= 0.04)


# Pinned 2026-07-07 on wide-2y-1h cache + scanner_v2 artifacts (post breadth-fix
# model). 10 signals in week 2026-06-22..29: bursts on Jun 22 (5) and Jun 27 (4),
# Jun 28 (1), silence otherwise. Re-pin consciously after any scanner change.
WEEK_BASELINE_COUNT = 10

if __name__ == "__main__":
    unittest.main()
