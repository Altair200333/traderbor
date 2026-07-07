"""Pluggable scanner providers for the deterministic runner.

Two providers:
- "legacy": the existing screener.scan (called directly at the replay seam so
  test patches of run_screener keep working; nothing here touches that path).
- "v2": the research scanner (research/scanner_lab/scanner_v2.py) — long-only
  P1 breakouts, BTC>EMA50 regime gate, LightGBM score, EV threshold, one signal
  per symbol per UTC day. This module adapts it to the production surfaces:
  LocalMarketCache in, ScanResult out, production state blocks applied.

Heavy deps (pandas/lightgbm) are imported lazily inside scan_v2 so the screener
package stays dependency-light for legacy use.

Env:
  TRADERBOT_SCANNER_PROVIDER   legacy|v2 (default legacy) — replay default
  TRADERBOT_SCANNER_V2_ARTIFACTS  artifact dir override
  TRADERBOT_SCANNER_V2_MIN_EV  EV threshold override (float, default 0.0)
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ScreenerConfig
from .gates import scan_global_blocks, state_blocks
from .plan import PlanPrimitives
from .screener import ScanResult, SymbolRow
from .state import TradingState, candidate_cooldown_key

SCANNER_PROVIDERS = ("legacy", "v2")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCANNER_LAB = _REPO_ROOT / "research" / "scanner_lab"
_DEFAULT_ARTIFACTS = _REPO_ROOT / "research" / "artifacts" / "scanner_v2"

_v2_runtime_cache: dict[str, Any] = {}


def resolve_scanner_provider(name: str | None) -> str:
    resolved = (name or os.environ.get("TRADERBOT_SCANNER_PROVIDER") or "legacy").strip().lower()
    if resolved not in SCANNER_PROVIDERS:
        raise ValueError(f"unknown scanner provider: {resolved!r} (expected one of {SCANNER_PROVIDERS})")
    return resolved


def _artifact_dir() -> Path:
    override = os.environ.get("TRADERBOT_SCANNER_V2_ARTIFACTS")
    return Path(override) if override else _DEFAULT_ARTIFACTS


def _load_v2_runtime(artifact_dir: Path) -> Any:
    key = str(artifact_dir)
    if key not in _v2_runtime_cache:
        import sys
        if str(_SCANNER_LAB) not in sys.path:
            sys.path.insert(0, str(_SCANNER_LAB))
        from scanner_v2 import ScannerV2  # research runtime, single source of truth
        _v2_runtime_cache[key] = ScannerV2(artifact_dir=artifact_dir)
    return _v2_runtime_cache[key]


def _cached_1h_symbols(cache_path: str | Path) -> list[str]:
    conn = sqlite3.connect(f"file:{Path(cache_path)}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM candles WHERE provider='binance' AND market='spot' AND interval='1h'"
        ).fetchall()
    finally:
        conn.close()
    return sorted(r[0] for r in rows)


def _frames_from_cache(cache_path: str | Path, symbols: list[str], as_of_ms: int,
                       limit: int = 1100) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    frames: dict[str, Any] = {}
    conn = sqlite3.connect(f"file:{Path(cache_path)}?mode=ro", uri=True)
    try:
        for symbol in symbols:
            rows = conn.execute(
                "SELECT open_time, open, high, low, close, volume, quote_volume, trade_count, taker_buy_base_volume "
                "FROM candles WHERE provider='binance' AND market='spot' AND symbol=? AND interval='1h' "
                "AND close_time <= ? ORDER BY open_time DESC LIMIT ?",
                (symbol, as_of_ms, limit),
            ).fetchall()
            if len(rows) < 400:
                continue
            arr = list(reversed(rows))
            df = pd.DataFrame(arr, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_base"])
            for col in ("quote_volume", "trades", "taker_buy_base"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            df.replace({None: np.nan}, inplace=True)
            frames[symbol] = df
    finally:
        conn.close()
    return frames


def _emitted_today(state: TradingState, as_of_ms: int) -> set[str]:
    day = datetime.fromtimestamp(as_of_ms / 1000, tz=timezone.utc).date()
    out: set[str] = set()
    for key, ts in state.last_candidate_ts.items():
        if not key.endswith(":long"):
            continue
        if datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date() == day:
            out.add(key.rsplit(":", 1)[0])
    return out


def _config_hash(artifact_dir: Path) -> str:
    meta = artifact_dir / "meta.json"
    payload = meta.read_bytes() if meta.exists() else b"missing"
    return hashlib.sha256(payload).hexdigest()[:12]


def scan_v2(
    symbols: list[str],
    as_of_ms: int,
    cfg: ScreenerConfig | None = None,
    state: TradingState | None = None,
    cache_path: str | Path | None = None,
    min_ev: float | None = None,
) -> ScanResult:
    """Scanner-v2 scan adapted to the production ScanResult surface.

    Market features (breadth/ranks) are computed over ALL 1h symbols present in
    the cache; candidate rows are emitted only for `symbols`.
    """
    cfg = cfg or ScreenerConfig()
    state = state or TradingState()
    if cache_path is None:
        raise ValueError("scanner v2 requires cache_path")
    if min_ev is None:
        min_ev = float(os.environ.get("TRADERBOT_SCANNER_V2_MIN_EV", "0.0"))

    artifact_dir = _artifact_dir()
    runtime = _load_v2_runtime(artifact_dir)
    universe = _cached_1h_symbols(cache_path)
    frames = _frames_from_cache(cache_path, universe, as_of_ms)
    as_of_iso = datetime.fromtimestamp(as_of_ms / 1000, tz=timezone.utc).isoformat()

    data_warnings: list[str] = []
    if len(frames) < 100:
        data_warnings.append(
            f"scanner_v2: market breadth computed over {len(frames)} cached symbols "
            f"(model trained on 149) - scores may drift")

    wanted = {s.upper() for s in symbols}
    rows: dict[str, SymbolRow] = {}
    for symbol in sorted(wanted):
        frame = frames.get(symbol)
        if frame is None:
            rows[symbol] = SymbolRow(symbol=symbol, status="insufficient_data",
                                     data_issue={"reason": "scanner_v2: <400 1h bars in cache"})
        else:
            rows[symbol] = SymbolRow(symbol=symbol, status="ok",
                                     close=float(frame["close"].iloc[-1]))

    btc_roc_4h = None
    if "BTCUSDT" in frames:
        c = frames["BTCUSDT"]["close"]
        if len(c) > 4:
            btc_roc_4h = float(c.iloc[-1] / c.iloc[-5] - 1.0)

    global_blocks = scan_global_blocks(state, cfg, as_of_ms)
    result_candidates: list[str] = []

    if "BTCUSDT" not in frames:
        data_warnings.append("scanner_v2: BTCUSDT missing from cache - scan skipped")
        scan_out = {"candidates": [], "heat24": 0, "regime_ok": False}
    else:
        scan_out = runtime.scan(frames, as_of_ms,
                                emitted_today=_emitted_today(state, as_of_ms),
                                min_ev=min_ev)
    if not scan_out.get("regime_ok", False):
        global_blocks = list(global_blocks) + ["btc_regime_bear"]

    for cand in scan_out["candidates"]:
        symbol = cand["symbol"]
        if symbol not in wanted:
            continue
        row = rows[symbol]
        plan = PlanPrimitives(
            pattern_used="P1",
            boundary_price=cand["boundary"],
            trigger_age_bars=0,
            invalidation_price=cand["invalidation"],
            d_atr=cand["d_atr"],
            d_struct=cand["d_struct"],
            d_noise=cand["d_noise"],
            d_final=cand["d_final"],
            stop_feasible=True,
            tp_rr_default=cand["tp_rr_exec"],
            ref_entry=cand["entry_ref"],
        )
        blocked, extra_global = state_blocks(symbol, "long", state, as_of_ms, cfg,
                                             signal_quality="hard")
        for item in extra_global:
            if item not in global_blocks:
                global_blocks.append(item)
        update = {
            "close": cand["entry_ref"],
            "roc_4h": cand["roc_4h"],
            "roc_24h": cand["roc_24h"],
            "roc_1h_last": cand["roc_1h"],
            "vol_ratio": cand["vol_ratio"],
            "rsi": cand["rsi14"],
            "atr_pct": cand["atr_pct"],
            "ema20_ext_atr": cand["ema20_ext_atr"],
            "patterns_long": ["P1"],
            "signal_candidate_before_state": "long",
            "signal_quality_before_state": "hard",
            "plan": plan,
            "candidate_score": cand["ev_r"],
        }
        if blocked or global_blocks_block_entry(global_blocks):
            update["blocked_by"] = blocked or ["global"]
        else:
            update["candidate"] = "long"
            update["candidate_quality"] = "hard"
            result_candidates.append(symbol)
        rows[symbol] = row.model_copy(update=update)

    return ScanResult(
        as_of_ms=as_of_ms,
        as_of_iso=as_of_iso,
        screener_version="scanner-v2-0.1.0",
        config_hash=_config_hash(artifact_dir),
        symbols=[rows[s] for s in sorted(rows)],
        candidates=result_candidates,
        global_blocks=global_blocks,
        btc_roc_4h=btc_roc_4h,
        data_warnings=data_warnings,
    )


def global_blocks_block_entry(global_blocks: list[str]) -> bool:
    """Blocks that must suppress new candidates entirely (regime gate included:
    scanner-v2 policy is long-only in bull regime)."""
    blocking = {"halt_active", "max_trades_per_day", "max_positions",
                "daily_loss_limit", "weekly_loss_limit", "btc_regime_bear"}
    return any(b in blocking for b in global_blocks)
