from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.market import DEFAULT_CACHE_PATH, INTERVAL_MS, normalize_symbol, parse_time_ms
from traderbot_ai.screener.render import to_markdown_table
from traderbot_ai.screener.screener import SymbolRow, scan
from traderbot_ai.screener.state import ScreenerStateStore, TradingState
from traderbot_ai.simulator.market_cache import LocalMarketCache


def scan_window_summary(
    symbols: list[str] | str,
    *,
    start_ms: int | str | float,
    end_ms: int | str | float,
    step_interval: str = "1h",
    cache_path: str | Path | None = None,
    cfg: ScreenerConfig | None = None,
    include_forward: bool = True,
    forward_horizon_hours: int = 24,
    near_miss_limit: int = 50,
) -> dict[str, Any]:
    if step_interval not in INTERVAL_MS:
        raise ValueError(f"unsupported step_interval: {step_interval}")
    start = parse_time_ms(start_ms)
    end = parse_time_ms(end_ms)
    if start is None or end is None:
        raise ValueError("start_ms and end_ms are required")
    if start >= end:
        raise ValueError("start_ms must be before end_ms")
    normalized_symbols = _split_symbols(symbols)
    cfg = cfg or ScreenerConfig()
    cache = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
    state_store = ScreenerStateStore()
    status_counts: Counter[str] = Counter()
    single_gate_counts: Counter[str] = Counter()
    extension_gate_counts: Counter[str] = Counter()
    data_warnings: Counter[str] = Counter()
    candidate_forward_counts: Counter[str] = Counter()
    candidate_quality_counts: Counter[str] = Counter()
    hard_candidate_cooldown_block_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    extension_misses: list[dict[str, Any]] = []
    hard_candidate_cooldown_blocks: list[dict[str, Any]] = []
    checked_rows = 0
    max_markdown_chars = 0
    step_count = 0
    step_ms = INTERVAL_MS[step_interval]
    as_of = start

    while as_of < end:
        step_count += 1
        state = TradingState(last_candidate_ts=dict(state_store.last_candidate_ts), last_candidate_quality=dict(state_store.last_candidate_quality))
        result = scan(normalized_symbols, as_of_ms=as_of, cfg=cfg, state=state, cache_path=cache)
        rows = [row.model_dump(mode="json") for row in result.symbols]
        state_store.update_from_scan_rows(rows, as_of)
        max_markdown_chars = max(max_markdown_chars, len(to_markdown_table(result)))
        data_warnings.update(result.data_warnings)
        for row in result.symbols:
            checked_rows += 1
            status_counts[row.status] += 1
            if row.candidate in {"long", "short"}:
                item = _candidate_item(row, as_of, cache, forward_horizon_hours if include_forward else 0)
                candidates.append(item)
                if row.candidate_quality:
                    candidate_quality_counts[row.candidate_quality] += 1
                outcome = item.get("forward", {}).get("tp_sl_first") if isinstance(item.get("forward"), dict) else None
                if outcome:
                    candidate_forward_counts[str(outcome)] += 1
            if row.status == "ok":
                if row.signal_quality_before_state == "hard" and row.candidate not in {"long", "short"} and "cooldown_candidate" in row.blocked_by:
                    origin = row.cooldown_source_quality or "unknown"
                    hard_candidate_cooldown_block_counts[origin] += 1
                    if len(hard_candidate_cooldown_blocks) < near_miss_limit:
                        hard_candidate_cooldown_blocks.append(_cooldown_block_item(row, as_of))
                extension_key = _extension_failure_key(row.failed_gates)
                if extension_key:
                    extension_gate_counts[extension_key] += 1
                    if len(extension_misses) < near_miss_limit:
                        extension_misses.append(_extension_miss_item(row, as_of, cache, forward_horizon_hours if include_forward else 0))
                if row.candidate not in {"long", "short"} and len(row.failed_gates) == 1:
                    single_gate_counts[row.failed_gates[0]] += 1
                    if len(near_misses) < near_miss_limit:
                        near_misses.append(_near_miss_item(row, as_of, cache, forward_horizon_hours if include_forward else 0))
        as_of += step_ms

    return {
        "symbols": normalized_symbols,
        "start_ms": start,
        "start_iso": _iso(start),
        "end_ms": end,
        "end_iso": _iso(end),
        "step_interval": step_interval,
        "step_count": step_count,
        "checked_rows": checked_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "candidate_count": len(candidates),
        "candidate_forward_counts": dict(sorted(candidate_forward_counts.items())),
        "candidate_quality_counts": dict(sorted(candidate_quality_counts.items())),
        "hard_candidate_cooldown_block_counts": dict(sorted(hard_candidate_cooldown_block_counts.items())),
        "candidates": candidates,
        "single_gate_counts": dict(single_gate_counts.most_common()),
        "extension_gate_counts": dict(extension_gate_counts.most_common()),
        "near_misses": near_misses,
        "extension_misses": extension_misses,
        "hard_candidate_cooldown_blocks": hard_candidate_cooldown_blocks,
        "data_warnings": dict(sorted(data_warnings.items())),
        "max_markdown_chars": max_markdown_chars,
        "cooldown_keys": dict(sorted(state_store.last_candidate_ts.items())),
    }


def _candidate_item(row: SymbolRow, as_of_ms: int, cache_path: Path, forward_horizon_hours: int) -> dict[str, Any]:
    item = {
        "as_of_ms": as_of_ms,
        "as_of_iso": _iso(as_of_ms),
        "symbol": row.symbol,
        "side": row.candidate,
        "quality": row.candidate_quality or "hard",
        "failed_gates": row.failed_gates,
        "marginal_reasons": row.marginal_reasons,
        "marginal_score": row.marginal_score,
        "pattern": row.plan.pattern_used if row.plan else None,
        "close": row.close,
        "roc_4h": row.roc_4h,
        "roc_24h": row.roc_24h,
        "roc_1h_last": row.roc_1h_last,
        "vol_ratio": row.vol_ratio,
        "rsi": row.rsi,
        "atr_pct": row.atr_pct,
        "ema20_ext_atr": row.ema20_ext_atr,
        "funding": row.funding,
        "plan": None if row.plan is None else row.plan.model_dump(mode="json"),
    }
    if forward_horizon_hours > 0 and row.candidate in {"long", "short"} and row.plan is not None:
        item["forward"] = _forward_label(cache_path, row.symbol, row.candidate, as_of_ms, row.plan.ref_entry, row.plan.d_final, row.plan.tp_rr_default, forward_horizon_hours)
    return item


def _near_miss_item(row: SymbolRow, as_of_ms: int, cache_path: Path, forward_horizon_hours: int) -> dict[str, Any]:
    side = _best_side(row)
    details = row.gate_details_long if side == "long" else row.gate_details_short
    item = {
        "as_of_ms": as_of_ms,
        "as_of_iso": _iso(as_of_ms),
        "symbol": row.symbol,
        "side": side,
        "failed_gate": row.failed_gates[0],
        "patterns": row.patterns_long if side == "long" else row.patterns_short,
        "roc_4h": row.roc_4h,
        "roc_24h": row.roc_24h,
        "roc_1h_last": row.roc_1h_last,
        "vol_ratio": row.vol_ratio,
        "rsi": row.rsi,
        "atr_pct": row.atr_pct,
        "ema20_ext_atr": row.ema20_ext_atr,
        "funding": row.funding,
        "s9a": _gate_value(details, "S9a"),
        "s9b": _gate_value(details, "S9b"),
        "s9c": _gate_value(details, "S9c"),
    }
    if forward_horizon_hours > 0 and row.close is not None:
        item["forward"] = _forward_move_label(cache_path, row.symbol, side, as_of_ms, row.close, forward_horizon_hours)
    return item


def _extension_miss_item(row: SymbolRow, as_of_ms: int, cache_path: Path, forward_horizon_hours: int) -> dict[str, Any]:
    item = _near_miss_item(row, as_of_ms, cache_path, forward_horizon_hours)
    item["failed_gates"] = row.failed_gates
    item["extension_key"] = _extension_failure_key(row.failed_gates)
    item["candidate"] = row.candidate
    item["candidate_quality"] = row.candidate_quality
    item["blocked_by"] = row.blocked_by
    item["signal_candidate_before_state"] = row.signal_candidate_before_state
    item["signal_quality_before_state"] = row.signal_quality_before_state
    item["signal_marginal_reasons_before_state"] = row.signal_marginal_reasons_before_state
    item["signal_marginal_score_before_state"] = row.signal_marginal_score_before_state
    item["marginal_reasons"] = row.marginal_reasons
    item["marginal_score"] = row.marginal_score
    return item


def _cooldown_block_item(row: SymbolRow, as_of_ms: int) -> dict[str, Any]:
    return {
        "as_of_ms": as_of_ms,
        "as_of_iso": _iso(as_of_ms),
        "symbol": row.symbol,
        "side": row.signal_candidate_before_state,
        "signal_quality_before_state": row.signal_quality_before_state,
        "cooldown_source_quality": row.cooldown_source_quality,
        "blocked_by": row.blocked_by,
        "patterns": row.patterns_long if row.signal_candidate_before_state == "long" else row.patterns_short,
        "failed_gates": row.failed_gates,
        "roc_4h": row.roc_4h,
        "roc_24h": row.roc_24h,
        "vol_ratio": row.vol_ratio,
        "ema20_ext_atr": row.ema20_ext_atr,
    }


def _forward_label(cache_path: Path, symbol: str, side: str, as_of_ms: int, entry: float, stop_distance: float, tp_rr: float, horizon_hours: int) -> dict[str, Any]:
    if entry <= 0 or stop_distance <= 0 or tp_rr <= 0:
        return {"tp_sl_first": "invalid_plan"}
    if side == "long":
        stop = entry * (1.0 - stop_distance)
        take = entry * (1.0 + stop_distance * tp_rr)
    else:
        stop = entry * (1.0 + stop_distance)
        take = entry * (1.0 - stop_distance * tp_rr)
    horizon_ms = int(horizon_hours) * 60 * 60_000
    cache = LocalMarketCache(cache_path)
    candles = cache.get_candles(symbol, "1m", start_ms=as_of_ms, end_ms=as_of_ms + horizon_ms)
    if not candles:
        return {"tp_sl_first": "missing_1m", "stop": stop, "take_profit": take}

    max_high = max(candle.high for candle in candles)
    min_low = min(candle.low for candle in candles)
    if side == "long":
        max_favorable = (max_high - entry) / entry
        max_adverse = (entry - min_low) / entry
    else:
        max_favorable = (entry - min_low) / entry
        max_adverse = (max_high - entry) / entry

    first = "none"
    first_time_ms = None
    ambiguous = False
    for candle in candles:
        if side == "long":
            hit_stop = candle.low <= stop
            hit_take = candle.high >= take
        else:
            hit_stop = candle.high >= stop
            hit_take = candle.low <= take
        if hit_stop and hit_take:
            first = "ambiguous"
            first_time_ms = candle.open_time
            ambiguous = True
            break
        if hit_stop:
            first = "sl"
            first_time_ms = candle.open_time
            break
        if hit_take:
            first = "tp"
            first_time_ms = candle.open_time
            break

    return {
        "tp_sl_first": first,
        "first_time_ms": first_time_ms,
        "first_time_iso": None if first_time_ms is None else _iso(first_time_ms),
        "ambiguous": ambiguous,
        "stop": stop,
        "take_profit": take,
        "max_favorable_pct": max_favorable,
        "max_adverse_pct": max_adverse,
        "horizon_hours": horizon_hours,
        "bars_1m": len(candles),
    }


def _forward_move_label(cache_path: Path, symbol: str, side: str, as_of_ms: int, entry: float, horizon_hours: int) -> dict[str, Any]:
    if entry <= 0:
        return {"ok": False, "error": "invalid_entry"}
    horizon_ms = int(horizon_hours) * 60 * 60_000
    cache = LocalMarketCache(cache_path)
    candles = cache.get_candles(symbol, "1m", start_ms=as_of_ms, end_ms=as_of_ms + horizon_ms)
    if not candles:
        return {"ok": False, "error": "missing_1m", "horizon_hours": horizon_hours}
    max_high = max(candle.high for candle in candles)
    min_low = min(candle.low for candle in candles)
    if side == "long":
        max_favorable = (max_high - entry) / entry
        max_adverse = (entry - min_low) / entry
    else:
        max_favorable = (entry - min_low) / entry
        max_adverse = (max_high - entry) / entry
    return {
        "ok": True,
        "max_favorable_pct": max_favorable,
        "max_adverse_pct": max_adverse,
        "horizon_hours": horizon_hours,
        "bars_1m": len(candles),
    }


def _best_side(row: SymbolRow) -> str:
    long_failures = sum(1 for value in row.gates_long.values() if value is False)
    short_failures = sum(1 for value in row.gates_short.values() if value is False)
    return "long" if long_failures <= short_failures else "short"


def _gate_value(details: dict[str, Any], name: str) -> Any:
    gate = details.get(name)
    return None if gate is None else gate.value


def _extension_failure_key(failed_gates: list[str]) -> str | None:
    extension_failures = [gate for gate in failed_gates if gate in {"S9b", "S9c"}]
    if not extension_failures or len(extension_failures) != len(failed_gates):
        return None
    return "+".join(sorted(extension_failures))


def _split_symbols(symbols: list[str] | str) -> list[str]:
    if isinstance(symbols, str):
        values = symbols.split(",")
    else:
        values = symbols
    return [normalize_symbol(str(symbol)) for symbol in values if str(symbol).strip()]


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan deterministic screener over a time window.")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-interval", default="1h")
    parser.add_argument("--cache-path")
    parser.add_argument("--output")
    parser.add_argument("--no-forward", action="store_true")
    parser.add_argument("--forward-horizon-hours", type=int, default=24)
    parser.add_argument("--near-miss-limit", type=int, default=50)
    args = parser.parse_args()

    summary = scan_window_summary(
        args.symbols,
        start_ms=args.start,
        end_ms=args.end,
        step_interval=args.step_interval,
        cache_path=args.cache_path,
        include_forward=not args.no_forward,
        forward_horizon_hours=args.forward_horizon_hours,
        near_miss_limit=args.near_miss_limit,
    )
    payload = json.dumps(summary, ensure_ascii=True, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
