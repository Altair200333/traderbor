from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from traderbot_ai.screener.config import ScreenerConfig, config_hash
from traderbot_ai.screener.data import DataIssue, load_closed, validate_frame
from traderbot_ai.screener.gates import GateResult, evaluate_signal_gates, failed_gate_names, gates_pass, scan_global_blocks, state_blocks
from traderbot_ai.screener.indicators import atr_wilder, ema, range_expansion_last, roc, rolling_median_previous, rsi_wilder, zscore_last
from traderbot_ai.screener.market import DEFAULT_CACHE_PATH, INTERVAL_MS, normalize_symbol, parse_time_ms
from traderbot_ai.screener.patterns import PatternHit, detect_patterns, patterns_for_side
from traderbot_ai.screener.plan import PlanPrimitives, build_plan_primitives
from traderbot_ai.screener.score import score_candidate
from traderbot_ai.screener.state import TradingState, candidate_cooldown_entry


CandidateSide = Literal["long", "short"]
CandidateQuality = Literal["hard", "marginal_extension"]
SymbolStatus = Literal["ok", "insufficient_data", "data_gap", "bad_data"]


class SymbolRow(BaseModel):
    symbol: str
    status: SymbolStatus
    close: float | None = None
    roc_4h: float | None = None
    roc_24h: float | None = None
    roc_1h_last: float | None = None
    vol_ratio: float | None = None
    rsi: float | None = None
    atr_pct: float | None = None
    ema20_ext_atr: float | None = None
    above_ema50: bool | None = None
    funding: float | None = None
    patterns_long: list[str] = Field(default_factory=list)
    patterns_short: list[str] = Field(default_factory=list)
    gates_long: dict[str, bool | None] = Field(default_factory=dict)
    gates_short: dict[str, bool | None] = Field(default_factory=dict)
    gate_details_long: dict[str, GateResult] = Field(default_factory=dict)
    gate_details_short: dict[str, GateResult] = Field(default_factory=dict)
    signal_candidate_before_state: CandidateSide | None = None
    signal_quality_before_state: CandidateQuality | None = None
    signal_marginal_reasons_before_state: list[str] = Field(default_factory=list)
    signal_marginal_score_before_state: float | None = None
    cooldown_source_quality: CandidateQuality | None = None
    candidate: CandidateSide | None = None
    candidate_quality: CandidateQuality | None = None
    candidate_score: float | None = None
    marginal_reasons: list[str] = Field(default_factory=list)
    marginal_score: float | None = None
    failed_gates: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    plan: PlanPrimitives | None = None
    data_issue: dict[str, Any] | None = None


class ScanResult(BaseModel):
    as_of_ms: int
    as_of_iso: str
    screener_version: str
    config_hash: str
    symbols: list[SymbolRow]
    candidates: list[str]
    global_blocks: list[str] = Field(default_factory=list)
    btc_roc_4h: float | None = None
    data_warnings: list[str] = Field(default_factory=list)


def scan(
    symbols: list[str] | str,
    as_of_ms: int | str | float,
    cfg: ScreenerConfig | None = None,
    state: TradingState | None = None,
    cache_path: str | Path | None = None,
    funding: dict[str, float] | None = None,
) -> ScanResult:
    cfg = cfg or ScreenerConfig()
    state = state or TradingState()
    parsed_as_of = parse_time_ms(as_of_ms)
    if parsed_as_of is None:
        raise ValueError("as_of_ms is required")
    normalized_symbols = _split_symbols(symbols)
    if not normalized_symbols:
        raise ValueError("at least one symbol is required")
    if cfg.btc_symbol not in normalized_symbols:
        scan_symbols = [*normalized_symbols, cfg.btc_symbol]
    else:
        scan_symbols = list(normalized_symbols)
    cache = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
    raw_rows: dict[str, SymbolRow] = {}
    btc_roc = None
    data_warnings = []
    global_blocks: list[str] = scan_global_blocks(state, cfg, parsed_as_of)
    for symbol in scan_symbols:
        row = _scan_symbol(symbol, parsed_as_of, cfg, state, cache, funding or {}, btc_roc_4h=None)
        raw_rows[symbol] = row
        if symbol == cfg.btc_symbol:
            btc_roc = row.roc_4h
    btc_row = raw_rows.get(cfg.btc_symbol)
    if cfg.btc_symbol not in normalized_symbols and btc_row is not None and btc_row.data_issue:
        data_warnings.append(f"{btc_row.symbol}:{btc_row.data_issue.get('reason')}")
    rows: list[SymbolRow] = []
    for symbol in normalized_symbols:
        row = _scan_symbol(symbol, parsed_as_of, cfg, state, cache, funding or {}, btc_roc_4h=btc_roc)
        rows.append(row)
        if row.data_issue:
            data_warnings.append(f"{row.symbol}:{row.data_issue.get('reason')}")
    _apply_marginal_candidate_cap(rows, cfg)
    _score_candidates(rows, cfg)
    _apply_candidate_rank_cap(rows, cfg)
    candidates = [row.symbol for row in rows if row.candidate in {"long", "short"}]
    return ScanResult(
        as_of_ms=parsed_as_of,
        as_of_iso=datetime.fromtimestamp(parsed_as_of / 1000.0, tz=timezone.utc).isoformat(),
        screener_version=cfg.version,
        config_hash=config_hash(cfg),
        symbols=rows,
        candidates=candidates,
        global_blocks=sorted(set(global_blocks)),
        btc_roc_4h=btc_roc,
        data_warnings=sorted(set(data_warnings)),
    )


def get_setup_digest(symbol: str, side: CandidateSide, as_of_ms: int | str | float, cfg: ScreenerConfig | None = None, state: TradingState | None = None, cache_path: str | Path | None = None) -> dict[str, Any]:
    result = scan([symbol], as_of_ms=as_of_ms, cfg=cfg, state=state, cache_path=cache_path)
    row = result.symbols[0]
    parsed_as_of = result.as_of_ms
    cfg = cfg or ScreenerConfig()
    cache = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
    frame = load_closed(cache, row.symbol, INTERVAL_MS["1h"], parsed_as_of, cfg.min_1h_bars)
    issue = validate_frame(frame, parsed_as_of, cfg.min_1h_bars)
    if issue is not None:
        return {
            "ok": False,
            "as_of_ms": result.as_of_ms,
            "symbol": row.symbol,
            "side": side,
            "row": agent_row_view(row),
            "error": issue.reason,
            "data_issue": {"reason": issue.reason, **issue.detail},
        }
    t = frame.length - 1
    close = frame.close[t]
    atr_value = atr_wilder(frame.high, frame.low, frame.close, cfg.atr_period)[t]
    support_1h, resistance_1h = _support_resistance_levels(frame, close, atr_value, "1h")
    frame_4h = _load_optional_frame(cache, row.symbol, INTERVAL_MS["4h"], parsed_as_of, cfg.min_4h_bars)
    if frame_4h is None:
        support_4h: list[dict[str, Any]] = []
        resistance_4h: list[dict[str, Any]] = []
    else:
        atr_4h_series = atr_wilder(frame_4h.high, frame_4h.low, frame_4h.close, cfg.atr_period)
        support_4h, resistance_4h = _support_resistance_levels(frame_4h, close, atr_4h_series[-1], "4h")
    support_levels = _nearest_levels([*support_1h, *support_4h], close, is_support=True)
    resistance_levels = _nearest_levels([*resistance_1h, *resistance_4h], close, is_support=False)
    plan = row.plan if row.candidate == side else None
    trigger_age_bars = plan.trigger_age_bars if plan is not None else None
    return {
        "ok": row.status == "ok",
        "as_of_ms": result.as_of_ms,
        "symbol": row.symbol,
        "side": side,
        "row": agent_row_view(row),
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "trigger_age_bars": trigger_age_bars,
        "retest_seen": _retest_seen(frame, side, plan.boundary_price if plan is not None else None, atr_value, trigger_age_bars),
    }


def _scan_symbol(symbol: str, as_of_ms: int, cfg: ScreenerConfig, state: TradingState, cache_path: Path, funding: dict[str, float], btc_roc_4h: float | None) -> SymbolRow:
    normalized = normalize_symbol(symbol)
    try:
        frame = load_closed(cache_path, normalized, INTERVAL_MS["1h"], as_of_ms, cfg.min_1h_bars)
        issue = validate_frame(frame, as_of_ms, cfg.min_1h_bars)
    except Exception as error:
        return SymbolRow(symbol=normalized, status="bad_data", data_issue={"reason": str(error)})
    if issue is not None:
        return SymbolRow(symbol=normalized, status=issue.status, data_issue={"reason": issue.reason, **issue.detail})
    closes = frame.close
    t = frame.length - 1
    ema20_series = ema(closes, cfg.ema_fast)
    ema50_series = ema(closes, cfg.ema_slow)
    atr_series = atr_wilder(frame.high, frame.low, closes, cfg.atr_period)
    rsi_series = rsi_wilder(closes, cfg.rsi_period)
    roc_1h_series = roc(closes, 1)
    roc_4h_series = roc(closes, 4)
    roc_24h_series = roc(closes, 24)
    vol_base = rolling_median_previous(frame.volume, cfg.vol_median_window)
    atr_value = atr_series[t]
    ema20_value = ema20_series[t]
    ema50_value = ema50_series[t]
    close = closes[t]
    current_vol_base = vol_base[t]
    vol_ratio = None if current_vol_base in (None, 0.0) else frame.volume[t] / current_vol_base
    atr_pct = None if atr_value is None or close == 0 else atr_value / close
    ema20_ext = None if atr_value in (None, 0.0) or ema20_value is None else (close - ema20_value) / atr_value
    range_expansion = range_expansion_last(frame.high, frame.low, cfg.s10_lookback_bars)
    price_zscore = zscore_last(closes, cfg.s11_zscore_window)
    all_patterns = detect_patterns(frame, ema20_series, ema50_series, atr_series, cfg)
    side_rows: dict[str, tuple[dict[str, GateResult], list[PatternHit], PlanPrimitives | None, CandidateQuality | None, list[str], float | None]] = {}
    for side in ("long", "short"):
        side_patterns = patterns_for_side(all_patterns, side)
        gates = evaluate_signal_gates(
            side,
            close=close,
            roc_4h=roc_4h_series[t],
            roc_24h=roc_24h_series[t],
            roc_1h_last=roc_1h_series[t],
            vol_ratio=vol_ratio,
            rsi=rsi_series[t],
            atr=atr_value,
            atr_pct=atr_pct,
            ema20=ema20_value,
            ema50=ema50_value,
            btc_roc_4h=roc_4h_series[t] if normalized == cfg.btc_symbol and btc_roc_4h is None else btc_roc_4h,
            funding=funding.get(normalized),
            patterns=side_patterns,
            cfg=cfg,
            range_expansion=range_expansion,
            zscore=price_zscore,
        )
        hard_pass = gates_pass(gates)
        plan = build_plan_primitives(side, side_patterns, frame, atr_value, cfg) if atr_value is not None and (hard_pass or _is_marginal_extension(side, gates, cfg)) else None
        quality: CandidateQuality | None = "hard" if hard_pass and plan is not None else None
        marginal_reasons: list[str] = []
        marginal_score = None
        if quality is None and plan is not None:
                marginal_reasons = _marginal_extension_reasons(side, gates, cfg, plan)
                if marginal_reasons:
                    quality = "marginal_extension"
                    marginal_score = _marginal_extension_score(gates, cfg)
        side_rows[side] = (gates, side_patterns, plan, quality, marginal_reasons, marginal_score)
    signal_side = _choose_signal_side(side_rows)
    signal_quality = side_rows[signal_side][3] if signal_side is not None else None
    signal_marginal_reasons = side_rows[signal_side][4] if signal_side is not None else []
    signal_marginal_score = side_rows[signal_side][5] if signal_side is not None else None
    candidate = None
    candidate_quality = None
    blocked_by: list[str] = []
    plan = None
    marginal_reasons: list[str] = []
    marginal_score = None
    cooldown_source_quality = None
    if signal_side is not None:
        last_candidate, last_quality = candidate_cooldown_entry(normalized, signal_side, state)
        if last_candidate is not None and int(as_of_ms) - int(last_candidate) < cfg.cooldown_candidate_ms:
            cooldown_source_quality = last_quality
        per_symbol_blocks, global_blocks = state_blocks(normalized, signal_side, state, as_of_ms, cfg, signal_quality=signal_quality)
        blocked_by = per_symbol_blocks
        if not per_symbol_blocks and not global_blocks:
            candidate = signal_side
            plan = side_rows[signal_side][2]
            candidate_quality = signal_quality
            marginal_reasons = side_rows[signal_side][4]
            marginal_score = side_rows[signal_side][5]
    best_side = signal_side or _least_failed_side(side_rows)
    failed = failed_gate_names(side_rows[best_side][0])
    return SymbolRow(
        symbol=normalized,
        status="ok",
        close=close,
        roc_4h=roc_4h_series[t],
        roc_24h=roc_24h_series[t],
        roc_1h_last=roc_1h_series[t],
        vol_ratio=vol_ratio,
        rsi=rsi_series[t],
        atr_pct=atr_pct,
        ema20_ext_atr=ema20_ext,
        above_ema50=None if ema50_value is None else close > ema50_value,
        funding=funding.get(normalized),
        patterns_long=[pattern.id for pattern in side_rows["long"][1]],
        patterns_short=[pattern.id for pattern in side_rows["short"][1]],
        gates_long={name: gate.passed for name, gate in side_rows["long"][0].items()},
        gates_short={name: gate.passed for name, gate in side_rows["short"][0].items()},
        gate_details_long=side_rows["long"][0],
        gate_details_short=side_rows["short"][0],
        signal_candidate_before_state=signal_side,
        signal_quality_before_state=signal_quality,
        signal_marginal_reasons_before_state=signal_marginal_reasons,
        signal_marginal_score_before_state=signal_marginal_score,
        cooldown_source_quality=cooldown_source_quality,
        candidate=candidate,
        candidate_quality=candidate_quality,
        marginal_reasons=marginal_reasons,
        marginal_score=marginal_score,
        failed_gates=failed,
        blocked_by=blocked_by,
        plan=plan,
    )


def _choose_signal_side(side_rows: dict[str, tuple[dict[str, GateResult], list[PatternHit], PlanPrimitives | None, CandidateQuality | None, list[str], float | None]]) -> CandidateSide | None:
    passed: list[CandidateSide] = []
    for side in ("long", "short"):
        if side_rows[side][3] == "hard" and side_rows[side][2] is not None:
            passed.append(side)
    if len(passed) == 1:
        return passed[0]
    if len(passed) == 2:
        long_roc = abs(float(side_rows["long"][0]["S1"].value or 0.0))
        short_roc = abs(float(side_rows["short"][0]["S1"].value or 0.0))
        return "long" if long_roc >= short_roc else "short"
    marginal: list[CandidateSide] = []
    for side in ("long", "short"):
        if side_rows[side][3] == "marginal_extension" and side_rows[side][2] is not None:
            marginal.append(side)
    if len(marginal) == 1:
        return marginal[0]
    if len(marginal) == 2:
        long_score = float(side_rows["long"][5] or 0.0)
        short_score = float(side_rows["short"][5] or 0.0)
        return "long" if long_score >= short_score else "short"
    return None


def _least_failed_side(side_rows: dict[str, tuple[dict[str, GateResult], list[PatternHit], PlanPrimitives | None, CandidateQuality | None, list[str], float | None]]) -> CandidateSide:
    long_failures = len(failed_gate_names(side_rows["long"][0]))
    short_failures = len(failed_gate_names(side_rows["short"][0]))
    return "long" if long_failures <= short_failures else "short"


def _is_marginal_extension(side: CandidateSide, gates: dict[str, GateResult], cfg: ScreenerConfig) -> bool:
    if not cfg.marginal_extension_enabled or side not in cfg.marginal_extension_sides:
        return False
    failed = failed_gate_names(gates)
    if not failed or not set(failed).issubset({"S9b", "S9c"}):
        return False
    s4 = gates.get("S4")
    s9a = gates.get("S9a")
    if s4 is None or s4.passed is not True or s9a is None or s9a.passed is not True:
        return False
    s9b = gates.get("S9b")
    s9c = gates.get("S9c")
    s9b_value = _gate_float(s9b)
    s9c_value = _gate_float(s9c)
    if s9b is not None and s9b.passed is False and (s9b_value is None or s9b_value > cfg.marginal_ext_atr_max):
        return False
    if s9c is not None and s9c.passed is False and (s9c_value is None or s9c_value > cfg.marginal_breakout_dist_atr_max):
        return False
    return True


def _marginal_extension_reasons(side: CandidateSide, gates: dict[str, GateResult], cfg: ScreenerConfig, plan: PlanPrimitives | None) -> list[str]:
    if not _is_marginal_extension(side, gates, cfg):
        return []
    if plan is None or plan.pattern_used not in set(cfg.marginal_extension_patterns):
        return []
    reasons = []
    s9b = gates.get("S9b")
    s9c = gates.get("S9c")
    s9b_value = _gate_float(s9b)
    s9c_value = _gate_float(s9c)
    if s9b is not None and s9b.passed is False and s9b_value is not None:
        reasons.append(f"S9b_extension={s9b_value:.2f}<=marginal_{cfg.marginal_ext_atr_max:.2f}")
    if s9c is not None and s9c.passed is False and s9c_value is not None:
        reasons.append(f"S9c_breakout={s9c_value:.2f}<=marginal_{cfg.marginal_breakout_dist_atr_max:.2f}")
    return reasons


def _marginal_extension_score(gates: dict[str, GateResult], cfg: ScreenerConfig) -> float:
    s9b_value = _gate_float(gates.get("S9b"))
    s9c_value = _gate_float(gates.get("S9c"))
    s9b_excess = 0.0 if s9b_value is None else max(0.0, s9b_value - cfg.ext_atr_max)
    s9c_excess = 0.0 if s9c_value is None else max(0.0, s9c_value - cfg.breakout_dist_atr_max)
    return -(s9b_excess + s9c_excess)


def _gate_float(gate: GateResult | None) -> float | None:
    if gate is None or gate.value is None:
        return None
    try:
        return float(gate.value)
    except (TypeError, ValueError):
        return None


def _apply_marginal_candidate_cap(rows: list[SymbolRow], cfg: ScreenerConfig) -> None:
    cap = int(cfg.marginal_extension_max_per_scan)
    marginal_rows = [row for row in rows if row.candidate in {"long", "short"} and row.candidate_quality == "marginal_extension"]
    if cap < 0 or len(marginal_rows) <= cap:
        return
    ranked = sorted(
        marginal_rows,
        key=lambda row: (-(row.marginal_score or 0.0), row.symbol, row.candidate or ""),
    )
    allowed = {id(row) for row in ranked[:cap]}
    for row in marginal_rows:
        if id(row) in allowed:
            continue
        row.blocked_by = [*row.blocked_by, "marginal_extension_cap"]
        row.candidate = None
        row.candidate_quality = None
        row.marginal_reasons = []
        row.marginal_score = None
        row.plan = None


def _score_candidates(rows: list[SymbolRow], cfg: ScreenerConfig) -> None:
    for row in rows:
        if row.candidate not in {"long", "short"}:
            continue
        details = row.gate_details_long if row.candidate == "long" else row.gate_details_short
        row.candidate_score = score_candidate(details, row.candidate_quality, cfg)


def _apply_candidate_rank_cap(rows: list[SymbolRow], cfg: ScreenerConfig) -> None:
    cap = int(cfg.max_candidates_per_scan)
    candidate_rows = [row for row in rows if row.candidate in {"long", "short"}]
    if cap < 0 or len(candidate_rows) <= cap:
        return
    ranked = sorted(candidate_rows, key=lambda row: (-(row.candidate_score or 0.0), row.symbol, row.candidate or ""))
    allowed = {id(row) for row in ranked[:cap]}
    for row in candidate_rows:
        if id(row) in allowed:
            continue
        row.blocked_by = [*row.blocked_by, "candidate_rank_cap"]
        row.candidate = None
        row.candidate_quality = None
        row.marginal_reasons = []
        row.marginal_score = None
        row.plan = None


def agent_row_view(row: SymbolRow) -> dict[str, Any]:
    """Row dump for agent-visible surfaces: scanner facts only, no reference plan or internal scores."""
    data = row.model_dump(mode="json")
    data.pop("plan", None)
    data.pop("candidate_score", None)
    data.pop("marginal_score", None)
    data.pop("signal_marginal_score_before_state", None)
    return data


def _split_symbols(symbols: list[str] | str) -> list[str]:
    if isinstance(symbols, str):
        values = symbols.split(",")
    else:
        values = symbols
    return [normalize_symbol(str(symbol)) for symbol in values if str(symbol).strip()]


def _load_optional_frame(cache_path: Path, symbol: str, interval_ms: int, as_of_ms: int, limit: int) -> CandleFrame | None:
    try:
        frame = load_closed(cache_path, symbol, interval_ms, as_of_ms, limit)
    except Exception:
        return None
    if validate_frame(frame, as_of_ms, limit) is not None:
        return None
    return frame


def _support_resistance_levels(frame: CandleFrame, close: float, atr_value: float | None, timeframe: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tolerance = _level_tolerance(close, atr_value)
    pivot_lows: list[tuple[float, int]] = []
    pivot_highs: list[tuple[float, int]] = []
    left = 2
    right = 2
    for index in range(left, frame.length - right):
        low = frame.low[index]
        high = frame.high[index]
        if low <= min(frame.low[index - left : index] + frame.low[index + 1 : index + right + 1]):
            pivot_lows.append((low, index))
        if high >= max(frame.high[index - left : index] + frame.high[index + 1 : index + right + 1]):
            pivot_highs.append((high, index))
    supports = _cluster_price_levels(pivot_lows, frame.length, close, tolerance, timeframe, is_support=True)
    resistances = _cluster_price_levels(pivot_highs, frame.length, close, tolerance, timeframe, is_support=False)
    return supports, resistances


def _level_tolerance(close: float, atr_value: float | None) -> float:
    pct_tolerance = abs(close) * 0.002
    if atr_value is None:
        return pct_tolerance
    return max(pct_tolerance, abs(atr_value) * 0.15)


def _cluster_price_levels(
    pivots: list[tuple[float, int]],
    frame_length: int,
    close: float,
    tolerance: float,
    timeframe: str,
    *,
    is_support: bool,
) -> list[dict[str, Any]]:
    if not pivots:
        return []
    clusters: list[dict[str, Any]] = []
    for price, index in sorted(pivots, key=lambda item: item[0]):
        for cluster in clusters:
            if abs(price - float(cluster["price"])) <= tolerance:
                touches = int(cluster["touches"]) + 1
                cluster["price"] = (float(cluster["price"]) * int(cluster["touches"]) + price) / touches
                cluster["touches"] = touches
                cluster["last_index"] = max(int(cluster["last_index"]), index)
                break
        else:
            clusters.append({"price": price, "touches": 1, "last_index": index})
    levels = []
    for cluster in clusters:
        price = float(cluster["price"])
        if is_support and price > close:
            continue
        if not is_support and price < close:
            continue
        levels.append(
            {
                "price": price,
                "touches": int(cluster["touches"]),
                "age_bars": frame_length - 1 - int(cluster["last_index"]),
                "distance_pct": None if close == 0 else abs(close - price) / abs(close),
                "timeframe": timeframe,
            }
        )
    return _nearest_levels(levels, close, is_support=is_support)


def _nearest_levels(levels: list[dict[str, Any]], close: float, *, is_support: bool) -> list[dict[str, Any]]:
    filtered = [level for level in levels if (float(level["price"]) <= close if is_support else float(level["price"]) >= close)]
    return sorted(filtered, key=lambda level: (float(level["distance_pct"] or 0.0), -int(level["touches"]), int(level["age_bars"])))[:5]


def _retest_seen(frame: CandleFrame, side: CandidateSide, boundary: float | None, atr_value: float | None, trigger_age_bars: int | None) -> bool | None:
    if boundary is None or atr_value is None:
        return None
    if trigger_age_bars is None:
        return None
    if trigger_age_bars <= 0:
        return False
    tolerance = 0.25 * abs(atr_value)
    t = frame.length - 1
    trigger_index = t - int(trigger_age_bars)
    start = trigger_index + 1
    if start > t:
        return False
    for index in range(start, t + 1):
        if side == "long" and frame.low[index] <= boundary + tolerance and frame.close[index] >= boundary:
            return True
        if side == "short" and frame.high[index] >= boundary - tolerance and frame.close[index] <= boundary:
            return True
    return False
