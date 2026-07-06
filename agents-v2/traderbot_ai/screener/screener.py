from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from traderbot_ai.screener.config import ScreenerConfig, config_hash
from traderbot_ai.screener.data import DataIssue, load_closed, validate_frame
from traderbot_ai.screener.gates import GateResult, evaluate_signal_gates, failed_gate_names, gates_pass, scan_global_blocks, state_blocks
from traderbot_ai.screener.indicators import atr_wilder, ema, roc, rolling_median_previous, rsi_wilder
from traderbot_ai.screener.market import DEFAULT_CACHE_PATH, INTERVAL_MS, normalize_symbol, parse_time_ms
from traderbot_ai.screener.patterns import PatternHit, detect_patterns, patterns_for_side
from traderbot_ai.screener.plan import PlanPrimitives, build_plan_primitives
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
    global_blocks: list[str] = scan_global_blocks(state, cfg)
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
            "row": row.model_dump(mode="json"),
            "error": issue.reason,
            "data_issue": {"reason": issue.reason, **issue.detail},
        }
    bars = []
    for index in range(max(0, frame.length - cfg.setup_digest_bars), frame.length):
        bars.append(
            {
                "i": index - (frame.length - 1),
                "o": frame.open[index],
                "h": frame.high[index],
                "l": frame.low[index],
                "c": frame.close[index],
                "v": frame.volume[index],
            }
        )
    return {
        "ok": row.status == "ok",
        "as_of_ms": result.as_of_ms,
        "symbol": row.symbol,
        "side": side,
        "row": row.model_dump(mode="json"),
        "recent_1h_csv": _bars_csv(bars),
        "range_20_high": _previous_window_max(frame.high, cfg.p1_lookback),
        "range_20_low": _previous_window_min(frame.low, cfg.p1_lookback),
        "range_48h_high": _previous_window_max(frame.high, cfg.p3_range_bars),
        "range_48h_low": _previous_window_min(frame.low, cfg.p3_range_bars),
        "last_3_high": _tail_max(frame.high, cfg.p2_pullback_bars),
        "last_3_low": _tail_min(frame.low, cfg.p2_pullback_bars),
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
        )
        hard_pass = gates_pass(gates)
        plan = build_plan_primitives(side, side_patterns, frame, atr_value, cfg) if atr_value is not None and (hard_pass or _is_marginal_extension(gates, cfg)) else None
        quality: CandidateQuality | None = "hard" if hard_pass and plan is not None else None
        marginal_reasons: list[str] = []
        marginal_score = None
        if quality is None and plan is not None:
                marginal_reasons = _marginal_extension_reasons(gates, cfg, plan)
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


def _is_marginal_extension(gates: dict[str, GateResult], cfg: ScreenerConfig) -> bool:
    if not cfg.marginal_extension_enabled:
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


def _marginal_extension_reasons(gates: dict[str, GateResult], cfg: ScreenerConfig, plan: PlanPrimitives | None) -> list[str]:
    if not _is_marginal_extension(gates, cfg):
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


def _split_symbols(symbols: list[str] | str) -> list[str]:
    if isinstance(symbols, str):
        values = symbols.split(",")
    else:
        values = symbols
    return [normalize_symbol(str(symbol)) for symbol in values if str(symbol).strip()]


def _bars_csv(bars: list[dict[str, Any]]) -> str:
    lines = ["i,o,h,l,c,v"]
    for bar in bars:
        lines.append(f"{bar['i']},{bar['o']:.8g},{bar['h']:.8g},{bar['l']:.8g},{bar['c']:.8g},{bar['v']:.8g}")
    return "\n".join(lines)


def _previous_window_max(values: list[float], window: int) -> float | None:
    if len(values) < window + 1:
        return None
    return max(values[-window - 1 : -1])


def _previous_window_min(values: list[float], window: int) -> float | None:
    if len(values) < window + 1:
        return None
    return min(values[-window - 1 : -1])


def _tail_max(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return max(values[-window:])


def _tail_min(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return min(values[-window:])
