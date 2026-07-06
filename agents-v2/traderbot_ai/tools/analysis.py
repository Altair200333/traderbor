from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from agents import function_tool

from traderbot_ai.paths import ARTIFACTS_DIR, TMP_DIR, display_agents_path, ensure_runtime_dirs
from traderbot_ai.simulator.clock import active_simulation_clock_ms
from traderbot_ai.tools.market import fetch_candles, normalize_symbol
from traderbot_ai.tools.simulator import get_cached_candles_impl, get_candles_impl


AnalysisSource = Literal["auto", "cache", "live"]

DEFAULT_INDICATORS = "strategy"
MAX_TAIL_ROWS = 20
MAX_CODE_CHARS = 12000
MAX_CODE_SECONDS = 120


def compute_indicators_impl(
    symbol: str,
    interval: str = "1h",
    as_of: str | int | float | None = None,
    limit: int = 170,
    indicators: str = DEFAULT_INDICATORS,
    source: AnalysisSource = "auto",
    tail: int = 5,
    render_chart: bool = False,
) -> dict[str, Any]:
    """Load candles and compute a broad set of lookahead-safe indicators."""
    try:
        loaded = _load_candles(symbol=symbol, interval=interval, as_of=as_of, limit=limit, source=source)
        if loaded.get("ok") is not True:
            return loaded
        candles = loaded["candles"]
        if not candles:
            return _error("no candles available", symbol=normalize_symbol(symbol), interval=interval)
        df = _candles_to_df(candles)
        requested = _requested_indicators(indicators)
        computed = _compute_indicator_frame(df, requested)
        artifact = _write_indicator_artifact(loaded, computed, requested)
        chart = _render_indicator_chart(loaded, computed, requested) if render_chart else {}
        tail_count = max(1, min(int(tail), MAX_TAIL_ROWS))
        latest = _json_value(computed.iloc[-1].to_dict())
        tail_rows = _json_value(computed.tail(tail_count).to_dict(orient="records"))
        return {
            "ok": True,
            "schema_version": "analysis-indicators/v1",
            "source": loaded["source"],
            "symbol": loaded["symbol"],
            "interval": loaded["interval"],
            "as_of_ms": loaded.get("as_of_ms"),
            "closed_only": loaded.get("closed_only", True),
            "candle_summary": loaded["summary"],
            "indicators": sorted(set(computed.columns) - {"t", "o", "h", "l", "c", "v"}),
            "latest": latest,
            "tail": tail_rows,
            "artifact_path": artifact.get("artifact_path"),
            "chart_path": chart.get("chart_path"),
            "warnings": [*loaded.get("warnings", []), *artifact.get("warnings", []), *chart.get("warnings", [])],
        }
    except Exception as error:
        return _error(str(error), symbol=symbol, interval=interval)


def run_analysis_code_impl(
    code: str,
    symbol: str,
    interval: str = "1h",
    as_of: str | int | float | None = None,
    limit: int = 170,
    source: AnalysisSource = "auto",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run agent-supplied Python against a guarded candle DataFrame."""
    try:
        if len(code) > MAX_CODE_CHARS:
            return _error(f"code is too large; max {MAX_CODE_CHARS} chars")
        loaded = _load_candles(symbol=symbol, interval=interval, as_of=as_of, limit=limit, source=source)
        if loaded.get("ok") is not True:
            return loaded
        ensure_runtime_dirs()
        run_id = uuid.uuid4().hex
        artifact_dir = ARTIFACTS_DIR / f"analysis-code-{loaded['symbol']}-{loaded['interval']}-{run_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        input_path = artifact_dir / "input.json"
        result_path = artifact_dir / "result.json"
        script_path = TMP_DIR / f"analysis-code-{run_id}.py"
        input_path.write_text(json.dumps({"candles": loaded["candles"], "meta": _json_value(loaded)}, ensure_ascii=True), encoding="utf-8")
        script_path.write_text(_analysis_script(input_path, result_path, artifact_dir, code), encoding="utf-8")
        timeout = max(1, min(int(timeout_seconds), MAX_CODE_SECONDS))
        env = _analysis_subprocess_env()
        completed = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            cwd=str(artifact_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        result_payload: dict[str, Any] = {}
        if result_path.exists():
            try:
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                result_payload = {"result_error": f"invalid result json: {error}"}
        return {
            "ok": completed.returncode == 0,
            "schema_version": "analysis-code/v1",
            "source": loaded["source"],
            "symbol": loaded["symbol"],
            "interval": loaded["interval"],
            "as_of_ms": loaded.get("as_of_ms"),
            "closed_only": loaded.get("closed_only", True),
            "candle_summary": loaded["summary"],
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
            "result": result_payload.get("result"),
            "result_error": result_payload.get("result_error"),
            "artifact_dir": display_agents_path(artifact_dir),
            "script_path": display_agents_path(script_path),
            "input_path": display_agents_path(input_path),
            "result_path": display_agents_path(result_path),
        }
    except subprocess.TimeoutExpired as error:
        return _error(
            "analysis code timed out",
            stdout=(error.stdout or "")[-12000:] if isinstance(error.stdout, str) else "",
            stderr=(error.stderr or "")[-12000:] if isinstance(error.stderr, str) else "Timed out",
        )
    except Exception as error:
        return _error(str(error), symbol=symbol, interval=interval)


@function_tool
def compute_indicators(
    symbol: str,
    interval: str = "1h",
    as_of: str | int | float | None = None,
    limit: int = 170,
    indicators: str = DEFAULT_INDICATORS,
    source: AnalysisSource = "auto",
    tail: int = 5,
    render_chart: bool = False,
) -> dict[str, Any]:
    """Load candles and compute indicators. indicators: strategy, all, or CSV names."""
    return compute_indicators_impl(
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        limit=limit,
        indicators=indicators,
        source=source,
        tail=tail,
        render_chart=render_chart,
    )


@function_tool
def run_analysis_code(
    code: str,
    symbol: str,
    interval: str = "1h",
    as_of: str | int | float | None = None,
    limit: int = 170,
    source: AnalysisSource = "auto",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run Python on guarded candle data. Code receives df, candles, pd, math, artifact_dir; set result."""
    return run_analysis_code_impl(
        code=code,
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        limit=limit,
        source=source,
        timeout_seconds=timeout_seconds,
    )


def _load_candles(symbol: str, interval: str, as_of: str | int | float | None, limit: int, source: AnalysisSource) -> dict[str, Any]:
    if source not in {"auto", "cache", "live"}:
        return _error("source must be auto, cache, or live", source=source)
    if source == "live" and _cache_context_active():
        return _error("source=live is not allowed while a cache or simulation context is active; use source=auto or source=cache")
    normalized = normalize_symbol(symbol)
    requested_limit = max(1, int(limit))
    use_cache = source == "cache" or (source == "auto" and _cache_context_active())
    if use_cache:
        result = get_candles_impl(symbol=normalized, interval=interval, as_of=as_of, limit=requested_limit)
        if result.get("ok") is not True and interval == "4h" and result.get("error") == "stale cached candle data":
            aggregated = _load_cached_4h_from_1h(normalized, as_of, requested_limit)
            if aggregated.get("ok") is True:
                return aggregated
        if result.get("ok") is not True:
            return result
        candles = result.get("candles") or []
        return {
            "ok": True,
            "source": "local_cache",
            "symbol": result.get("symbol") or normalized,
            "interval": result.get("interval") or interval,
            "as_of_ms": result.get("as_of_ms"),
            "closed_only": True,
            "candles": candles,
            "summary": result.get("summary") or _candle_summary(candles),
            "warnings": [],
        }
    candles = fetch_candles(symbol=normalized, interval=interval, limit=requested_limit, end_time=as_of)
    return {
        "ok": True,
        "source": "live_binance",
        "symbol": normalized,
        "interval": interval,
        "as_of_ms": None,
        "closed_only": True,
        "candles": candles,
        "summary": _candle_summary(candles),
        "warnings": [],
    }


def _load_cached_4h_from_1h(symbol: str, as_of: str | int | float | None, limit: int) -> dict[str, Any]:
    one_hour_lookback = max(4, int(limit) * 4 + 3)
    result = get_cached_candles_impl(symbol=symbol, interval="1h", as_of=as_of, lookback=one_hour_lookback)
    if result.get("ok") is not True:
        return result
    candles = _aggregate_1h_to_4h(result.get("candles") or [])
    if not candles:
        return _error(
            "no complete 4h candles available from cached 1h data",
            symbol=symbol,
            interval="4h",
            source="local_cache",
            as_of_ms=result.get("as_of_ms"),
            closed_only=True,
        )
    candles = candles[-max(1, int(limit)) :]
    return {
        "ok": True,
        "source": "local_cache",
        "symbol": symbol,
        "interval": "4h",
        "as_of_ms": result.get("as_of_ms"),
        "closed_only": True,
        "candles": candles,
        "summary": _candle_summary(candles),
        "warnings": ["4h candles aggregated from cached 1h candles"],
    }


def _aggregate_1h_to_4h(candles: list[dict[str, Any]]) -> list[dict[str, float]]:
    one_hour_ms = 60 * 60 * 1000
    four_hours_ms = 4 * one_hour_ms
    buckets: dict[int, list[dict[str, Any]]] = {}
    for candle in sorted(candles, key=lambda item: int(float(item["t"]))):
        timestamp = int(float(candle["t"]))
        bucket_start = (timestamp // four_hours_ms) * four_hours_ms
        buckets.setdefault(bucket_start, []).append(candle)

    aggregated: list[dict[str, float]] = []
    for bucket_start in sorted(buckets):
        bucket = sorted(buckets[bucket_start], key=lambda item: int(float(item["t"])))
        timestamps = [int(float(candle["t"])) for candle in bucket]
        if timestamps != [bucket_start + index * one_hour_ms for index in range(4)]:
            continue
        aggregated.append(
            {
                "t": float(bucket_start),
                "o": float(bucket[0]["o"]),
                "h": max(float(candle["h"]) for candle in bucket),
                "l": min(float(candle["l"]) for candle in bucket),
                "c": float(bucket[-1]["c"]),
                "v": sum(float(candle["v"]) for candle in bucket),
            }
        )
    return aggregated


def _cache_context_active() -> bool:
    return bool(
        os.getenv("TRADERBOT_MARKET_CACHE_PATH")
        or os.getenv("TRADERBOT_EXCHANGE_STATE_PATH")
        or os.getenv("TRADERBOT_SCREENER_MODE") == "deterministic"
        or active_simulation_clock_ms() is not None
    )


def _candles_to_df(candles: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    for column in ("t", "o", "h", "l", "c", "v"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[["t", "o", "h", "l", "c", "v"]].copy()


def _requested_indicators(value: str) -> set[str]:
    text = (value or DEFAULT_INDICATORS).strip().lower()
    if text in {"", "strategy"}:
        return {
            "price",
            "sma",
            "ema",
            "rma",
            "roc",
            "rsi",
            "atr",
            "macd",
            "bbands",
            "donchian",
            "vwap",
            "obv",
            "volume",
            "adx",
            "stoch",
            "williams",
            "cci",
        }
    if text == "all":
        return {
            "price",
            "sma",
            "ema",
            "wma",
            "rma",
            "roc",
            "momentum",
            "rsi",
            "atr",
            "macd",
            "bbands",
            "keltner",
            "donchian",
            "vwap",
            "obv",
            "mfi",
            "cmf",
            "volume",
            "adx",
            "stoch",
            "stochrsi",
            "williams",
            "cci",
            "aroon",
            "bop",
            "ichimoku",
            "supertrend",
            "psar",
        }
    aliases = {
        "bollinger": "bbands",
        "bollinger_bands": "bbands",
        "willr": "williams",
        "williams_r": "williams",
        "tr": "atr",
        "natr": "atr",
        "di": "adx",
        "dmi": "adx",
    }
    return {aliases.get(part.strip(), part.strip()) for part in text.split(",") if part.strip()}


def _compute_indicator_frame(df: pd.DataFrame, requested: set[str]) -> pd.DataFrame:
    out = df.copy()
    high = out["h"]
    low = out["l"]
    close = out["c"]
    volume = out["v"]
    if "price" in requested:
        out["hl2"] = (high + low) / 2.0
        out["hlc3"] = (high + low + close) / 3.0
        out["ohlc4"] = (out["o"] + high + low + close) / 4.0
        out["weighted_close"] = (high + low + close * 2.0) / 4.0
    if "sma" in requested:
        for period in (9, 20, 50, 100, 200):
            out[f"sma_{period}"] = close.rolling(period, min_periods=period).mean()
    if "ema" in requested:
        for period in (9, 20, 50, 100, 200):
            out[f"ema_{period}"] = _ema(close, period)
    if "wma" in requested:
        for period in (20, 50):
            out[f"wma_{period}"] = _wma(close, period)
    if "rma" in requested:
        out["rma_14"] = _rma(close, 14)
    if "roc" in requested:
        for period in (1, 4, 12, 24):
            out[f"roc_{period}"] = close / close.shift(period) - 1.0
    if "momentum" in requested:
        for period in (4, 10, 24):
            out[f"momentum_{period}"] = close - close.shift(period)
    if "rsi" in requested:
        out["rsi_14"] = _rsi(close, 14)
    if "atr" in requested:
        tr = _true_range(high, low, close)
        atr = _atr_wilder(high, low, close, 14)
        out["true_range"] = tr
        out["atr_14"] = atr
        out["natr_14"] = atr / close * 100.0
    if "macd" in requested:
        macd = _ema(close, 12) - _ema(close, 26)
        signal = _ema(macd, 9)
        out["macd_12_26_9"] = macd
        out["macd_signal_12_26_9"] = signal
        out["macd_hist_12_26_9"] = macd - signal
    if "bbands" in requested:
        mid = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std(ddof=0)
        upper = mid + 2.0 * std
        lower = mid - 2.0 * std
        out["bb_mid_20_2"] = mid
        out["bb_upper_20_2"] = upper
        out["bb_lower_20_2"] = lower
        out["bb_bandwidth_20_2"] = (upper - lower) / mid
        out["bb_percent_b_20_2"] = (close - lower) / (upper - lower)
    if "keltner" in requested:
        mid = _ema(close, 20)
        atr = _atr_wilder(high, low, close, 14)
        out["keltner_mid_20"] = mid
        out["keltner_upper_20_2"] = mid + 2.0 * atr
        out["keltner_lower_20_2"] = mid - 2.0 * atr
    if "donchian" in requested:
        for period in (20, 48):
            upper = high.rolling(period, min_periods=period).max()
            lower = low.rolling(period, min_periods=period).min()
            out[f"donchian_upper_{period}"] = upper
            out[f"donchian_lower_{period}"] = lower
            out[f"donchian_mid_{period}"] = (upper + lower) / 2.0
    if "vwap" in requested:
        typical = (high + low + close) / 3.0
        pv = typical * volume
        out["vwap_window"] = pv.cumsum() / volume.replace(0, pd.NA).cumsum()
        out["vwap_20"] = pv.rolling(20, min_periods=20).sum() / volume.rolling(20, min_periods=20).sum()
    if "obv" in requested:
        out["obv"] = _obv(close, volume)
    if "mfi" in requested:
        out["mfi_14"] = _mfi(high, low, close, volume, 14)
    if "cmf" in requested:
        out["cmf_20"] = _cmf(high, low, close, volume, 20)
    if "volume" in requested:
        out["volume_sma_20"] = volume.rolling(20, min_periods=20).mean()
        out["volume_ratio_20"] = volume / out["volume_sma_20"]
    if "adx" in requested:
        adx = _adx(high, low, close, 14)
        out["plus_di_14"] = adx["plus_di"]
        out["minus_di_14"] = adx["minus_di"]
        out["dx_14"] = adx["dx"]
        out["adx_14"] = adx["adx"]
    if "stoch" in requested:
        highest = high.rolling(14, min_periods=14).max()
        lowest = low.rolling(14, min_periods=14).min()
        k = (close - lowest) / (highest - lowest) * 100.0
        out["stoch_k_14"] = k
        out["stoch_d_14_3"] = k.rolling(3, min_periods=3).mean()
    if "stochrsi" in requested:
        rsi = _rsi(close, 14)
        lowest = rsi.rolling(14, min_periods=14).min()
        highest = rsi.rolling(14, min_periods=14).max()
        k = (rsi - lowest) / (highest - lowest) * 100.0
        out["stochrsi_k_14_3"] = k.rolling(3, min_periods=3).mean()
        out["stochrsi_d_14_3_3"] = out["stochrsi_k_14_3"].rolling(3, min_periods=3).mean()
    if "williams" in requested:
        highest = high.rolling(14, min_periods=14).max()
        lowest = low.rolling(14, min_periods=14).min()
        out["willr_14"] = -100.0 * (highest - close) / (highest - lowest)
    if "cci" in requested:
        typical = (high + low + close) / 3.0
        mid = typical.rolling(20, min_periods=20).mean()
        mad = typical.rolling(20, min_periods=20).apply(lambda values: float(abs(values - values.mean()).mean()), raw=False)
        out["cci_20"] = (typical - mid) / (0.015 * mad)
    if "aroon" in requested:
        up, down = _aroon(high, low, 25)
        out["aroon_up_25"] = up
        out["aroon_down_25"] = down
        out["aroon_osc_25"] = up - down
    if "bop" in requested:
        out["bop"] = (close - out["o"]) / (high - low)
    if "ichimoku" in requested:
        conversion = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2.0
        base = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2.0
        span_b = (high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2.0
        out["ichimoku_tenkan_9"] = conversion
        out["ichimoku_kijun_26"] = base
        out["ichimoku_span_a_current"] = (conversion + base) / 2.0
        out["ichimoku_span_b_current"] = span_b
    if "supertrend" in requested:
        st = _supertrend(high, low, close, 10, 3.0)
        out["supertrend_10_3"] = st["line"]
        out["supertrend_dir_10_3"] = st["direction"]
    if "psar" in requested:
        out["psar_0.02_0.2"] = _psar(high, low, close, 0.02, 0.2)
    return out


def _ema(values: pd.Series, period: int) -> pd.Series:
    return _recursive_average(values, period, alpha=2.0 / (period + 1.0))


def _rma(values: pd.Series, period: int) -> pd.Series:
    return _recursive_average(values, period, alpha=1.0 / period)


def _recursive_average(values: pd.Series, period: int, alpha: float) -> pd.Series:
    result = [math.nan] * len(values)
    finite: list[float] = []
    for index, raw in enumerate(values.tolist()):
        value = float(raw) if raw is not None else math.nan
        if math.isfinite(value):
            finite.append(value)
        else:
            finite = []
        if len(finite) == period:
            result[index] = sum(finite) / period
        elif len(finite) > period:
            previous = result[index - 1]
            result[index] = value * alpha + previous * (1.0 - alpha) if math.isfinite(previous) else math.nan
    return pd.Series(result, index=values.index, dtype="float64")


def _wma(values: pd.Series, period: int) -> pd.Series:
    weights = list(range(1, period + 1))
    denom = sum(weights)
    return values.rolling(period, min_periods=period).apply(lambda raw: float(sum(value * weight for value, weight in zip(raw, weights)) / denom), raw=True)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(avg_loss != 0, 100.0)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    return pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    result = [math.nan] * len(close)
    if len(close) <= period:
        return pd.Series(result, index=close.index, dtype="float64")
    values = tr.tolist()
    first = [float(value) for value in values[1 : period + 1]]
    if not all(math.isfinite(value) for value in first):
        return pd.Series(result, index=close.index, dtype="float64")
    atr = sum(first) / period
    result[period] = atr
    for index in range(period + 1, len(values)):
        value = float(values[index]) if values[index] is not None else math.nan
        if math.isfinite(value):
            atr = (atr * (period - 1) + value) / period
            result[index] = atr
    return pd.Series(result, index=close.index, dtype="float64")


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series([0.0] * len(high), index=high.index)
    minus_dm = pd.Series([0.0] * len(high), index=high.index)
    plus_dm[(up_move > down_move) & (up_move > 0.0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0.0)] = down_move
    tr_smooth = _wilder_sum(_true_range(high, low, close), period)
    plus_smooth = _wilder_sum(plus_dm, period)
    minus_smooth = _wilder_sum(minus_dm, period)
    plus_di = 100.0 * plus_smooth / tr_smooth
    minus_di = 100.0 * minus_smooth / tr_smooth
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([math.inf, -math.inf], math.nan)
    adx = _rma(dx.dropna().reindex(dx.index), period)
    return {"plus_di": plus_di, "minus_di": minus_di, "dx": dx, "adx": adx}


def _wilder_sum(values: pd.Series, period: int) -> pd.Series:
    result = [math.nan] * len(values)
    vals = values.tolist()
    for index in range(len(vals)):
        value = float(vals[index]) if vals[index] is not None else math.nan
        if index == period:
            sample = [float(item) for item in vals[1 : period + 1]]
            result[index] = sum(sample) if all(math.isfinite(item) for item in sample) else math.nan
        elif index > period:
            previous = result[index - 1]
            result[index] = previous - previous / period + value if math.isfinite(previous) and math.isfinite(value) else math.nan
    return pd.Series(result, index=values.index, dtype="float64")


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    result = [0.0] * len(close)
    for index in range(1, len(close)):
        if close.iloc[index] > close.iloc[index - 1]:
            result[index] = result[index - 1] + float(volume.iloc[index])
        elif close.iloc[index] < close.iloc[index - 1]:
            result[index] = result[index - 1] - float(volume.iloc[index])
        else:
            result[index] = result[index - 1]
    return pd.Series(result, index=close.index, dtype="float64")


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    positive = raw_flow.where(typical > typical.shift(1), 0.0)
    negative = raw_flow.where(typical < typical.shift(1), 0.0)
    pos_sum = positive.rolling(period, min_periods=period).sum()
    neg_sum = negative.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum
    return (100.0 - 100.0 / (1.0 + ratio)).where(neg_sum != 0, 100.0)


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    denominator = high - low
    multiplier = ((close - low) - (high - close)) / denominator
    multiplier = multiplier.where(denominator != 0, 0.0)
    flow = multiplier * volume
    return flow.rolling(period, min_periods=period).sum() / volume.rolling(period, min_periods=period).sum()


def _aroon(high: pd.Series, low: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    def up_func(values: pd.Series) -> float:
        raw = values.to_numpy()
        peak = max(raw)
        position = max(index for index, value in enumerate(raw) if value == peak)
        periods_since = len(values) - 1 - position
        return 100.0 * (period - periods_since) / period

    def down_func(values: pd.Series) -> float:
        raw = values.to_numpy()
        trough = min(raw)
        position = max(index for index, value in enumerate(raw) if value == trough)
        periods_since = len(values) - 1 - position
        return 100.0 * (period - periods_since) / period

    window = period + 1
    return (
        high.rolling(window, min_periods=window).apply(up_func, raw=False),
        low.rolling(window, min_periods=window).apply(down_func, raw=False),
    )


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int, multiplier: float) -> dict[str, pd.Series]:
    atr = _atr_wilder(high, low, close, period)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper = [math.nan] * len(close)
    final_lower = [math.nan] * len(close)
    line = [math.nan] * len(close)
    direction = [0.0] * len(close)
    for index in range(len(close)):
        if not math.isfinite(float(basic_upper.iloc[index])) or index == 0:
            continue
        previous_upper = final_upper[index - 1]
        previous_lower = final_lower[index - 1]
        prev_close = float(close.iloc[index - 1])
        final_upper[index] = (
            float(basic_upper.iloc[index])
            if not math.isfinite(previous_upper) or basic_upper.iloc[index] < previous_upper or prev_close > previous_upper
            else previous_upper
        )
        final_lower[index] = (
            float(basic_lower.iloc[index])
            if not math.isfinite(previous_lower) or basic_lower.iloc[index] > previous_lower or prev_close < previous_lower
            else previous_lower
        )
        previous_line = line[index - 1]
        current_close = float(close.iloc[index])
        if not math.isfinite(previous_line):
            direction[index] = 1.0 if current_close >= final_lower[index] else -1.0
        elif previous_line == previous_upper:
            direction[index] = 1.0 if current_close > final_upper[index] else -1.0
        else:
            direction[index] = -1.0 if current_close < final_lower[index] else 1.0
        line[index] = final_lower[index] if direction[index] > 0 else final_upper[index]
    return {
        "line": pd.Series(line, index=close.index, dtype="float64"),
        "direction": pd.Series(direction, index=close.index, dtype="float64"),
    }


def _psar(high: pd.Series, low: pd.Series, close: pd.Series, step: float, max_step: float) -> pd.Series:
    if len(high) == 0:
        return pd.Series(dtype="float64")
    sar = [math.nan] * len(high)
    long = True if len(close) == 1 else float(close.iloc[1]) >= float(close.iloc[0])
    af = step
    ep = float(high.iloc[0]) if long else float(low.iloc[0])
    sar[0] = float(low.iloc[0]) if long else float(high.iloc[0])
    for index in range(1, len(high)):
        previous = sar[index - 1]
        sar[index] = previous + af * (ep - previous)
        if long:
            sar[index] = min(sar[index], float(low.iloc[index - 1]))
            if index > 1:
                sar[index] = min(sar[index], float(low.iloc[index - 2]))
            if low.iloc[index] < sar[index]:
                long = False
                sar[index] = ep
                ep = float(low.iloc[index])
                af = step
            elif high.iloc[index] > ep:
                ep = float(high.iloc[index])
                af = min(af + step, max_step)
        else:
            sar[index] = max(sar[index], float(high.iloc[index - 1]))
            if index > 1:
                sar[index] = max(sar[index], float(high.iloc[index - 2]))
            if high.iloc[index] > sar[index]:
                long = True
                sar[index] = ep
                ep = float(high.iloc[index])
                af = step
            elif low.iloc[index] < ep:
                ep = float(low.iloc[index])
                af = min(af + step, max_step)
    return pd.Series(sar, index=high.index, dtype="float64")


def _write_indicator_artifact(loaded: dict[str, Any], computed: pd.DataFrame, requested: set[str]) -> dict[str, Any]:
    ensure_runtime_dirs()
    artifact_id = uuid.uuid4().hex
    path = ARTIFACTS_DIR / f"analysis-indicators-{loaded['symbol']}-{loaded['interval']}-{artifact_id}.json"
    payload = {
        "schema_version": "analysis-indicators/v1",
        "source": loaded["source"],
        "symbol": loaded["symbol"],
        "interval": loaded["interval"],
        "as_of_ms": loaded.get("as_of_ms"),
        "requested": sorted(requested),
        "summary": loaded["summary"],
        "rows": _json_value(computed.to_dict(orient="records")),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return {"artifact_path": display_agents_path(path), "warnings": []}


def _render_indicator_chart(loaded: dict[str, Any], computed: pd.DataFrame, requested: set[str]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ensure_runtime_dirs()
        artifact_id = uuid.uuid4().hex
        path = ARTIFACTS_DIR / f"analysis-chart-{loaded['symbol']}-{loaded['interval']}-{artifact_id}.png"
        times = pd.to_datetime(computed["t"], unit="ms", utc=True)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
        axes[0].plot(times, computed["c"], label="close", linewidth=1.3)
        for column in ("ema_20", "ema_50", "sma_20", "bb_upper_20_2", "bb_lower_20_2"):
            if column in computed:
                axes[0].plot(times, computed[column], label=column, linewidth=0.9)
        axes[0].legend(loc="best", fontsize=8)
        axes[0].set_title(f"{loaded['symbol']} {loaded['interval']} indicators")
        axes[1].bar(times, computed["v"], label="volume", width=0.02)
        axes[1].legend(loc="best", fontsize=8)
        if "rsi_14" in computed:
            axes[2].plot(times, computed["rsi_14"], label="rsi_14", linewidth=1.0)
            axes[2].axhline(70, color="red", linewidth=0.6)
            axes[2].axhline(30, color="green", linewidth=0.6)
        elif "macd_12_26_9" in computed:
            axes[2].plot(times, computed["macd_12_26_9"], label="macd", linewidth=1.0)
            axes[2].plot(times, computed["macd_signal_12_26_9"], label="signal", linewidth=1.0)
        axes[2].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return {"chart_path": display_agents_path(path), "warnings": []}
    except Exception as error:
        return {"warnings": [f"chart_render_failed:{error}"]}


def _analysis_script(input_path: Path, result_path: Path, artifact_dir: Path, code: str) -> str:
    indented = textwrap.indent(code.strip(), "    ")
    return f"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pandas as pd

payload = json.loads(Path({str(input_path)!r}).read_text(encoding="utf-8"))
candles = payload["candles"]
meta = payload["meta"]
df = pd.DataFrame(candles)
for column in ("t", "o", "h", "l", "c", "v"):
    df[column] = pd.to_numeric(df[column], errors="coerce")
artifact_dir = Path({str(artifact_dir)!r})
result = None

def _agent_main():
    global result
{indented if indented else "    pass"}

try:
    _agent_main()
    Path({str(result_path)!r}).write_text(json.dumps({{"result": result}}, ensure_ascii=True, default=str), encoding="utf-8")
except Exception as error:
    Path({str(result_path)!r}).write_text(json.dumps({{"result_error": str(error)}}, ensure_ascii=True), encoding="utf-8")
    raise
"""


def _analysis_subprocess_env() -> dict[str, str]:
    allowed = (
        "COMSPEC",
        "NUMBER_OF_PROCESSORS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _candle_summary(candles: list[dict[str, Any]]) -> dict[str, Any]:
    if not candles:
        return {"count": 0}
    return {
        "count": len(candles),
        "first_timestamp": int(candles[0]["t"]),
        "last_timestamp": int(candles[-1]["t"]),
        "open": float(candles[0]["o"]),
        "latest_close": float(candles[-1]["c"]),
        "high": max(float(candle["h"]) for candle in candles),
        "low": min(float(candle["l"]) for candle in candles),
        "volume_sum": sum(float(candle["v"]) for candle in candles),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _error(message: str, **data: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **data}
