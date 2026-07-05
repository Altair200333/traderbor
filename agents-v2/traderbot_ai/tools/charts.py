from __future__ import annotations

import json
import uuid
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import mplfinance as mpf
import pandas as pd
from agents import function_tool

from traderbot_ai.paths import ARTIFACTS_DIR, display_agents_path, ensure_runtime_dirs, safe_agents_path
from traderbot_ai.tools.market import fetch_candles, normalize_symbol


ChartKind = Literal["candle", "line"]

MAX_CANDLE_CHART_POINTS = 300
MAX_LINE_CHART_POINTS = 1200


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _error(message: str, **data) -> dict:
    return {"ok": False, "error": message, **data}


def _candles_to_df(candles: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df.rename(
        columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        },
        inplace=True,
    )
    return df


def _summary(candles: list[dict[str, float]]) -> dict:
    if not candles:
        return {"count": 0}
    return {
        "count": len(candles),
        "first_timestamp": int(candles[0]["t"]),
        "last_timestamp": int(candles[-1]["t"]),
        "open": candles[0]["o"],
        "latest_close": candles[-1]["c"],
        "high": max(candle["h"] for candle in candles),
        "low": min(candle["l"] for candle in candles),
        "volume_sum": sum(candle["v"] for candle in candles),
    }


def store_chart_file(
    symbol: str,
    interval: str = "15m",
    limit: int = 120,
    start_time: str | None = None,
    end_time: str | None = None,
    kind: ChartKind = "candle",
    volume: bool = True,
) -> dict:
    try:
        ensure_runtime_dirs()
        if kind not in {"candle", "line"}:
            return _error("kind must be candle or line", kind=kind)
        normalized = normalize_symbol(symbol)
        max_points = MAX_LINE_CHART_POINTS if kind == "line" else MAX_CANDLE_CHART_POINTS
        requested_limit = limit
        clamped_limit = max(1, min(limit, max_points))
        warnings = []
        if requested_limit != clamped_limit:
            warnings.append(f"limit clamped from {requested_limit} to {clamped_limit}")

        candles = fetch_candles(
            symbol=normalized,
            interval=interval,
            limit=clamped_limit,
            start_time=start_time,
            end_time=end_time,
        )
        if not candles:
            return _error("no candles returned", symbol=normalized, interval=interval)

        df = _candles_to_df(candles)
        artifact_id = uuid.uuid4().hex
        base = ARTIFACTS_DIR / f"chart-{normalized}-{interval}-{artifact_id}"
        png_path = base.with_suffix(".png")
        json_path = base.with_suffix(".json")
        data_path = ARTIFACTS_DIR / f"market-{normalized}-{interval}-{artifact_id}.json"

        mav = (20,) if kind == "candle" and len(candles) >= 20 else None
        plot_kwargs = {
            "type": kind,
            "volume": volume,
            "style": "yahoo",
            "title": f"{normalized} {interval}",
            "savefig": str(png_path),
        }
        if mav:
            plot_kwargs["mav"] = mav
        mpf.plot(df, **plot_kwargs)

        data_payload = {
            "schema_version": "market-candles/v1",
            "provider": "binance",
            "market": "spot",
            "symbol": normalized,
            "timeframe": interval,
            "time_unit": "ms",
            "volume_unit": "base_asset",
            "request": {
                "limit": requested_limit,
                "used_limit": clamped_limit,
                "start_time": start_time,
                "end_time": end_time,
            },
            "summary": _summary(candles),
            "candles": candles,
        }
        data_path.write_text(json.dumps(data_payload, indent=2, ensure_ascii=True), encoding="utf-8")

        metadata = {
            "schema_version": "chart/v1",
            "kind": kind,
            "provider": "binance",
            "market": "spot",
            "symbol": normalized,
            "timeframe": interval,
            "chart_path": display_agents_path(png_path),
            "metadata_path": display_agents_path(json_path),
            "data_artifact_path": display_agents_path(data_path),
            "render": {
                "type": kind,
                "volume": volume,
                "mav": [20] if mav else [],
                "style": "yahoo",
                "format": "png",
            },
            "summary": _summary(candles),
            "warnings": warnings,
        }
        json_path.write_text(json.dumps({"metadata": metadata}, indent=2, ensure_ascii=True), encoding="utf-8")

        return _ok(
            path=display_agents_path(png_path),
            absolute_path=str(png_path),
            bytes=png_path.stat().st_size,
            **metadata,
        )
    except Exception as error:
        return _error(str(error), symbol=normalize_symbol(symbol), interval=interval, kind=kind)


@function_tool
def store_chart(
    symbol: str,
    interval: str = "15m",
    limit: int = 120,
    start_time: str | None = None,
    end_time: str | None = None,
    kind: ChartKind = "candle",
    volume: bool = True,
) -> dict:
    """Save chart PNG. Args: symbol, interval, limit, kind candle or line."""
    return store_chart_file(
        symbol=symbol,
        interval=interval,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        kind=kind,
        volume=volume,
    )


@function_tool
def load_chart_metadata(path: str) -> dict:
    """Read chart metadata JSON under agents-v2."""
    try:
        file_path = safe_agents_path(path)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return _ok(path=display_agents_path(file_path), metadata=data.get("metadata", data))
    except Exception as error:
        return _error(str(error), path=path)


@function_tool
def list_stored_charts(max_items: int = 50) -> dict:
    """List saved chart metadata files."""
    try:
        ensure_runtime_dirs()
        limit = max(1, min(max_items, 200))
        items = []
        for path in sorted(ARTIFACTS_DIR.glob("chart-*.json"), reverse=True)[:limit]:
            try:
                metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
                items.append({"path": display_agents_path(path), **metadata})
            except Exception as error:
                items.append({"path": display_agents_path(path), "error": str(error)})
        return _ok(charts=items)
    except Exception as error:
        return _error(str(error))
