from __future__ import annotations

import json

from traderbot_ai.screener.screener import ScanResult, SymbolRow


def to_canonical_json(result: ScanResult) -> str:
    return json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def to_markdown_table(result: ScanResult) -> str:
    lines = [
        f"as_of={result.as_of_iso} config={result.config_hash} btc_roc4h={_pct(result.btc_roc_4h)}",
        "| sym | side | close | roc4h | roc24h | vol | rsi | atr% | ext | pat | fail/block |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in result.symbols:
        lines.append(_row(row))
    return "\n".join(lines)


def _row(row: SymbolRow) -> str:
    marker = "*" if row.candidate else ""
    side = row.candidate or row.signal_candidate_before_state or "-"
    quality = f":{row.candidate_quality}" if row.candidate_quality else ""
    patterns = ",".join(row.patterns_long if side == "long" else row.patterns_short if side == "short" else row.patterns_long + row.patterns_short) or "-"
    fail = ",".join(row.blocked_by or row.failed_gates or ([row.data_issue.get("reason", row.status)] if row.data_issue else [])) or "-"
    return (
        f"| {marker}{row.symbol} | {side}{quality} | {_num(row.close)} | {_pct(row.roc_4h)} | {_pct(row.roc_24h)} | "
        f"{_num(row.vol_ratio, 1)} | {_num(row.rsi, 0)} | {_pct(row.atr_pct)} | {_ext(row.ema20_ext_atr)} | {patterns} | {fail} |"
    )


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.2f}%"


def _ext(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}atr"
