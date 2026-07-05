from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_INT_FIELD = re.compile(r"(?P<name>input_tokens|cached_tokens|output_tokens|total_tokens)=(?P<value>\d+)")


@dataclass(frozen=True)
class UsageTotals:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)

    @property
    def cache_hit_ratio(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens


def usage_totals(records: list[Any]) -> UsageTotals:
    totals = UsageTotals()
    for record in records:
        for usage in _usage_entries(record):
            parsed = _parse_usage_entry(usage)
            totals = UsageTotals(
                input_tokens=totals.input_tokens + parsed.input_tokens,
                cached_input_tokens=totals.cached_input_tokens + parsed.cached_input_tokens,
                output_tokens=totals.output_tokens + parsed.output_tokens,
                total_tokens=totals.total_tokens + parsed.total_tokens,
            )
    return totals


def _usage_entries(record: Any) -> list[Any]:
    if isinstance(record, dict) and isinstance(record.get("usage"), list):
        return list(record["usage"])
    return [record]


def _parse_usage_entry(entry: Any) -> UsageTotals:
    if isinstance(entry, dict):
        input_tokens = int(entry.get("input_tokens") or 0)
        cached = entry.get("cached_input_tokens")
        if cached is None:
            details = entry.get("input_tokens_details")
            if isinstance(details, dict):
                cached = details.get("cached_tokens")
        return UsageTotals(
            input_tokens=input_tokens,
            cached_input_tokens=int(cached or 0),
            output_tokens=int(entry.get("output_tokens") or 0),
            total_tokens=int(entry.get("total_tokens") or 0),
        )
    if isinstance(entry, str):
        fields = {match.group("name"): int(match.group("value")) for match in _INT_FIELD.finditer(entry)}
        return UsageTotals(
            input_tokens=fields.get("input_tokens", 0),
            cached_input_tokens=fields.get("cached_tokens", 0),
            output_tokens=fields.get("output_tokens", 0),
            total_tokens=fields.get("total_tokens", 0),
        )
    return UsageTotals()
