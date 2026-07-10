"""Trading universe: frozen 149-pair list + Bybit instrument metadata +
$1M liquidity gate (30d-median daily quote volume) + delisting watch.

The pair list is the frozen research universe (bot/artifacts/universe.json,
generated from docs/notes/2026-07-07/bybit-trading-universe.md). Each pair
maps to a Bybit linear symbol (scaled-contract overrides live in the
artifact); pairs missing from instruments-info are excluded with an alarm.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from bot.config import ARTIFACTS_DIR
from bot.journal import Journal

log = logging.getLogger("bot.universe")

DAY_MS = 86_400_000


@dataclass
class Instrument:
    pair: str                  # research pair name
    bybit_symbol: str
    tick_size: float = 0.0
    qty_step: float = 0.0
    min_qty: float = 0.0
    tradeable: bool = False
    delisting: bool = False    # announcements flag: no new entries


@dataclass
class Universe:
    journal: Journal
    liq_gate_usd: float = 1_000_000.0
    instruments: dict[str, Instrument] = field(default_factory=dict)  # by pair

    @classmethod
    def load(cls, journal: Journal, liq_gate_usd: float = 1_000_000.0,
             path: Path | None = None) -> "Universe":
        p = path or (ARTIFACTS_DIR / "universe.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        uni = cls(journal=journal, liq_gate_usd=liq_gate_usd)
        for row in data["pairs"]:
            uni.instruments[row["pair"]] = Instrument(
                pair=row["pair"], bybit_symbol=row.get("bybit_symbol", row["pair"]))
        return uni

    # ------------------------------------------------------------ symbols --
    def pairs(self) -> list[str]:
        return list(self.instruments)

    def tradeable_pairs(self) -> list[str]:
        return [p for p, ins in self.instruments.items()
                if ins.tradeable and not ins.delisting]

    def bybit_symbol(self, pair: str) -> str:
        return self.instruments[pair].bybit_symbol

    def pair_for_bybit(self, bybit_symbol: str) -> str | None:
        for p, ins in self.instruments.items():
            if ins.bybit_symbol == bybit_symbol:
                return p
        return None

    # -------------------------------------------------- instruments-info --
    def apply_instruments_info(self, info_list: list[dict]) -> None:
        by_sym = {r["symbol"]: r for r in info_list}
        missing = []
        for pair, ins in self.instruments.items():
            r = by_sym.get(ins.bybit_symbol)
            if r is None or r.get("status") != "Trading":
                ins.tradeable = False
                missing.append(pair)
                continue
            ins.tradeable = True
            lot = r.get("lotSizeFilter", {})
            ins.tick_size = float(r.get("priceFilter", {}).get("tickSize", 0) or 0)
            ins.qty_step = float(lot.get("qtyStep", 0) or 0)
            ins.min_qty = float(lot.get("minOrderQty", 0) or 0)
        if missing:
            self.journal.event("warn", "universe",
                               f"{len(missing)} pairs not tradeable on Bybit",
                               {"pairs": missing})
            log.warning("not tradeable on Bybit: %s", missing)

    def apply_announcements(self, anns: list[dict]) -> None:
        """Flag universe symbols mentioned in delisting announcements."""
        flagged = []
        for ann in anns:
            title = (ann.get("title") or "").upper()
            for pair, ins in self.instruments.items():
                base = pair.removesuffix("USDT")
                if base and base in title and not ins.delisting:
                    ins.delisting = True
                    flagged.append(pair)
        if flagged:
            self.journal.event("alarm", "delisting",
                               f"possible delisting announcement: {flagged}")

    # -------------------------------------------------------- liquidity --
    def liquidity_ok(self, pair: str, now_ms: int) -> bool:
        """Causal $1M gate: median of the last 30 COMPLETED UTC days."""
        day_ms = (now_ms // DAY_MS) * DAY_MS
        med = self.journal.median_daily_volume(pair, day_ms, 30)
        return med is not None and math.isfinite(med) and med > self.liq_gate_usd
