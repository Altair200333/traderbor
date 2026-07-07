"""Trading universe: parsed from docs/notes/2026-07-07/bybit-trading-universe.md.

Single source of truth is the note's markdown table (149 main-universe coins).
We parse it at runtime to avoid hand-copying 149 tickers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_NOTE = REPO_ROOT / "docs" / "notes" / "2026-07-07" / "bybit-trading-universe.md"

_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*([A-Z0-9]+)\s*\|\s*(T[123])\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([YN])\s*\|"
)


@dataclass(frozen=True)
class Coin:
    symbol: str          # e.g. "BTC"
    pair: str            # e.g. "BTCUSDT" (Binance spot)
    tier: str            # T1 | T2 | T3
    perp_turnover_musd: float
    spot30_musd: float
    adr_pct: float
    bybit_spot: bool


def load_universe(note_path: Path = UNIVERSE_NOTE) -> list[Coin]:
    coins: list[Coin] = []
    for line in note_path.read_text(encoding="utf-8").splitlines():
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        sym, tier, perp, spot30, adr, byspot = m.groups()
        coins.append(
            Coin(
                symbol=sym,
                pair=f"{sym}USDT",
                tier=tier,
                perp_turnover_musd=float(perp),
                spot30_musd=float(spot30),
                adr_pct=float(adr),
                bybit_spot=byspot == "Y",
            )
        )
    if not (140 <= len(coins) <= 160):
        raise RuntimeError(f"universe parse suspicious: {len(coins)} coins from {note_path}")
    return coins


if __name__ == "__main__":
    u = load_universe()
    tiers = {}
    for c in u:
        tiers[c.tier] = tiers.get(c.tier, 0) + 1
    print(f"{len(u)} coins: {tiers}")
    print(",".join(c.pair for c in u[:10]), "...")
