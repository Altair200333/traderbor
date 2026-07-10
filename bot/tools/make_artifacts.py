"""One-shot generator of the bot's frozen artifacts (run manually, output committed).

Artifacts:
  bot/artifacts/universe.json       149-pair main universe, parsed from the
                                    frozen note docs/notes/2026-07-07/bybit-trading-universe.md
                                    (same regex as research/scanner_lab/universe.py).
  bot/artifacts/liqrev_overlay.json frozen DEV score distribution for the
                                    BTC-context sizing overlay (M1_dumb x R3_rankw,
                                    adopted 2026-07-09, results_ml.json verdict ADOPT).
                                    Scores = -btc_ret_6h of DEV filled events with
                                    exit_known (ts+25h) < 2024-12-25 UTC, taken VERBATIM
                                    from research/data/liqrev/ml_dataset.parquet.
                                    NOT re-derived: this is a copy of the frozen research
                                    training distribution (liqrev_ml_model.py holdout shot).

Usage: python -m bot.tools.make_artifacts
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ART_DIR = REPO_ROOT / "bot" / "artifacts"
UNIVERSE_NOTE = REPO_ROOT / "docs" / "notes" / "2026-07-07" / "bybit-trading-universe.md"
ML_DATASET = REPO_ROOT / "research" / "data" / "liqrev" / "ml_dataset.parquet"

_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*([A-Z0-9]+)\s*\|\s*(T[123])\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([YN])\s*\|"
)

# Bybit linear-perp ticker overrides for coins that trade as scaled contracts.
# Verified against instruments-info at bot startup; unmatched pairs are excluded
# with an alarm, so a wrong guess here degrades to "symbol skipped", never to a
# wrong order.
BYBIT_OVERRIDES = {
    "PEPEUSDT": "1000PEPEUSDT",
    "BONKUSDT": "1000BONKUSDT",
    "FLOKIUSDT": "1000FLOKIUSDT",
    "SHIBUSDT": "SHIB1000USDT",
}


def make_universe() -> dict:
    pairs = []
    for line in UNIVERSE_NOTE.read_text(encoding="utf-8").splitlines():
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        sym, tier, _perp, _spot30, _adr, _byspot = m.groups()
        pair = f"{sym}USDT"
        pairs.append({
            "symbol": sym,
            "pair": pair,
            "tier": tier,
            "bybit_symbol": BYBIT_OVERRIDES.get(pair, pair),
        })
    if not (140 <= len(pairs) <= 160):
        raise RuntimeError(f"universe parse suspicious: {len(pairs)} pairs")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(UNIVERSE_NOTE.relative_to(REPO_ROOT)),
        "n": len(pairs),
        "pairs": pairs,
    }


def make_overlay() -> dict:
    df = pd.read_parquet(ML_DATASET, columns=["ts", "filled", "btc_ret_6h"])
    d = df[df["filled"]].sort_values("ts")
    cut = pd.Timestamp("2024-12-25", tz="UTC")
    train = d[d["ts"] + pd.Timedelta("25h") < cut]
    scores = np.sort((-train["btc_ret_6h"].to_numpy(dtype=float)))
    scores = scores[np.isfinite(scores)]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "research/data/liqrev/ml_dataset.parquet",
        "spec": ("liqrev_ml_model.py holdout shot: M1_dumb x R3_rankw; "
                 "score=-btc_ret_6h; train = DEV filled events with ts+25h < 2024-12-25 UTC; "
                 "P = searchsorted(scores, s, side='right')/n; weight = min(2.0, 2.0*P)"),
        "train_cut_utc": "2024-12-25T00:00:00+00:00",
        "n": int(len(scores)),
        "scores_sorted": [float(x) for x in scores],
    }


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    uni = make_universe()
    (ART_DIR / "universe.json").write_text(json.dumps(uni, indent=1), encoding="utf-8")
    print(f"universe.json: {uni['n']} pairs")
    ov = make_overlay()
    (ART_DIR / "liqrev_overlay.json").write_text(json.dumps(ov), encoding="utf-8")
    print(f"liqrev_overlay.json: n={ov['n']} scores, "
          f"min={ov['scores_sorted'][0]:.5f} max={ov['scores_sorted'][-1]:.5f}")


if __name__ == "__main__":
    main()
