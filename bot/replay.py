"""P1 signal-parity replay: run the bot's liqrev detector over the research
historical data and compare against the research pipeline's canonical events.

Canonical event list = research/data/liqrev/ml_dataset.parquet (built by
liqrev_ml_features.py directly from liqrev_v2.detect_events — one row per
detected event, symbol + ts at trigger-bar OPEN time, 1308 events 2022-2026).

The bot detector (bot.strategy.liqrev_v2.detect_series) is fed the SAME inputs
the research pipeline used: Binance spot 1h klines (research/data/v3/klines/1h)
and 5m OI (research/data/perp/metrics_5m), with the research liquidity-gate
semantics and the research tail guard (25 bars). Target: EXACT (symbol, hour)
match. Every diff must be explained before shipping.

Also verifies the frozen overlay artifact: weights computed by the bot's
Overlay (bot/artifacts/liqrev_overlay.json) are compared against an
independent recomputation from ml_dataset.parquet (research filter:
DEV filled, ts+25h < 2024-12-25).

Usage:
  python -m bot.replay --start 2026-01-01 --end 2026-07-05
  python -m bot.replay                      (full span)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bot.config import ARTIFACTS_DIR
from bot.strategy.liqrev_v2 import (HOLD_BARS, Overlay, detect_series,
                                    research_liquidity_gate, research_oi_hourly)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "research" / "data"


def bot_events_for_pair(pair: str, kl_dir: Path, oi_dir: Path) -> pd.DataFrame:
    kp, op = kl_dir / f"{pair}.parquet", oi_dir / f"{pair}.parquet"
    if not kp.exists() or not op.exists():
        return pd.DataFrame()
    k = pd.read_parquet(kp, columns=["open_time", "close", "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    oi = pd.read_parquet(op, columns=["ts_ms", "sum_open_interest"])
    oi_s = pd.Series(oi["sum_open_interest"].to_numpy(),
                     index=pd.to_datetime(oi["ts_ms"], unit="ms", utc=True))
    oi_h = research_oi_hourly(oi_s, k.index)
    liq_ok = research_liquidity_gate(k["quote_volume"])
    trig = detect_series(k["close"], oi_h, liq_ok, min_tail_bars=HOLD_BARS + 1)
    if not trig:
        return pd.DataFrame()
    return pd.DataFrame({"symbol": pair, "ts": [t.ts for t in trig],
                         "ret6": [t.ret6 for t in trig],
                         "doi6": [t.doi6 for t in trig]})


def run_parity(data_root: Path, canonical: Path, start: str | None,
               end: str | None) -> dict:
    kl_dir = data_root / "v3" / "klines" / "1h"
    oi_dir = data_root / "perp" / "metrics_5m"
    uni = json.loads((ARTIFACTS_DIR / "universe.json").read_text(encoding="utf-8"))
    pairs = [r["pair"] for r in uni["pairs"]]

    parts = []
    for i, pair in enumerate(pairs, 1):
        e = bot_events_for_pair(pair, kl_dir, oi_dir)
        if len(e):
            parts.append(e)
        if i % 40 == 0:
            print(f"[{i}/{len(pairs)}] bot events so far: "
                  f"{sum(len(x) for x in parts)}", flush=True)
    bot_ev = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=["symbol", "ts"]))

    ml = pd.read_parquet(canonical, columns=["symbol", "ts", "btc_ret_6h", "filled"])
    lo = pd.Timestamp(start, tz="UTC") if start else None
    hi = pd.Timestamp(end, tz="UTC") if end else None

    def window(df: pd.DataFrame) -> pd.DataFrame:
        if lo is not None:
            df = df[df["ts"] >= lo]
        if hi is not None:
            df = df[df["ts"] < hi]
        return df

    bw, rw = window(bot_ev), window(ml)
    bot_set = set(zip(bw["symbol"], bw["ts"]))
    res_set = set(zip(rw["symbol"], rw["ts"]))
    missing = sorted(res_set - bot_set, key=lambda x: (x[1], x[0]))
    extra = sorted(bot_set - res_set, key=lambda x: (x[1], x[0]))

    # ---- overlay weight parity (independent recomputation vs bot artifact) --
    d = ml[ml["filled"]].sort_values("ts")
    train = d[d["ts"] + pd.Timedelta("25h") < pd.Timestamp("2024-12-25", tz="UTC")]
    scores = np.sort((-train["btc_ret_6h"].to_numpy(dtype=float)))
    scores = scores[np.isfinite(scores)]
    overlay = Overlay.load()
    x = rw["btc_ret_6h"].to_numpy(dtype=float)
    finite = np.isfinite(x)
    ref_w = np.minimum(2.0, 2.0 * np.searchsorted(scores, -x[finite], side="right")
                       / len(scores))
    bot_w = np.array([overlay.weight(v) for v in x[finite]])
    w_max_diff = float(np.max(np.abs(ref_w - bot_w))) if len(ref_w) else 0.0

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start or "data-start", "end": end or "data-end"},
        "data_root": str(data_root),
        "canonical": str(canonical),
        "n_research": len(res_set),
        "n_bot": len(bot_set),
        "n_matched": len(res_set & bot_set),
        "match_rate": round(len(res_set & bot_set) / len(res_set), 6) if res_set else None,
        "missing_in_bot": [{"symbol": s, "ts": str(t)} for s, t in missing],
        "extra_in_bot": [{"symbol": s, "ts": str(t)} for s, t in extra],
        "overlay": {"n_checked": int(finite.sum()),
                    "artifact_n": len(overlay.scores),
                    "recomputed_n": int(len(scores)),
                    "weight_max_abs_diff": w_max_diff},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="liqrev P1 signal-parity replay")
    ap.add_argument("--start", default=None, help="UTC date, e.g. 2026-01-01")
    ap.add_argument("--end", default=None, help="UTC date (exclusive)")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA))
    ap.add_argument("--canonical",
                    default=str(DEFAULT_DATA / "liqrev" / "ml_dataset.parquet"))
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    rep = run_parity(Path(args.data_root), Path(args.canonical), args.start, args.end)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("missing_in_bot", "extra_in_bot")}, indent=2))
    print(f"missing_in_bot: {len(rep['missing_in_bot'])}")
    for r in rep["missing_in_bot"][:50]:
        print("  MISSING", r["symbol"], r["ts"])
    print(f"extra_in_bot: {len(rep['extra_in_bot'])}")
    for r in rep["extra_in_bot"][:50]:
        print("  EXTRA  ", r["symbol"], r["ts"])
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    ok = (rep["match_rate"] == 1.0 and not rep["extra_in_bot"]
          and rep["overlay"]["weight_max_abs_diff"] == 0.0)
    print("PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
