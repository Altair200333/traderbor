"""Export scanner-v3 champion pings for the agents-v2 v3-file replay provider.

Joins the Phase-D per-event predictions (selected_ev = champion EV>0 @25bps)
with events_v3a2 geometry/facts, restricts to one calendar month, drops symbols
without settlement data in the target market cache, and writes:
  <out>/<month>.pings.parquet   input for TRADERBOT_SCANNER_V3_PINGS
  <out>/<month>.phase1.jsonl    synthetic phase-1 log for --phase1-replay

Usage:
  python export_v3_pings.py --month 2026-06 \
      --preds ../data/v3/artifacts/20260707T185311Z_phase_d_forward_champion/vault_forward.preds.parquet \
      --events ../data/events_v3a2.parquet \
      --cache ../../agents-v2/data/wide-2y-1h/market_cache.sqlite3 \
      --out ../../agents-v2/data/v3pings
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

FACT_COLS = ["pattern", "side", "entry", "boundary", "d_atr", "d_struct", "d_noise",
             "d_final", "tp_rr", "stop_feasible", "roc_4h", "roc_24h", "roc_1h",
             "vol_ratio", "rsi14", "atr_pct", "ema20_ext_atr", "btc_roc_4h"]


def cache_symbols(cache_path: Path, interval: str, start_ms: int, end_ms: int) -> set[str]:
    conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM candles WHERE interval=? AND open_time BETWEEN ? AND ?",
            (interval, start_ms, end_ms)).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="calendar month, e.g. 2026-06")
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path, help="market cache the replay will settle against")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    preds = pd.read_parquet(args.preds)
    ts = pd.to_datetime(preds["as_of"], unit="ms", utc=True)
    sel = preds[preds["selected_ev"] & (ts.dt.strftime("%Y-%m") == args.month)]
    print(f"{args.month}: {len(sel)} selected events, {sel['as_of'].nunique()} bars")

    events = pd.read_parquet(args.events)
    keep = ["symbol", "as_of"] + [c for c in FACT_COLS if c in events.columns]
    pings = sel[["symbol", "as_of", "p_hat", "ev_val"]].merge(
        events[keep], on=["symbol", "as_of"], how="left", validate="one_to_one")
    missing_geom = pings["entry"].isna().sum()
    if missing_geom:
        raise SystemExit(f"ERROR: {missing_geom} pings missing geometry after join")

    start_ms, end_ms = int(pings["as_of"].min()), int(pings["as_of"].max())
    covered = (cache_symbols(args.cache, "5m", start_ms, end_ms)
               & cache_symbols(args.cache, "1h", start_ms, end_ms))
    dropped = sorted(set(pings["symbol"]) - covered)
    if dropped:
        n_before = len(pings)
        pings = pings[pings["symbol"].isin(covered)]
        print(f"WARN: dropped {n_before - len(pings)} pings on {len(dropped)} symbols "
              f"without 5m+1h cache coverage: {dropped}")

    args.out.mkdir(parents=True, exist_ok=True)
    pings_path = args.out / f"{args.month}.pings.parquet"
    pings.reset_index(drop=True).to_parquet(pings_path, index=False)

    phase1_path = args.out / f"{args.month}.phase1.jsonl"
    with open(phase1_path, "w", encoding="utf-8") as fh:
        for as_of_ms, group in pings.groupby("as_of"):
            record = {"type": "step_completed",
                      "payload": {"as_of_ms": int(as_of_ms),
                                  "scan": {"candidates": sorted(group["symbol"])}}}
            fh.write(json.dumps(record) + "\n")

    print(f"pings:  {pings_path}  ({len(pings)} rows, {pings['as_of'].nunique()} bars, "
          f"{pings['symbol'].nunique()} symbols)")
    print(f"phase1: {phase1_path}")


if __name__ == "__main__":
    main()
