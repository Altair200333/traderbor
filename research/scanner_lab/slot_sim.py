"""Slot-based portfolio simulation: the honest capacity model.

Stream = EV>0 long-P1-bull events, one per symbol-day (first trigger),
chronological. S position slots; an event is taken iff a slot is free at
its as_of; the slot stays busy until the event's exit (t_exit_min; 'none'
outcomes hold the full 24h). No ranking — time priority, matching how the
runner would consume scanner output. Also: heat-gated variant, and a
same-symbol cooldown (skip symbol if traded within 24h).

Usage: python slot_sim.py [--model lgbm] [--cost-bps 25]
"""
from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402

PRED = REPO_ROOT / "research" / "data" / "predictions.parquet"
HOLD_NONE_MIN = 24 * 60.0


def run_slots(ev: pd.DataFrame, slots: int, sym_cooldown_h: float = 24.0) -> pd.DataFrame:
    busy: list[float] = []          # heap of busy-until (minutes since epoch)
    sym_until: dict[str, float] = {}
    taken_idx = []
    for row in ev.itertuples():
        t_min = row.as_of / 60_000.0
        while busy and busy[0] <= t_min:
            heapq.heappop(busy)
        if len(busy) >= slots:
            continue
        if sym_until.get(row.symbol, -1e18) > t_min:
            continue
        hold = row.t_exit_min if np.isfinite(row.t_exit_min) else HOLD_NONE_MIN
        heapq.heappush(busy, t_min + hold)
        sym_until[row.symbol] = t_min + sym_cooldown_h * 60.0
        taken_idx.append(row.Index)
    return ev.loc[taken_idx]


def report(sub: pd.DataFrame, name: str) -> dict:
    if len(sub) == 0:
        return {"policy": name, "n": 0}
    m = sub.groupby("month")["rm"].sum()
    res = sub[sub["outcome"].isin(["tp", "sl"])]
    eq = sub.sort_values("as_of")["rm"].cumsum()
    return {
        "policy": name, "n": len(sub), "per_mo": round(len(sub) / sub["month"].nunique(), 1),
        "tp": round((res["outcome"] == "tp").mean(), 3) if len(res) else np.nan,
        "R": round(sub["rm"].sum(), 1), "avg": round(sub["rm"].mean(), 3),
        "mo_pos": f"{(m > 0).sum()}/{len(m)}", "worst_mo": round(m.min(), 1),
        "maxDD_R": round((eq - eq.cummax()).min(), 1),
        "shp": round(m.mean() / m.std(), 2) if len(m) > 2 and m.std() > 0 else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--pred", default=str(PRED))
    args = ap.parse_args()

    df = pd.read_parquet(args.pred)
    df = df[df["model"] == args.model].copy()
    cost_r = args.cost_bps / 1e4 / df["d_final"]
    df["rm"] = df["r_mkt_eval"] - cost_r
    df["ev"] = df["p_hat"] * df["tp_rr"] - (1 - df["p_hat"]) - cost_r
    df["day"] = pd.to_datetime(df["as_of"], unit="ms", utc=True).dt.date

    base = (df["side"] == "long") & (df["pattern"] == "P1") & (df["btc_above_ema50"] == 1)
    stream = (df[base & (df["ev"] > 0)].sort_values("as_of")
              .groupby(["symbol", "day"], as_index=False).head(1).sort_values("as_of"))
    t = stream["as_of"].to_numpy()
    lo = np.searchsorted(t, t - 24 * 3_600_000, side="left")
    hi = np.searchsorted(t, t, side="right")
    stream = stream.assign(heat24=hi - lo)

    rows = [report(stream, "stream (no capacity)")]
    for s in (2, 3, 4, 6, 10):
        rows.append(report(run_slots(stream, s), f"slots={s}"))
    hot = stream[stream["heat24"] >= 3]
    for s in (3, 4):
        rows.append(report(run_slots(hot, s), f"heat>=3 slots={s}"))
    # control: random-p stream (is EV>0 needed at all under slots?)
    ctl = (df[base].sort_values("as_of").groupby(["symbol", "day"], as_index=False)
           .head(1).sort_values("as_of"))
    rows.append(report(run_slots(ctl, 3), "no-ML slots=3 (control)"))

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(out.to_string(index=False))

    print("\n=== slots=3 monthly ===")
    picked = run_slots(stream, 3)
    g = picked.groupby("month").agg(n=("rm", "size"), R=("rm", "sum"),
                                    tp=("outcome", lambda s: round((s == "tp").mean(), 2)))
    print(g.round(1).to_string())
    vc = picked["symbol"].value_counts()
    print(f"top symbols: {dict(vc.head(6))}, uniq: {len(vc)}")
    print(f"R without top-10 events: {picked['rm'].sum() - picked.nlargest(10, 'rm')['rm'].sum():+.1f}")
    print(f"avg concurrent-day trades: {picked.groupby('day').size().mean():.1f}")


if __name__ == "__main__":
    main()
