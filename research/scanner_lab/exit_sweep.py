"""Exit-parameter sweep on the selected stream (long P1 bull EV>0, sym-day dedup).

Grid: stop_mult x rr x hold. Outcomes recomputed on 5m paths. Metrics per cell:
- avg r_eff (R in effective-stop units = fixed-risk sizing world)
- avg pnl_pct (raw % move per trade)
- slot-sim (3 slots) total r_eff with cell-specific exit times
Gate-lab prior: 1.5x stops best, tight stops worst; winners need ~19h.

Usage: python exit_sweep.py [--model lgbm] [--cost-bps 25]
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
KL5 = REPO_ROOT / "research" / "data" / "klines" / "5m"

STOP_MULTS = [0.75, 1.0, 1.25, 1.5]
RRS = [1.5, 2.0, 2.5, 3.0, 4.0]
HOLD_H = [24, 48]


def outcomes_for_cell(path_t, path_h, path_l, path_c, as_of, entry, d_eff, rr, hold_h):
    j1 = np.searchsorted(path_t, as_of + hold_h * 3_600_000, side="left")
    h, l, c = path_h[:j1], path_l[:j1], path_c[:j1]
    if len(h) < 3:
        return np.nan, np.nan
    stop_px, tp_px = entry * (1 - d_eff), entry * (1 + d_eff * rr)
    hit_sl, hit_tp = l <= stop_px, h >= tp_px
    j_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
    j_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
    if j_sl < 0 and j_tp < 0:
        r = (c[-1] - entry) / (entry * d_eff)
        t_exit = (path_t[j1 - 1] - as_of) / 60_000 + 5
    elif j_sl >= 0 and (j_tp < 0 or j_sl < j_tp):
        r, t_exit = -1.0, (path_t[j_sl] - as_of) / 60_000 + 5
    elif j_tp >= 0 and (j_sl < 0 or j_tp < j_sl):
        r, t_exit = rr, (path_t[j_tp] - as_of) / 60_000 + 5
    else:
        r, t_exit = -1.0, (path_t[j_sl] - as_of) / 60_000 + 5  # ambiguous -> conservative sl
    return r, t_exit


def slot_r(as_of_ms: np.ndarray, syms: np.ndarray, rs: np.ndarray,
           t_exits: np.ndarray, slots: int = 3) -> tuple[float, int]:
    busy: list[float] = []
    sym_until: dict[str, float] = {}
    tot, n = 0.0, 0
    for i in range(len(as_of_ms)):
        t_min = as_of_ms[i] / 60_000.0
        while busy and busy[0] <= t_min:
            heapq.heappop(busy)
        if len(busy) >= slots or sym_until.get(syms[i], -1e18) > t_min:
            continue
        if not np.isfinite(rs[i]):
            continue
        te = t_exits[i] if np.isfinite(t_exits[i]) else 1440.0
        heapq.heappush(busy, t_min + te)
        sym_until[syms[i]] = t_min + 1440.0
        tot += rs[i]
        n += 1
    return tot, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--cost-bps", type=float, default=25.0)
    args = ap.parse_args()

    df = pd.read_parquet(PRED)
    df = df[df["model"] == args.model].copy()
    cost = args.cost_bps / 1e4
    df["ev"] = df["p_hat"] * df["tp_rr"] - (1 - df["p_hat"]) - cost / df["d_final"]
    df["day"] = pd.to_datetime(df["as_of"], unit="ms", utc=True).dt.date
    base = (df["side"] == "long") & (df["pattern"] == "P1") & (df["btc_above_ema50"] == 1)
    stream = (df[base & (df["ev"] > 0)].sort_values("as_of")
              .groupby(["symbol", "day"], as_index=False).head(1).sort_values("as_of")
              .reset_index(drop=True))
    print(f"stream: {len(stream)} events")

    results = {}
    for sym, sub in stream.groupby("symbol"):
        p5 = KL5 / f"{sym}.parquet"
        d5 = pd.read_parquet(p5, columns=["open_time", "high", "low", "close"])
        t5 = d5["open_time"].to_numpy(np.int64)
        h5 = d5["high"].to_numpy()
        l5 = d5["low"].to_numpy()
        c5 = d5["close"].to_numpy()
        for row in sub.itertuples():
            j0 = np.searchsorted(t5, row.as_of, side="left")
            j2 = np.searchsorted(t5, row.as_of + 49 * 3_600_000, side="left")
            pt, ph, pl, pc = t5[j0:j2], h5[j0:j2], l5[j0:j2], c5[j0:j2]
            for sm in STOP_MULTS:
                d_eff = row.d_final * sm
                for rr in RRS:
                    for hh in HOLD_H:
                        r, te = outcomes_for_cell(pt, ph, pl, pc, row.as_of,
                                                  row.entry, d_eff, rr, hh)
                        if np.isfinite(r):
                            r -= cost / d_eff
                        results.setdefault((sm, rr, hh), []).append(
                            (row.Index, r, te))

    as_of_all = stream["as_of"].to_numpy(np.int64)
    syms_all = stream["symbol"].to_numpy()
    d_all = stream["d_final"].to_numpy()
    order = np.argsort(as_of_all)
    rows = []
    for (sm, rr, hh), vals in sorted(results.items()):
        idx, rs, tes = zip(*vals)
        rs_a = np.full(len(stream), np.nan)
        te_a = np.full(len(stream), np.nan)
        rs_a[list(idx)] = rs
        te_a[list(idx)] = tes
        tot, n_taken = slot_r(as_of_all[order], syms_all[order],
                              rs_a[order], te_a[order], slots=3)
        rows.append({
            "stop_mult": sm, "rr": rr, "hold_h": hh,
            "avg_r": round(np.nanmean(rs_a), 3),
            "avg_pnl_pct": round(np.nanmean(rs_a * d_all * sm) * 100, 3),
            "win": round(np.nanmean(rs_a > 0), 3),
            "med_hold_h": round(np.nanmedian(np.array(tes, dtype=float)) / 60, 1),
            "slot3_R": round(tot, 1), "slot3_n": n_taken,
        })
    out = pd.DataFrame(rows).sort_values("slot3_R", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
