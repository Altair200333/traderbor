"""Liquidation-cascade reversion: PRE-REGISTERED ROBUSTNESS MAP (fragility check).

This is NOT an optimizer. Its sole purpose is to decide whether the frozen live
trigger (ret <= -8% AND OI_change <= -10% over a 6h window) sits on a PLATEAU
(edge survives parameter perturbation -> mechanism likely real) or on a NEEDLE
(edge appears only at the exact frozen values -> likely noise / overfit).

PRE-COMMITMENT (binding, stated before the run):
  The live parameters (-8%, -10%, 6h) will NOT be changed on the basis of this
  map, whatever it shows. Nothing on this grid is "selected" for deployment.
  This is a one-shot fragility diagnostic around a frozen point, full stop.

HOLDOUT PROTOCOL (deliberate):
  Per-cell metrics are computed on (a) FULL span and (b) DEV only (< 2025-01-01).
  We do NOT compute per-cell holdout (>= 2025-01-01) numbers. The 2025-26 holdout
  has already been spent validating the frozen cell; slicing it per cell across a
  36-cell grid would mine/burn it further. Omitting it is a protocol choice, not
  an oversight. FULL-span aggregates are reported (they include 2025-26 only as
  part of the whole, never isolated per cell).

DECLARED GRID (fixed now; nothing added later):
  T_ret in {-6%, -8%, -10%, -12%}  x  T_oi in {-5%, -10%, -15%}  x  W in {4h,6h,12h}
  W applies JOINTLY to the price return lookback AND the OI-change lookback.
  36 cells. Frozen cell = (-8%, -10%, 6h).

EVERYTHING ELSE STAYS FROZEN (imported from liqrev_v2, single source of truth):
  liquidity filter (30d-median daily quote_volume > $1M), 24h per-symbol cooldown,
  maker execution (post-only limit at trigger close, filled iff next 1h bar
  low < limit), 10bps round-trip cost, NO stop, exit at close of bar i+24,
  15-slot 1/15-equity portfolio. Detection & simulation replicate liqrev_v2
  exactly; only the three trigger parameters vary.

Data: spot 1h klines research/data/v3/klines/1h/{PAIR}.parquet;
      OI 5m research/data/perp/metrics_5m/{PAIR}.parquet (sum_open_interest,
      resample 1h last, ffill) -- copies detect_events' approach.

Artifact: research/data/liqrev/results_robustness.json (all 36 cells x
          {full, dev} + pre-registered plateau analysis).

Usage: python liqrev_robustness.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402
# Reuse frozen execution primitives verbatim (do not re-implement -> DRY, no drift):
from liqrev_v2 import (  # noqa: E402
    KL_DIR, OI_DIR, ART_DIR, HOLD_BARS, SLOTS, simulate, portfolio,
)

# --- declared grid -----------------------------------------------------------
T_RET = [-0.06, -0.08, -0.10, -0.12]   # price return thresholds (<=)
T_OI = [-0.05, -0.10, -0.15]           # OI-change thresholds (<=)
W_BARS = [4, 6, 12]                    # joint lookback window (bars == hours on 1h grid)
FROZEN = (-0.08, -0.10, 6)             # frozen live cell
DEV_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
# Frozen execution knobs (maker, no stop, 10bps RT) -- must match liqrev_v2 spec:
EXEC = dict(entry_mode="maker", stop_mode="none", rt_cost=0.0010, subset_deep=False)


def load_symbol(pair: str):
    """Load klines+OI ONCE; precompute per-window ret/doi and the liquidity mask.

    Mirrors liqrev_v2.detect_events' data handling exactly, then vectorizes the
    windowed lookbacks so the 36 cells reuse one load per symbol.
    """
    kp, op = KL_DIR / f"{pair}.parquet", OI_DIR / f"{pair}.parquet"
    if not kp.exists() or not op.exists():
        return None
    k = pd.read_parquet(kp, columns=["open_time", "open", "high", "low", "close",
                                     "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    oi = pd.read_parquet(op, columns=["ts_ms", "sum_open_interest"])
    oi_s = pd.Series(oi["sum_open_interest"].to_numpy(),
                     index=pd.to_datetime(oi["ts_ms"], unit="ms", utc=True))
    oi_h = oi_s.resample("1h").last().reindex(k.index).ffill()
    dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
    liq_ok = dvol30.reindex(k.index, method="ffill") > 1e6
    ret = {w: k["close"].pct_change(w) for w in W_BARS}
    doi = {w: oi_h.pct_change(w) for w in W_BARS}
    return {
        "ohlc": k[["open", "high", "low", "close"]],   # kept in kcache for simulate()
        "ret": ret, "doi": doi, "liq_ok": liq_ok,
    }


def detect_cell(pair: str, sd: dict, t_ret: float, t_oi: float, w: int) -> pd.DataFrame:
    """Replicate liqrev_v2.detect_events for arbitrary (t_ret, t_oi, w).

    Same order of guards: 24h cooldown checked first (last accepted trigger),
    then the +25 bar future-data boundary, then accept. Cooldown updates only on
    acceptance -- identical to v1.
    """
    k = sd["ohlc"]
    ret, doi, liq = sd["ret"][w], sd["doi"][w], sd["liq_ok"]
    mask = ((ret <= t_ret) & (doi <= t_oi) & liq).fillna(False).to_numpy()
    idx = k.index
    lo = k["low"].to_numpy()
    cl = k["close"].to_numpy()
    rv = ret.to_numpy()
    n = len(k)
    rows, last_i = [], None
    for i in np.flatnonzero(mask):
        i = int(i)
        if last_i is not None and (idx[i] - idx[last_i]) < pd.Timedelta("24h"):
            continue
        if i + HOLD_BARS + 1 >= n:
            continue
        last_i = i
        rows.append({"symbol": pair, "ts": idx[i], "i": i,
                     "ret6": float(rv[i]),           # only read by simulate when deep
                     "trig_low": float(lo[i]),        # only read when a stop is set
                     "trig_close": float(cl[i])})
    return pd.DataFrame(rows)


def cell_metrics(tr: pd.DataFrame) -> dict:
    """Per-cell metrics from a simulated trade frame (filled + unfilled rows)."""
    n_events = int(len(tr))
    if n_events == 0:
        return {"n_events": 0, "n_filled": 0, "fill_rate": None, "net_mean": None,
                "net_median": None, "win": None, "total": None, "cagr": None,
                "maxDD": None}
    f = tr[tr["filled"]]
    nf = int(len(f))
    port = portfolio(tr) if nf else {}
    return {
        "n_events": n_events, "n_filled": nf,
        "fill_rate": round(nf / n_events, 3),
        "net_mean": round(float(f["ret"].mean()), 4) if nf else None,
        "net_median": round(float(f["ret"].median()), 4) if nf else None,
        "win": round(float((f["ret"] > 0).mean()), 3) if nf else None,
        "total": port.get("total"), "cagr": port.get("cagr"),
        "maxDD": port.get("maxDD"),
    }


def build_cells(cell_events: dict, kcache: dict) -> dict:
    """Simulate each cell once (full); derive DEV metrics by filtering entry ts."""
    cells = {}
    for cell, ev_list in cell_events.items():
        if ev_list:
            ev = pd.concat(ev_list, ignore_index=True).sort_values("ts")
            ev = ev.reset_index(drop=True)
            tr = simulate(ev, kcache, EXEC["entry_mode"], EXEC["stop_mode"],
                          EXEC["rt_cost"], EXEC["subset_deep"])
        else:
            tr = pd.DataFrame(columns=["symbol", "ts", "filled"])
        tr_dev = tr[tr["ts"] < DEV_CUTOFF] if len(tr) else tr
        cells[cell] = {"full": cell_metrics(tr), "dev": cell_metrics(tr_dev)}
    return cells


# --- pre-registered plateau analysis ----------------------------------------
def axis_neighbors(cell):
    """The (up to) 6 axis-neighbors: T_ret +-1 step, T_oi +-1 step, W +-1 step."""
    tr, to, w = cell
    out = []
    ir, io, iw = T_RET.index(tr), T_OI.index(to), W_BARS.index(w)
    for axis, seq, idx in (("T_ret", T_RET, ir), ("T_oi", T_OI, io), ("W", W_BARS, iw)):
        for step in (-1, +1):
            j = idx + step
            if 0 <= j < len(seq):
                nc = list(cell)
                nc[{"T_ret": 0, "T_oi": 1, "W": 2}[axis]] = seq[j]
                out.append((axis, step, tuple(nc)))
    return out


def analyze(cells: dict) -> dict:
    fz = cells[FROZEN]["full"]
    fz_mean = fz["net_mean"]
    neigh_rows, ratios = [], []
    for axis, step, nc in axis_neighbors(FROZEN):
        nm_full = cells[nc]["full"]
        nm_dev = cells[nc]["dev"]
        r = (round(fz_mean / nm_full["net_mean"], 2)
             if nm_full["net_mean"] not in (None, 0) else None)
        ratios.append((nc, nm_full["net_mean"], r))
        neigh_rows.append({
            "cell": {"T_ret": nc[0], "T_oi": nc[1], "W": nc[2]},
            "axis": axis, "step": step,
            "net_mean_full": nm_full["net_mean"], "cagr_full": nm_full["cagr"],
            "net_mean_dev": nm_dev["net_mean"],
            "n_filled_full": nm_full["n_filled"],
            "ratio_frozen_over_neighbor": r,
        })
    # NEEDLE FLAG: frozen net mean > 2x EVERY axis-neighbor's net mean (isolated peak).
    # Literal per pre-registration. Caveat recorded: a neighbor with net_mean<=0
    # trivially satisfies 2x for a positive frozen mean, so we also record how many
    # neighbors are within-band (>= frozen/2) for the honest read.
    needle = all(
        (nm is not None and fz_mean is not None and fz_mean > 2 * nm)
        for _, nm, _ in ratios
    )
    n_within_band = sum(
        1 for _, nm, _ in ratios
        if nm is not None and fz_mean is not None and nm >= fz_mean / 2
    )

    # Structure/monotonicity: hold two frozen axes, walk the third by severity.
    def axis_walk(vary):
        seq = []
        for v in {"T_ret": T_RET, "T_oi": T_OI, "W": W_BARS}[vary]:
            cell = {"T_ret": (v, FROZEN[1], FROZEN[2]),
                    "T_oi": (FROZEN[0], v, FROZEN[2]),
                    "W": (FROZEN[0], FROZEN[1], v)}[vary]
            m = cells[cell]["full"]
            seq.append({"val": v, "n_filled": m["n_filled"],
                        "net_mean": m["net_mean"], "cagr": m["cagr"]})
        return seq

    # Severity order: more-negative threshold = more severe. For T_ret/T_oi the
    # declared lists already run least->most severe. Monotone-in-severity means
    # net_mean non-decreasing and n_filled non-increasing along that order.
    def mono(seq, key):
        vals = [s[key] for s in seq if s[key] is not None]
        return len(vals) > 1 and all(b >= a for a, b in zip(vals, vals[1:]))

    t_ret_walk = axis_walk("T_ret")
    t_oi_walk = axis_walk("T_oi")
    struct = {
        "T_ret_axis": {"walk": t_ret_walk,
                       "net_mean_monotone_up": mono(t_ret_walk, "net_mean"),
                       "n_filled_monotone_down": mono(
                           [{"net_mean": -s["n_filled"]} for s in t_ret_walk], "net_mean")},
        "T_oi_axis": {"walk": t_oi_walk,
                      "net_mean_monotone_up": mono(t_oi_walk, "net_mean"),
                      "n_filled_monotone_down": mono(
                          [{"net_mean": -s["n_filled"]} for s in t_oi_walk], "net_mean")},
    }

    # Thin cells: n_filled < 100 (stats unreliable).
    thin = [{"T_ret": c[0], "T_oi": c[1], "W": c[2],
             "n_filled": cells[c]["full"]["n_filled"]}
            for c in cells if cells[c]["full"]["n_filled"] < 100]

    return {
        "frozen_cell": {"T_ret": FROZEN[0], "T_oi": FROZEN[1], "W": FROZEN[2],
                        "net_mean_full": fz_mean, "cagr_full": fz["cagr"],
                        "net_mean_dev": cells[FROZEN]["dev"]["net_mean"],
                        "n_filled_full": fz["n_filled"]},
        "neighborhood": neigh_rows,
        "needle_flag": bool(needle),
        "neighbors_within_band_ge_half": n_within_band,
        "n_axis_neighbors": len(ratios),
        "monotonicity": struct,
        "thin_cells": thin,
    }


def main() -> None:
    pairs = [c.pair for c in load_universe()]
    cells_list = list(product(T_RET, T_OI, W_BARS))
    cell_events = {c: [] for c in cells_list}
    kcache = {}
    for n, pair in enumerate(pairs, 1):
        sd = load_symbol(pair)
        if sd is None:
            continue
        produced = False
        for cell in cells_list:
            ev = detect_cell(pair, sd, *cell)
            if len(ev):
                cell_events[cell].append(ev)
                produced = True
        if produced:
            kcache[pair] = sd["ohlc"]   # OHLC only, bounded memory (as liqrev_v2)
        if n % 40 == 0:
            tot = sum(sum(len(e) for e in v) for v in cell_events.values())
            print(f"[{n}/{len(pairs)}] cell-events so far {tot}", flush=True)

    cells = build_cells(cell_events, kcache)
    analysis = analyze(cells)

    # Serialize (tuple keys -> list rows).
    cell_rows = []
    for c in cells_list:
        row = {"T_ret": c[0], "T_oi": c[1], "W": c[2],
               "is_frozen": (c == FROZEN)}
        row.update({"full": cells[c]["full"], "dev": cells[c]["dev"]})
        cell_rows.append(row)

    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "study": "liqrev_robustness_map",
        "pre_commitment": "params NOT changed based on this map; fragility check only",
        "holdout_protocol": "per-cell = FULL + DEV(<2025-01-01) only; no per-cell holdout",
        "slots": SLOTS, "hold_bars": HOLD_BARS,
        "execution": {"mode": "maker_limit_at_trigger_close",
                      "fill_rule": "next_1h_bar_low < limit", "rt_cost_bps": 10,
                      "stop": "none", "exit": "close_of_bar_i+24"},
        "grid": {"T_ret": T_RET, "T_oi": T_OI, "W": W_BARS},
        "frozen_cell": {"T_ret": FROZEN[0], "T_oi": FROZEN[1], "W": FROZEN[2]},
        "cells": cell_rows,
        "analysis": analysis,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / "results_robustness.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # Compact console table (full-period).
    print("\n=== 36-CELL ROBUSTNESS MAP (full-period) ===")
    print(f"{'T_ret':>6} {'T_oi':>5} {'W':>3} | {'n':>5} {'fill':>5} "
          f"{'net_mean':>9} {'win':>5} {'cagr':>7} {'maxDD':>7} | dev_net")
    for c in cells_list:
        m, d = cells[c]["full"], cells[c]["dev"]
        star = "*" if c == FROZEN else " "
        print(f"{c[0]*100:>5.0f}% {c[1]*100:>4.0f}% {c[2]:>3}{star}| "
              f"{m['n_filled']:>5} {str(m['fill_rate']):>5} "
              f"{str(m['net_mean']):>9} {str(m['win']):>5} "
              f"{str(m['cagr']):>7} {str(m['maxDD']):>7} | {d['net_mean']}")
    print(f"\nneedle_flag={analysis['needle_flag']} "
          f"within_band(>=half)={analysis['neighbors_within_band_ge_half']}"
          f"/{analysis['n_axis_neighbors']}")
    print(f"artifact -> {out}")


if __name__ == "__main__":
    main()
