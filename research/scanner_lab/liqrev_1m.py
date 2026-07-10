"""Liquidation-cascade reversion: MINUTE-resolution FUTURES execution validation.

PRE-REGISTERED spec (frozen BEFORE first run; no parameter tuning). This is a
VALIDATION study of the frozen liqrev_v2 config against real 1-minute Binance
UM FUTURES prices (the instrument actually traded). Detection is unchanged and
runs on SPOT 1h klines via liqrev_v2.detect_events; only EXECUTION moves to 1m
futures data, exposing intra-hour path risk the 1h sim can hide:
  - limit may never actually be touched inside the hour,
  - the -20% disaster stop may breach intra-hour,
  - "entry at next open" is optimistic inside a still-falling cascade hour,
  - spot-vs-perp basis at the trigger.

EVENTS. Built exactly as liqrev_v2.main(): all 149 universe pairs, spot 1h
klines + 5m OI, ret_6h<=-8% AND OI_6h<=-10%, $1M liquidity, 24h cooldown.
detect_events returns per event: symbol, ts (trigger-bar OPEN time, UTC),
i (index into spot 1h klines), trig_low, trig_close (SPOT 1h close).

t := trigger-bar open time (= ev.ts). H := 3_600_000 ms.

COVERAGE (paired-comparison eligibility). Futures 1m must cover [t, t+26h]:
first bar open_time <= t AND last bar open_time >= t+26h, and the trigger hour
[t, t+1h) is non-empty. 4 pairs (BONK/PEPE/SHIB/FLOKI) trade as 1000x perps and
have NO plain-pair futures file -> all their events uncovered. Pairs listed on
futures later than spot -> early-year events uncovered (reported per-year).

1m EXECUTION MODEL (minute-granular mirror of the frozen 1h rules):
  fut_trig_close = close of the LAST 1m bar with t <= open_time < t+1h.
    basis_bps = (fut_trig_close/spot_trig_close - 1) * 1e4  [diagnostic].
  MAKER (the frozen config): limit BUY at fut_trig_close; active for 1m bars
    with t+1h <= open_time < t+2h; FILLED at the FIRST such bar whose low <
    limit; entry price = limit, entry time = that minute. 10bps RT.
    time-to-fill = (entry_open_time - (t+1h)) / 60000 minutes; first-5-min share
    = fraction of fills with time-to-fill < 5 (adverse-selection timing).
  MARKET (control): entry = open of the FIRST 1m bar with open_time >= t+1h.
    25bps RT. Always fills when covered.
  DISASTER stop = entry * 0.80 (-20%), checked per 1m bar from the entry minute
    through the exit bar: if bar open <= stop -> exit at open (gap-aware); elif
    bar low <= stop -> exit at stop. stop_rate = share of filled trades stopped.
  EXIT = close of the LAST 1m bar with open_time < t+25h (mirrors the 1h exit at
    close of bar i+24; calendar-fixed, independent of entry minute).

PORTFOLIO. liqrev_v2.portfolio() reused (15 slots, 1/15 equity, slot frees at
actual exit). Trade rows carry filled/ts/exit_ts/ret/stopped.

BASELINE (paired). liqrev_v2.simulate() at 1h resolution, restricted to the
SAME covered event subset:
  1h maker = simulate(maker, stop='none', 10bps)   [frozen has NO tight stop;
             the -20% disaster stop is not representable at 1h resolution and
             rarely binds within 24h, so 'none' is the fair 1h comparator]
  1h market = simulate(market, stop='none', 25bps)
Same event rows on both sides -> strictly paired.

PRE-REGISTERED VERDICT RULE:
  VALIDATED if (1m maker net mean) >= 0.60 * (1h maker net mean, same subset)
              AND (1m maker fill rate) >= 0.85
  DEGRADED  if 1m maker net mean > 0 but below either bar
  BROKEN    if 1m maker net mean <= 0
Reported for FULL, DEV (<2025-01-01) and HOLDOUT (>=2025-01-01), plus by-year.

Usage: python liqrev_1m.py   ->   research/data/liqrev/results_1m.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liqrev_v2 import detect_events, simulate, portfolio, KL_DIR  # noqa: E402
from universe import REPO_ROOT, load_universe  # noqa: E402

FUT_DIR = REPO_ROOT / "research" / "data" / "binance_um" / "klines_1m"
ART_DIR = REPO_ROOT / "research" / "data" / "liqrev"
H = 3_600_000  # ms per hour
DEV_CUT = pd.Timestamp("2025-01-01", tz="UTC")


# --------------------------------------------------------------------------- #
# event construction: identical to liqrev_v2.main()
# --------------------------------------------------------------------------- #
def build_events():
    evs, kcache = [], {}
    pairs = [c.pair for c in load_universe()]
    for i, pair in enumerate(pairs, 1):
        e = detect_events(pair)
        if len(e):
            evs.append(e)
            k = pd.read_parquet(KL_DIR / f"{pair}.parquet",
                                columns=["open_time", "open", "high", "low", "close"])
            k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
            kcache[pair] = k.set_index("ts").sort_index()
        if i % 40 == 0:
            print(f"[{i}/{len(pairs)}] events {sum(len(x) for x in evs)}", flush=True)
    ev = pd.concat(evs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    print(f"events total: {len(ev)}")
    return ev, kcache


# --------------------------------------------------------------------------- #
# 1m futures execution, per symbol (memory-frugal: one parquet at a time)
# --------------------------------------------------------------------------- #
def sim_1m(ev: pd.DataFrame) -> pd.DataFrame:
    """Return one record per event (index-aligned to ev) with coverage, basis,
    and maker/market outcomes under minute-resolution futures execution."""
    recs = []
    for sym, g in ev.groupby("symbol", sort=False):
        fp = FUT_DIR / f"{sym}.parquet"
        if not fp.exists():
            for idx, r in g.iterrows():
                recs.append({"idx": idx, "symbol": sym, "ts": r["ts"],
                             "covered": False})
            continue
        f = pd.read_parquet(fp, columns=["open_time", "open", "high", "low", "close"])
        f = f.sort_values("open_time")
        ot = f["open_time"].to_numpy(np.int64)
        op = f["open"].to_numpy(float)
        lo = f["low"].to_numpy(float)
        cl = f["close"].to_numpy(float)
        omin, omax = int(ot[0]), int(ot[-1])
        for idx, r in g.iterrows():
            t = int(r["ts"].value // 1_000_000)  # ns -> ms
            rec = {"idx": idx, "symbol": sym, "ts": r["ts"], "covered": False}
            if not (omin <= t and omax >= t + 26 * H):
                recs.append(rec)
                continue
            # trigger hour [t, t+1h)
            tw_lo = int(np.searchsorted(ot, t, "left"))
            tw_hi = int(np.searchsorted(ot, t + H, "left"))
            if tw_hi <= tw_lo:
                recs.append(rec)  # gap through the whole trigger hour
                continue
            fut_trig_close = cl[tw_hi - 1]
            # entry window [t+1h, t+2h)
            e_lo = int(np.searchsorted(ot, t + H, "left"))
            e_hi = int(np.searchsorted(ot, t + 2 * H, "left"))
            # exit bar: last bar with open_time < t+25h
            x_idx = int(np.searchsorted(ot, t + 25 * H, "left")) - 1
            if e_hi <= e_lo or x_idx <= e_lo:
                recs.append(rec)  # degenerate window (should not happen if covered)
                continue

            rec["covered"] = True
            rec["basis_bps"] = float((fut_trig_close / r["trig_close"] - 1.0) * 1e4)

            def hold_exit(entry_idx: int, entry_px: float):
                """Disaster-stop scan entry_idx..x_idx; return (exit_px, exit_ot, stopped)."""
                stop = entry_px * 0.80
                so = op[entry_idx:x_idx + 1]
                sl = lo[entry_idx:x_idx + 1]
                breach = (so <= stop) | (sl <= stop)
                if breach.any():
                    j0 = int(np.argmax(breach))
                    ex = so[j0] if so[j0] <= stop else stop
                    return float(ex), int(ot[entry_idx + j0]), True
                return float(cl[x_idx]), int(ot[x_idx]), False

            # ---- MAKER ----
            win_lo = lo[e_lo:e_hi]
            fills = win_lo < fut_trig_close
            if fills.any():
                fj = int(np.argmax(fills))
                m_entry_idx = e_lo + fj
                m_entry = fut_trig_close
                ex, ex_ot, st = hold_exit(m_entry_idx, m_entry)
                ttf = (int(ot[m_entry_idx]) - (t + H)) / 60000.0
                rec.update({
                    "mk_filled": True,
                    "mk_ret": ex / m_entry - 1.0 - 0.0010,
                    "mk_stopped": st,
                    "mk_exit_ts": pd.Timestamp(ex_ot, unit="ms", tz="UTC"),
                    "mk_ttf": ttf, "mk_first5": ttf < 5.0,
                })
            else:
                rec["mk_filled"] = False

            # ---- MARKET (control) ----
            mo_entry_idx = e_lo
            mo_entry = op[mo_entry_idx]
            ex, ex_ot, st = hold_exit(mo_entry_idx, mo_entry)
            rec.update({
                "mo_filled": True,
                "mo_ret": ex / mo_entry - 1.0 - 0.0025,
                "mo_stopped": st,
                "mo_exit_ts": pd.Timestamp(ex_ot, unit="ms", tz="UTC"),
            })
            recs.append(rec)
        del f, ot, op, lo, cl
    return pd.DataFrame(recs).set_index("idx")


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def _slice(df, lo, hi):
    d = df
    if lo is not None:
        d = d[d["ts"] >= lo]
    if hi is not None:
        d = d[d["ts"] < hi]
    return d


def stats(tr: pd.DataFrame, lo=None, hi=None, ttf=False) -> dict:
    """tr: one row per covered event, cols filled/ret/stopped/ts/exit_ts
    (+ttf/first5 for maker). Returns paired metrics + 15-slot portfolio."""
    d = _slice(tr, lo, hi)
    f = d[d["filled"]]
    out = {
        "n_events": int(len(d)),
        "n_filled": int(len(f)),
        "fill_rate": round(len(f) / len(d), 4) if len(d) else None,
        "net_mean": round(float(f["ret"].mean()), 5) if len(f) else None,
        "net_median": round(float(f["ret"].median()), 5) if len(f) else None,
        "win": round(float((f["ret"] > 0).mean()), 4) if len(f) else None,
        "stop_rate": round(float(f["stopped"].mean()), 4) if len(f) else None,
    }
    if ttf and len(f):
        out["ttf_median_min"] = round(float(f["ttf"].median()), 2)
        out["ttf_mean_min"] = round(float(f["ttf"].mean()), 2)
        out["first5min_share"] = round(float(f["first5"].mean()), 4)
    p = portfolio(d)
    out["cagr"] = p.get("cagr")
    out["maxDD"] = p.get("maxDD")
    out["total"] = p.get("total")
    out["n_taken"] = p.get("n_taken")
    return out


def periods(tr: pd.DataFrame, ttf=False) -> dict:
    return {
        "FULL": stats(tr, None, None, ttf),
        "DEV": stats(tr, None, DEV_CUT, ttf),
        "HOLDOUT": stats(tr, DEV_CUT, None, ttf),
    }


def by_year(tr: pd.DataFrame, ttf=False) -> dict:
    out = {}
    for y, d in tr.groupby(tr["ts"].dt.year):
        f = d[d["filled"]]
        row = {
            "n_events": int(len(d)), "n_filled": int(len(f)),
            "fill_rate": round(len(f) / len(d), 4) if len(d) else None,
            "net_mean": round(float(f["ret"].mean()), 5) if len(f) else None,
            "net_median": round(float(f["ret"].median()), 5) if len(f) else None,
            "win": round(float((f["ret"] > 0).mean()), 4) if len(f) else None,
            "stop_rate": round(float(f["stopped"].mean()), 4) if len(f) else None,
        }
        if ttf and len(f):
            row["ttf_median_min"] = round(float(f["ttf"].median()), 2)
            row["first5min_share"] = round(float(f["first5"].mean()), 4)
        out[str(int(y))] = row
    return out


def h1_frame(sim_out: pd.DataFrame) -> pd.DataFrame:
    """Normalize liqrev_v2.simulate() output to the common trade schema."""
    d = sim_out.copy()
    if "ret" not in d:
        d["ret"] = np.nan
    if "stopped" not in d:
        d["stopped"] = False
    if "exit_ts" not in d:
        d["exit_ts"] = pd.NaT
    d["stopped"] = d["stopped"].fillna(False)
    return d[["symbol", "ts", "filled", "ret", "stopped", "exit_ts"]]


def coverage_table(rec: pd.DataFrame, ev: pd.DataFrame) -> dict:
    cov = rec["covered"]
    per_year = {}
    yrs = ev["ts"].dt.year
    for y in sorted(yrs.unique()):
        m = yrs == y
        per_year[str(int(y))] = {
            "n_total": int(m.sum()),
            "n_covered": int((m & cov.reindex(ev.index, fill_value=False)).sum()),
        }
    return {
        "n_total": int(len(ev)),
        "n_covered": int(cov.sum()),
        "n_uncovered": int((~cov).sum()),
        "coverage_rate": round(float(cov.mean()), 4),
        "by_year": per_year,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ev, kcache = build_events()
    rec = sim_1m(ev)
    rec = rec.reindex(ev.index)          # align to full event set
    rec["covered"] = rec["covered"].fillna(False)
    rec["symbol"] = ev["symbol"]
    rec["ts"] = ev["ts"]

    cov_tbl = coverage_table(rec, ev)
    print("coverage:", json.dumps(cov_tbl))

    covered_idx = rec.index[rec["covered"]]
    ev_cov = ev.loc[covered_idx].copy()

    # ---- basis diagnostic ----
    b = rec.loc[covered_idx, "basis_bps"].to_numpy(float)
    basis = {
        "n": int(len(b)),
        "mean_bps": round(float(np.mean(b)), 3),
        "median_bps": round(float(np.median(b)), 3),
        "p5_bps": round(float(np.percentile(b, 5)), 3),
        "p95_bps": round(float(np.percentile(b, 95)), 3),
        "std_bps": round(float(np.std(b)), 3),
    }
    print("basis:", json.dumps(basis))

    # ---- 1m trade frames (covered events only) ----
    c = rec.loc[covered_idx]
    mk = pd.DataFrame({
        "symbol": c["symbol"], "ts": c["ts"],
        "filled": c["mk_filled"].fillna(False).astype(bool),
        "ret": c["mk_ret"], "stopped": c["mk_stopped"].fillna(False).astype(bool),
        "exit_ts": c["mk_exit_ts"], "ttf": c["mk_ttf"], "first5": c["mk_first5"],
    })
    mo = pd.DataFrame({
        "symbol": c["symbol"], "ts": c["ts"],
        "filled": c["mo_filled"].fillna(False).astype(bool),
        "ret": c["mo_ret"], "stopped": c["mo_stopped"].fillna(False).astype(bool),
        "exit_ts": c["mo_exit_ts"],
    })

    # ---- 1h baseline on the SAME covered subset (paired) ----
    h1_mk = h1_frame(simulate(ev_cov, kcache, "maker", "none", 0.0010, False))
    h1_mo = h1_frame(simulate(ev_cov, kcache, "market", "none", 0.0025, False))

    results = {
        "maker": {"1h_sim": periods(h1_mk), "1m_sim": periods(mk, ttf=True)},
        "market": {"1h_sim": periods(h1_mo), "1m_sim": periods(mo)},
    }
    byyear_1m_maker = by_year(mk, ttf=True)

    # ---- verdict (pre-registered) ----
    m1m = results["maker"]["1m_sim"]["FULL"]
    m1h = results["maker"]["1h_sim"]["FULL"]
    net_1m = m1m["net_mean"] or 0.0
    net_1h = m1h["net_mean"] or 0.0
    fill_1m = m1m["fill_rate"] or 0.0
    if net_1m <= 0:
        verdict = "BROKEN"
    elif net_1m >= 0.60 * net_1h and fill_1m >= 0.85:
        verdict = "VALIDATED"
    else:
        verdict = "DEGRADED"
    verdict_detail = {
        "verdict": verdict,
        "net_1m_maker": net_1m, "net_1h_maker": net_1h,
        "ratio_1m_over_1h": round(net_1m / net_1h, 4) if net_1h else None,
        "threshold_net": round(0.60 * net_1h, 5) if net_1h else None,
        "fill_1m": fill_1m, "threshold_fill": 0.85,
    }
    print("verdict:", json.dumps(verdict_detail))

    payload = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "liqrev_1m frozen-config minute-futures validation",
        "slots": 15, "hold_exit_h": 25, "cover_h": 26,
        "rt_bps": {"maker": 10, "market": 25}, "disaster_stop": -0.20,
        "coverage": cov_tbl,
        "basis": basis,
        "results": results,
        "byyear_1m_maker": byyear_1m_maker,
        "verdict": verdict_detail,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    outp = ART_DIR / "results_1m.json"
    outp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nartifacts -> {outp}")

    # console summary
    def line(tag, d):
        print(f"  {tag:16s} fill {d['fill_rate']} net {d['net_mean']} "
              f"med {d['net_median']} win {d['win']} stop {d['stop_rate']} "
              f"cagr {d['cagr']} dd {d['maxDD']}")
    for var in ("maker", "market"):
        for res in ("1h_sim", "1m_sim"):
            print(f"[{var} {res}]")
            for per in ("FULL", "DEV", "HOLDOUT"):
                line(per, results[var][res][per])
    print("[1m maker by-year]")
    for y, d in byyear_1m_maker.items():
        print(f"  {y} n{d['n_events']} fill {d['fill_rate']} net {d['net_mean']} "
              f"win {d['win']} stop {d['stop_rate']} "
              f"ttf_med {d.get('ttf_median_min')} first5 {d.get('first5min_share')}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
