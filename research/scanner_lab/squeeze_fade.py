"""Short-squeeze fade: the MIRROR of liquidation-cascade reversion (one-shot).

PRE-REGISTERED spec (frozen from the 2026-07-08 design pass BEFORE the first run;
this is the SHORT mirror of the proven liqrev_v2.py LONG study):

HYPOTHESIS. When price PUMPS hard AND open-interest COLLAPSES, shorts are being
force-liquidated (forced buying) -> the pump overshoots -> SHORT the overshoot,
fixed hold, no stop. Known risks stated up front: (1) up-moves continue more
often than they revert (momentum), and (2) a short PAYS funding when funding is
negative, and 2025-26 average funding is negative -> funding is a headwind.

EVENT (1h grid, full 4y span):
  ret_6h >= +T  AND  OI_6h_change <= -0.10  (OI collapse = shorts force-closed),
  liquidity filter: 30d-median of daily quote_volume > $1M,
  24h per-symbol cooldown (first-come, applied at the +8% level; the +12%
  "deep" branch is a SUBSET of the cooldown-filtered +8% events, exactly as
  liqrev_v2 derives its deep branch -- kept identical for comparability).

DECLARED GRID (frozen; NO axis may be added after seeing results):
  - severity  T   in {+8%, +12%}
  - hold      H   in {12, 24, 48} bars (=hours)
  - entry         in {market  = fill at next 1h OPEN;
                      maker    = limit SELL at trigger-bar CLOSE, valid the next
                                 1h bar only, filled IFF next bar HIGH > limit
                                 (trade-through ABOVE), unfilled = SKIPPED}
  - costs         25bps RT market / 10bps RT maker;
                  PLUS a 50bps RT sensitivity on the DEV-best cell only.
  => 2 x 3 x 2 = 12 base cells at declared costs, + 1 sensitivity.

DIRECTION = SHORT. Per-trade net return (pre-registered formula):
      net = (entry / exit - 1)            # short pnl, entry/exit convention
            - rt_cost                     # round-trip cost
            + funding_realized            # see below
  funding_realized = sum of fundingRate at settlements STRICTLY INSIDE
  (entry_ts, exit_ts]  (i.e. entry_ts < settle <= exit_ts). SHORT receives
  positive funding (+), pays negative funding (-); added with its own sign.
  entry_ts = open time of the entry bar; exit_ts = entry_ts + H hours (position
  is held exactly H bars, exiting at the close of the last held bar).
  NO STOP (event study; the 15-slot sizing is the only risk control).

PORTFOLIO: 15-slot, 1/15 equity/slot, slot frees at actual exit -- byte-for-byte
the liqrev_v2.portfolio() logic (imported behaviour, reimplemented identically).

CONTROLS (mandatory):
  (a) price-only : ret_6h >= +8% with NO OI condition, same pipeline, 24h hold,
                   market 25bps. Does OI-collapse specificity exist on the SHORT
                   side too, or is +8%/6h alone enough?
  (b) random-bars: ~1500 random symbol-bars passing ONLY the $1M liquidity
                   filter, same short sim (24h market 25bps). Baseline drift.

EVENT-STUDY (no trading, before costs): for the +8% OI-collapse events, raw
FORWARD PRICE returns close_{t+h}/close_t - 1 at h in {4,12,24,48}h, with mean
and P(down) [= fraction that fell = short-favourable]. Shows momentum-vs-reversion
shape before any cost/funding. Reported for events vs price-only vs random.

HONESTY PROTOCOL:
  DEV      = events with ts <  2025-01-01
  HOLDOUT  = events with ts >= 2025-01-01
  The best grid cell is chosen on DEV ONLY (metric: DEV per-event net mean, among
  base cells with DEV n_filled >= 30; tie-break DEV portfolio CAGR). The study's
  VERDICT number is that cell's HOLDOUT performance. The full grid is reported for
  BOTH windows (labelled diagnostic), plus a by-year table for the DEV-best cell
  and the controls. CAVEAT: the 149-coin universe was picked in 2026 and is
  survivorship-biased -- early years are inflated; treat pre-2024 with suspicion.

Usage:  python squeeze_fade.py    (run from research/scanner_lab)
Artifacts: research/data/squeeze/results.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402

KL_DIR = REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h"
OI_DIR = REPO_ROOT / "research" / "data" / "perp" / "metrics_5m"
FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
ART_DIR = REPO_ROOT / "research" / "data" / "squeeze"

SLOTS = 15
MAX_HOLD = 48          # detection guard so every hold-cell shares one event set
DEV_CUT = pd.Timestamp("2025-01-01", tz="UTC")
HORIZONS = [4, 12, 24, 48]
RANDOM_N = 1500
SEED = 20260708


# ----------------------------------------------------------------------------- funding
def load_funding_map(pairs: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per symbol: (settlement_times_ns_sorted, prefix_cumsum_of_rate[len+1]).
    Enables O(log n) sum of funding over (a, b] via searchsorted."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in pairs:
        fp = FUND_DIR / f"{p}.parquet"
        if not fp.exists():
            continue
        f = pd.read_parquet(fp, columns=["fundingTime", "fundingRate"])
        ts = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("1h")
        s = pd.Series(f["fundingRate"].to_numpy(), index=ts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # force ns so it matches Timestamp.value (parquet ts can be ms-resolution)
        times = np.asarray(s.index.as_unit("ns").asi8)  # ns since epoch
        cum = np.concatenate([[0.0], np.cumsum(s.to_numpy(dtype=float))])
        out[p] = (times, cum)
    return out


def funding_in(fund: tuple[np.ndarray, np.ndarray] | None,
               a: pd.Timestamp, b: pd.Timestamp) -> float:
    """Sum fundingRate over settlements with a < t <= b (short-sign, +received)."""
    if fund is None:
        return 0.0
    times, cum = fund
    an, bn = a.value, b.value
    lo = int(np.searchsorted(times, an, side="right"))  # first t > a
    hi = int(np.searchsorted(times, bn, side="right"))  # count t <= b
    return float(cum[hi] - cum[lo])


# ----------------------------------------------------------------------------- events
def detect_events(pair: str, use_oi: bool) -> pd.DataFrame:
    """Short-squeeze triggers: ret_6h >= +8% [AND OI_6h <= -10% if use_oi],
    $1M liquidity, 24h symbol cooldown. Severity split to +12% is done later."""
    kp, op = KL_DIR / f"{pair}.parquet", OI_DIR / f"{pair}.parquet"
    if not kp.exists() or (use_oi and not op.exists()):
        return pd.DataFrame()
    k = pd.read_parquet(kp, columns=["open_time", "open", "high", "low", "close",
                                     "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    ret6 = k["close"].pct_change(6)
    dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
    liq_ok = dvol30.reindex(k.index, method="ffill") > 1e6
    cond = (ret6 >= 0.08) & liq_ok
    if use_oi:
        oi = pd.read_parquet(op, columns=["ts_ms", "sum_open_interest"])
        oi_s = pd.Series(oi["sum_open_interest"].to_numpy(),
                         index=pd.to_datetime(oi["ts_ms"], unit="ms", utc=True))
        oi_h = oi_s.resample("1h").last().reindex(k.index).ffill()
        doi6 = oi_h.pct_change(6)
        cond = cond & (doi6 <= -0.10)
    mask = cond.fillna(False)

    rows, last_t = [], None
    n = len(k)
    for t in k.index[mask]:
        if last_t is not None and (t - last_t) < pd.Timedelta("24h"):
            continue
        i = k.index.get_loc(t)
        if i + 1 + MAX_HOLD >= n:            # need room for the deepest hold
            continue
        last_t = t
        rows.append({"symbol": pair, "ts": t, "i": int(i),
                     "ret6": float(ret6.loc[t]),
                     "trig_close": float(k["close"].iloc[i])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- sim
def simulate(ev: pd.DataFrame, kcache: dict, fundmap: dict,
             entry_mode: str, hold: int, rt_cost: float, deep: bool) -> pd.DataFrame:
    out = []
    for r in ev.itertuples(index=False):
        if deep and r.ret6 < 0.12:
            continue
        k = kcache[r.symbol]
        i = r.i
        entry_bar = i + 1
        if entry_bar + hold > len(k):
            continue
        if entry_mode == "market":
            entry = float(k["open"].iloc[entry_bar])
        else:  # maker: limit SELL at trigger close, fill iff next bar HIGH > limit
            limit = r.trig_close
            if float(k["high"].iloc[entry_bar]) > limit:
                entry = limit
            else:
                out.append({"symbol": r.symbol, "ts": r.ts, "filled": False})
                continue
        exit_px = float(k["close"].iloc[entry_bar + hold - 1])
        entry_ts = k.index[entry_bar]
        exit_ts = entry_ts + pd.Timedelta(hours=hold)
        gross = entry / exit_px - 1.0                       # SHORT pnl
        fund = funding_in(fundmap.get(r.symbol), entry_ts, exit_ts)
        net = gross - rt_cost + fund
        out.append({"symbol": r.symbol, "ts": r.ts, "filled": True,
                    "ret": net, "gross": gross, "fund": fund,
                    "exit_ts": exit_ts})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- portfolio (liqrev_v2-identical)
def portfolio(tr: pd.DataFrame) -> dict:
    t = tr[tr["filled"]].sort_values("ts")
    eq, busy, curve, n_taken = 1.0, [], [], 0
    for _, r in t.iterrows():
        busy = [b for b in busy if b > r["ts"]]
        if len(busy) < SLOTS:
            eq *= (1 + r["ret"] / SLOTS)
            busy.append(r["exit_ts"])
            n_taken += 1
        curve.append((r["ts"], eq))
    c = pd.Series(dict(curve))
    if c.empty:
        return {}
    years = max((c.index[-1] - c.index[0]).days / 365.25, 1e-9)
    m = c.resample("MS").last().ffill().pct_change().dropna()
    yearly = c.groupby(c.index.year).last() / c.groupby(c.index.year).first() - 1
    return {"total": round(float(eq - 1), 4),
            "cagr": round(float(eq ** (1 / years) - 1), 4),
            "maxDD": round(float((c / c.cummax() - 1).min()), 4),
            "worst_month": round(float(m.min()), 4) if len(m) else None,
            "n_taken": n_taken,
            "by_year": {str(k): round(float(v), 3) for k, v in yearly.items()}}


def _metrics(f: pd.DataFrame) -> dict:
    if not len(f):
        return {"n": 0}
    return {"n": int(len(f)),
            "net_mean": round(float(f["ret"].mean()), 4),
            "net_median": round(float(f["ret"].median()), 4),
            "win": round(float((f["ret"] > 0).mean()), 3),
            "gross_mean": round(float(f["gross"].mean()), 4),
            "fund_mean": round(float(f["fund"].mean()), 5),
            "p5": round(float(f["ret"].quantile(0.05)), 4)}


def report(tr: pd.DataFrame, tag: str) -> dict:
    f = tr[tr["filled"]]
    dev = f[f["ts"] < DEV_CUT]
    hold = f[f["ts"] >= DEV_CUT]
    return {"tag": tag,
            "n_events": int(len(tr)),
            "n_filled": int(len(f)),
            "fill_rate": round(float(len(f) / len(tr)), 3) if len(tr) else None,
            "ALL": {**_metrics(f), "portfolio": portfolio(tr)},
            "DEV": {**_metrics(dev), "portfolio": portfolio(tr[tr["ts"] < DEV_CUT])},
            "HOLDOUT": {**_metrics(hold),
                        "portfolio": portfolio(tr[tr["ts"] >= DEV_CUT])}}


# ----------------------------------------------------------------------------- event study
def event_study(ev: pd.DataFrame, kcache: dict) -> dict:
    """Raw forward PRICE returns close_{i+h}/close_i - 1 (long convention).
    positive = pump continued (bad for short); P(down) = short-favourable."""
    res = {}
    for h in HORIZONS:
        rr = []
        for r in ev.itertuples(index=False):
            k = kcache[r.symbol]
            i = r.i
            if i + h >= len(k):
                continue
            c0 = float(k["close"].iloc[i])
            ch = float(k["close"].iloc[i + h])
            rr.append(ch / c0 - 1.0)
        a = np.array(rr)
        res[f"{h}h"] = {"n": int(len(a)),
                        "mean_pricefwd": round(float(a.mean()), 4) if len(a) else None,
                        "median_pricefwd": round(float(np.median(a)), 4) if len(a) else None,
                        "p_down": round(float((a < 0).mean()), 3) if len(a) else None}
    return res


# ----------------------------------------------------------------------------- random control
def random_bars(pairs: list[str], kcache_full: dict, n_target: int) -> pd.DataFrame:
    """~n_target random symbol-bars passing ONLY the $1M liquidity filter."""
    rng = np.random.default_rng(SEED)
    pool = []
    for p in pairs:
        k = kcache_full.get(p)
        if k is None:
            continue
        dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
        liq_ok = (dvol30.reindex(k.index, method="ffill") > 1e6).to_numpy()
        n = len(k)
        idx = np.nonzero(liq_ok)[0]
        idx = idx[(idx > 6) & (idx + 1 + MAX_HOLD < n)]
        for i in idx:
            pool.append((p, int(i)))
    pool = np.array(pool, dtype=object)
    if len(pool) > n_target:
        sel = rng.choice(len(pool), size=n_target, replace=False)
        pool = pool[sel]
    rows = []
    for p, i in pool:
        k = kcache_full[p]
        rows.append({"symbol": p, "ts": k.index[i], "i": int(i),
                     "ret6": 0.0, "trig_close": float(k["close"].iloc[i])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main() -> None:
    pairs = [c.pair for c in load_universe()]
    print(f"universe: {len(pairs)} pairs")
    print("loading funding ...", flush=True)
    fundmap = load_funding_map(pairs)
    print(f"funding loaded for {len(fundmap)} symbols", flush=True)

    # detect OI-collapse events + price-only events; cache klines once per symbol
    ev_oi, ev_px, kcache, kcache_full = [], [], {}, {}
    for i, pair in enumerate(pairs, 1):
        eo = detect_events(pair, use_oi=True)
        ep = detect_events(pair, use_oi=False)
        # full OHLC cache (needed by every symbol for random control + sim)
        kf = pd.read_parquet(KL_DIR / f"{pair}.parquet",
                             columns=["open_time", "open", "high", "low", "close",
                                      "quote_volume"])
        kf["ts"] = pd.to_datetime(kf["open_time"], unit="ms", utc=True)
        kf = kf.set_index("ts").sort_index()
        kcache_full[pair] = kf
        kcache[pair] = kf
        if len(eo):
            ev_oi.append(eo)
        if len(ep):
            ev_px.append(ep)
        if i % 40 == 0:
            print(f"[{i}/{len(pairs)}] oi_events={sum(len(x) for x in ev_oi)} "
                  f"px_events={sum(len(x) for x in ev_px)}", flush=True)

    ev_oi = pd.concat(ev_oi, ignore_index=True).sort_values("ts").reset_index(drop=True)
    ev_px = pd.concat(ev_px, ignore_index=True).sort_values("ts").reset_index(drop=True)
    print(f"OI-collapse events (+8%): {len(ev_oi)}   price-only (+8%): {len(ev_px)}",
          flush=True)

    # ---- event study (before costs) --------------------------------------------------
    ev8 = ev_oi
    es = {"events_+8pct_OIcollapse": event_study(ev8, kcache),
          "control_price_only_+8pct": event_study(ev_px, kcache)}
    rnd_ev = random_bars(pairs, kcache_full, RANDOM_N)
    es["control_random_bars"] = event_study(rnd_ev, kcache)
    print("\n=== EVENT STUDY (forward price ret, no trading) ===")
    for k, v in es.items():
        print(k)
        for h, d in v.items():
            print(f"  {h:>4}: mean={d['mean_pricefwd']} med={d['median_pricefwd']} "
                  f"P(down)={d['p_down']} n={d['n']}")

    # ---- base grid: 2 severities x 3 holds x 2 entries -------------------------------
    entries = [("market", 0.0025), ("maker", 0.0010)]
    grid_reports = {}
    for T, deep in [(0.08, False), (0.12, True)]:
        for hold in [12, 24, 48]:
            for em, cost in entries:
                tag = f"T{int(T*100)}_h{hold}_{em}"
                tr = simulate(ev_oi, kcache, fundmap, em, hold, cost, deep)
                grid_reports[tag] = {"config": {"T": T, "hold": hold, "entry": em,
                                                "rt_cost_bps": cost * 1e4, "deep": deep},
                                     "report": report(tr, tag)}
                print(f"  {tag:<18} filled={grid_reports[tag]['report']['n_filled']:>4} "
                      f"DEV net={grid_reports[tag]['report']['DEV'].get('net_mean')} "
                      f"HOLD net={grid_reports[tag]['report']['HOLDOUT'].get('net_mean')}",
                      flush=True)

    # ---- pick DEV-best (DEV net_mean, DEV n>=30; tie-break DEV portfolio CAGR) --------
    def dev_key(item):
        rep = item[1]["report"]["DEV"]
        n = rep.get("n", 0)
        nm = rep.get("net_mean")
        cagr = (rep.get("portfolio") or {}).get("cagr", -9)
        if n < 30 or nm is None:
            return (-1e9, -1e9)
        return (nm, cagr if cagr is not None else -9)
    best_tag = max(grid_reports.items(), key=dev_key)[0]
    best = grid_reports[best_tag]
    print(f"\nDEV-best cell: {best_tag}  config={best['config']}")

    # ---- 50bps sensitivity on the DEV-best cell --------------------------------------
    cfg = best["config"]
    tr50 = simulate(ev_oi, kcache, fundmap, cfg["entry"], cfg["hold"], 0.0050, cfg["deep"])
    sens50 = report(tr50, f"{best_tag}__cost50bps")

    # ---- controls --------------------------------------------------------------------
    ctrl_price = report(simulate(ev_px, kcache, fundmap, "market", 24, 0.0025, False),
                        "control_price_only_h24_market25")
    ctrl_rnd = report(simulate(rnd_ev, kcache, fundmap, "market", 24, 0.0025, False),
                      "control_random_h24_market25")
    print(f"control price-only : DEV net={ctrl_price['DEV'].get('net_mean')} "
          f"HOLD net={ctrl_price['HOLDOUT'].get('net_mean')}")
    print(f"control random     : DEV net={ctrl_rnd['DEV'].get('net_mean')} "
          f"HOLD net={ctrl_rnd['HOLDOUT'].get('net_mean')}")

    # ---- artifact --------------------------------------------------------------------
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "direction": "SHORT",
            "event": "ret_6h>=+T AND OI_6h_change<=-0.10, $1M liq, 24h cooldown, 1h grid",
            "net_formula": "(entry/exit - 1) - rt_cost + funding_in(entry_ts, exit_ts]",
            "grid": {"T": [0.08, 0.12], "hold": [12, 24, 48],
                     "entry": ["market@next_open/25bps", "maker@trigclose/10bps"],
                     "sensitivity": "50bps RT on DEV-best cell only"},
            "portfolio": {"slots": SLOTS, "eq_per_slot": "1/15"},
            "dev_cut": DEV_CUT.isoformat(),
            "selection": "max DEV net_mean, DEV n>=30, tie-break DEV portfolio CAGR",
            "controls": ["price_only_+8pct_noOI", f"random_{RANDOM_N}_bars"],
            "caveat": "universe picked 2026 -> survivorship bias inflates pre-2024",
        },
        "counts": {"oi_events": int(len(ev_oi)), "price_only_events": int(len(ev_px)),
                   "random_bars": int(len(rnd_ev))},
        "event_study": es,
        "grid": grid_reports,
        "dev_best": {"tag": best_tag, "config": best["config"],
                     "report": best["report"], "sens_50bps": sens50},
        "controls": {"price_only": ctrl_price, "random": ctrl_rnd},
    }
    (ART_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str),
                                          encoding="utf-8")
    print(f"\nartifacts -> {ART_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
