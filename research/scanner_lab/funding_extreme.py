"""Funding-extreme mean reversion (one-shot, pre-registered).

HYPOTHESIS (never tested in this lab): when perp funding is extremely NEGATIVE,
the crowd is short and paying to stay short -> squeeze fuel is loaded -> go LONG.
The long ALSO receives the negative funding while it holds. This is a DIRECTIONAL
price-reversion study conditioned on funding, NOT delta-neutral carry (carry is
dead here, ~0-3% APR). Known lab fact: average funding was negative in 2025-26, so
"extreme" is defined RELATIVE via an annualized threshold, and a fixed threshold
mechanically catches more events in the negative-funding regime (reported as
context: median trailing funding by year).

PRE-REGISTERED SPEC (frozen in this docstring BEFORE the first run):
  Signal at settlement time t for symbol s: trailing MEAN funding rate over the
  last K settlements, annualized (mean_rate * 3 * 365), <= -X.
    grid: K in {3, 9, 21}; X in {10%, 30%};
    stabilization in {none, prior-24h price return >= 0 at t}  (avoid falling knife).
  Trade: enter LONG at the next 1h open AFTER t (bar opening at t+1h; no lookahead
    since funding at t settles at t). Hold in {24h, 72h} to close (exit at close of
    the bar covering hour H). Costs 25bps round-trip. Funding pnl for the LONG =
    MINUS sum of funding rates at settlements inside (entry, exit] (a long receives
    funding when funding is negative). 48h per-symbol cooldown.
  Liquidity filter (all cells + controls): 30d-median daily quote_volume > $1M.
  Full grid = K x X x stab x hold = 3*2*2*2 = 24 cells. Per-event stats
    (net mean incl. funding, median, win, n, p5) reported for EVERY cell.
  Portfolio: 15-slot equal-weight (liqrev_v2 pattern; slot frees at actual exit)
    for the DEV-best cell only; net return per trade INCLUDES funding pnl.
  Controls (mandatory):
    (a) random control: sample matched N settlement-bars (same liquidity filter,
        same 149-coin universe, 24h hold, incl. cost + funding) -> is the signal
        better than a random long? [note: sampled from settlement-aligned bars, so
        entries land at 01/09/17 UTC; this is the honest matched null.]
    (b) MIRROR diagnostic: extreme POSITIVE trailing funding (annualized >= +30%,
        K=9) -> report RAW forward 24h/72h price-return means only (no cost, no
        funding, no sim). Is crowded-long the sell signal?
  Also: event counts per year per cell; a cell firing <100 times in 4y is FLAGGED
    as too thin (unreliable).

HONESTY PROTOCOL:
  DEV = events with settlement t before 2025-01-01. HOLDOUT = t on/after 2025-01-01.
  The single best cell is selected on DEV ONLY (max DEV net-mean among non-thin
  cells, DEV n>=100). VERDICT = that exact cell evaluated on HOLDOUT. The full
  24-cell grid is reported for BOTH windows as a diagnostic (not for selection).
  By-year reported for the DEV-best cell. Portfolio run on the DEV-best config over
  the full sample.
CAVEATS (stated up front):
  - Survivorship bias: the 149-coin universe was picked in 2026; coins that died
    are absent, which flatters any long-reversion signal.
  - Negative-funding regime: fixed -X threshold fires more often in 2025-26 by
    construction; interpret event-count growth and DEV->HOLDOUT drift with that lens.
  - Single-name funding pnl uses realized settlement rates; slippage beyond the flat
    25bps, borrow/ADL, and partial-fill risk are NOT modelled.

Usage: python funding_extreme.py   (run from research/scanner_lab)
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

FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
KL_DIR = REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h"
ART_DIR = REPO_ROOT / "research" / "data" / "fundext"

SETTLES_PER_YEAR = 3 * 365          # 1095; matches carry_study annualization
RT_COST = 0.0025                    # 25bps round-trip
COOLDOWN_H = 48
LIQ_MIN = 1e6
SLOTS = 15
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")

K_GRID = [3, 9, 21]
X_GRID = [0.10, 0.30]               # annualized trailing funding threshold (<= -X)
STAB_GRID = ["none", "up24"]        # up24 = prior-24h price return >= 0 at t
HOLD_GRID = [24, 72]                # hold in hours
RNG_SEED = 42


# ----------------------------------------------------------------------------- #
def build_candidates() -> pd.DataFrame:
    """Master per-settlement candidate table across the universe (liquidity-passed,
    entry bar present). Columns cover every cell + control + mirror so each cell can
    dropna on exactly the fields it needs."""
    rows = []
    pairs = [c.pair for c in load_universe()]
    for i, pair in enumerate(pairs, 1):
        fp, kp = FUND_DIR / f"{pair}.parquet", KL_DIR / f"{pair}.parquet"
        if not fp.exists() or not kp.exists():
            continue
        f = pd.read_parquet(fp, columns=["fundingTime", "fundingRate"])
        ts = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("1h")
        r = pd.Series(f["fundingRate"].to_numpy(), index=ts)
        r = r[~r.index.duplicated(keep="last")].sort_index()
        if len(r) < 25:
            continue
        k = pd.read_parquet(kp, columns=["open_time", "open", "close", "quote_volume"])
        k.index = pd.to_datetime(k["open_time"], unit="ms", utc=True)
        k = k[~k.index.duplicated(keep="last")].sort_index()

        # liquidity: 30d median of daily quote_volume, hourly ffill
        dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
        liq_h = dvol30.reindex(k.index, method="ffill")

        t = r.index
        apr3 = (r.rolling(3).mean() * SETTLES_PER_YEAR).to_numpy()
        apr9 = (r.rolling(9).mean() * SETTLES_PER_YEAR).to_numpy()
        apr21 = (r.rolling(21).mean() * SETTLES_PER_YEAR).to_numpy()
        # LONG funding pnl = -sum(next m settlement rates after t); m=3 (24h), 9 (72h)
        fs3 = r.rolling(3).sum().shift(-3).to_numpy()
        fs9 = r.rolling(9).sum().shift(-9).to_numpy()

        oc = k["open"]
        cc = k["close"]
        entry_px = oc.reindex(t + pd.Timedelta("1h")).to_numpy()      # next 1h open
        px_t = oc.reindex(t).to_numpy()                              # price at t
        px_pr = oc.reindex(t - pd.Timedelta("24h")).to_numpy()       # 24h before t
        c24 = cc.reindex(t + pd.Timedelta("24h")).to_numpy()          # close of 24h bar
        c72 = cc.reindex(t + pd.Timedelta("72h")).to_numpy()          # close of 72h bar
        liq_ok = (liq_h.reindex(t).to_numpy() > LIQ_MIN)

        df = pd.DataFrame({
            "symbol": pair, "t": t, "entry_ts": t + pd.Timedelta("1h"),
            "apr3": apr3, "apr9": apr9, "apr21": apr21,
            "ret_prior24": px_t / px_pr - 1.0,
            "ret24": c24 / entry_px - 1.0, "ret72": c72 / entry_px - 1.0,
            "fpnl24": -fs3, "fpnl72": -fs9,
            "entry_px": entry_px, "liq_ok": liq_ok,
        })
        df = df[df["liq_ok"] & np.isfinite(df["entry_px"])]
        rows.append(df.drop(columns=["entry_px", "liq_ok"]))
        if i % 40 == 0:
            print(f"[{i}/{len(pairs)}] candidates so far: "
                  f"{sum(len(x) for x in rows)}", flush=True)
    out = pd.concat(rows, ignore_index=True).sort_values("t").reset_index(drop=True)
    print(f"master candidates (liquidity-passed): {len(out)}", flush=True)
    return out


def cooldown(df: pd.DataFrame, hours: int = COOLDOWN_H) -> pd.DataFrame:
    """Greedy per-symbol cooldown: keep an event only if >= `hours` since last kept.
    Linear pass over (symbol, t)-sorted rows."""
    if df.empty:
        return df
    d = df.sort_values(["symbol", "t"])
    syms = d["symbol"].to_numpy()
    ts = d["t"].to_numpy()                       # datetime64[ns] (UTC)
    gap = np.timedelta64(hours, "h")
    keep = np.zeros(len(d), dtype=bool)
    last_sym, last_t = None, None
    for j in range(len(d)):
        if syms[j] != last_sym or (ts[j] - last_t) >= gap:
            keep[j] = True
            last_sym, last_t = syms[j], ts[j]
    return d[keep]


def stats(net: np.ndarray) -> dict:
    net = net[np.isfinite(net)]
    n = int(len(net))
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "win": None, "p5": None}
    return {"n": n, "mean": round(float(np.mean(net)), 4),
            "median": round(float(np.median(net)), 4),
            "win": round(float(np.mean(net > 0)), 3),
            "p5": round(float(np.quantile(net, 0.05)), 4)}


def cell_events(master: pd.DataFrame, K: int, X: float, stab: str,
                hold: int) -> pd.DataFrame:
    apr = master[f"apr{K}"]
    ret, fp = master[f"ret{hold}"], master[f"fpnl{hold}"]
    m = apr.notna() & (apr <= -X) & ret.notna() & fp.notna()
    if stab == "up24":
        m = m & (master["ret_prior24"] >= 0)
    sub = master[m].copy()
    sub = cooldown(sub)
    sub["net"] = sub[f"ret{hold}"] - RT_COST + sub[f"fpnl{hold}"]
    return sub


def portfolio(ev: pd.DataFrame, hold: int) -> dict:
    """15-slot equal-weight, slot frees at actual exit (liqrev_v2 pattern)."""
    t = ev.sort_values("entry_ts")
    exit_ts = t["entry_ts"] + pd.Timedelta(f"{hold}h")
    eq, busy, curve, n_taken = 1.0, [], [], 0
    for (etime, xtime, ret) in zip(t["entry_ts"], exit_ts, t["net"]):
        busy = [b for b in busy if b > etime]
        if len(busy) < SLOTS:
            eq *= (1 + ret / SLOTS)
            busy.append(xtime)
            n_taken += 1
        curve.append((etime, eq))
    c = pd.Series(dict(curve))
    if c.empty:
        return {}
    years = (c.index[-1] - c.index[0]).days / 365.25
    mo = c.resample("MS").last().ffill().pct_change().dropna()
    yr = c.groupby(c.index.year).last() / c.groupby(c.index.year).first() - 1
    return {"total": round(float(eq - 1), 4),
            "cagr": round(float(eq ** (1 / years) - 1), 4) if years > 0 else None,
            "maxDD": round(float((c / c.cummax() - 1).min()), 4),
            "worst_month": round(float(mo.min()), 4) if len(mo) else None,
            "n_taken": n_taken,
            "by_year": {str(k): round(float(v), 3) for k, v in yr.items()}}


def by_year_counts(ev: pd.DataFrame) -> dict:
    return {str(k): int(v) for k, v in ev.groupby(ev["t"].dt.year).size().items()}


def by_year_net(ev: pd.DataFrame) -> dict:
    g = ev.groupby(ev["t"].dt.year)["net"]
    return {str(y): {"n": int(len(v)), "net_mean": round(float(v.mean()), 4)}
            for y, v in g}


# ----------------------------------------------------------------------------- #
def main() -> None:
    master = build_candidates()

    # context: median trailing funding (annualized, K=9) by year
    med_fund = (master.dropna(subset=["apr9"])
                .groupby(master["t"].dt.year)["apr9"].median().round(4))
    print("\n=== CONTEXT: median trailing funding APR (K=9) by year ===")
    print(med_fund.to_string())

    # ---- full 24-cell grid, DEV vs HOLDOUT ----
    cells = []
    print("\n=== GRID: per-event net (incl. funding) DEV | HOLDOUT ===")
    print(f"{'cell':<26}{'DEVn':>6}{'DEVmean':>9}{'DEVwin':>7}"
          f"{'HOLDn':>7}{'HOLDmean':>9}{'HOLDwin':>7}  flag")
    for K in K_GRID:
        for X in X_GRID:
            for stab in STAB_GRID:
                for hold in HOLD_GRID:
                    ev = cell_events(master, K, X, stab, hold)
                    dev = ev[ev["t"] < SPLIT]
                    hol = ev[ev["t"] >= SPLIT]
                    sd, sh = stats(dev["net"].to_numpy()), stats(hol["net"].to_numpy())
                    thin = len(ev) < 100
                    tag = f"K{K}_X{int(X*100)}_{stab}_{hold}h"
                    cells.append({
                        "tag": tag, "K": K, "X": X, "stab": stab, "hold": hold,
                        "n_total": int(len(ev)), "thin": bool(thin),
                        "by_year_counts": by_year_counts(ev),
                        "dev": sd, "holdout": sh})
                    print(f"{tag:<26}{sd['n']:>6}"
                          f"{(sd['mean'] if sd['mean'] is not None else 0):>9.4f}"
                          f"{(sd['win'] if sd['win'] is not None else 0):>7.2f}"
                          f"{sh['n']:>7}"
                          f"{(sh['mean'] if sh['mean'] is not None else 0):>9.4f}"
                          f"{(sh['win'] if sh['win'] is not None else 0):>7.2f}"
                          f"  {'THIN' if thin else ''}")

    # ---- DEV-best selection (non-thin, DEV n>=100, max DEV net mean) ----
    elig = [c for c in cells if not c["thin"] and c["dev"]["n"] >= 100
            and c["dev"]["mean"] is not None]
    best = max(elig, key=lambda c: c["dev"]["mean"]) if elig else None
    print("\n=== DEV-BEST CELL (selected on DEV only) ===")
    if best is None:
        print("no eligible non-thin cell with DEV n>=100")
    else:
        print(f"config: {best['tag']}  (K={best['K']}, X={best['X']}, "
              f"stab={best['stab']}, hold={best['hold']}h)")
        print(f"DEV: {best['dev']}")
        print(f"HOLDOUT verdict: {best['holdout']}")

    best_ev, best_by_year, port = pd.DataFrame(), {}, {}
    if best is not None:
        best_ev = cell_events(master, best["K"], best["X"], best["stab"], best["hold"])
        best_by_year = by_year_net(best_ev)
        port = portfolio(best_ev, best["hold"])
        print("\nby-year net (DEV-best, full sample):")
        for y, v in best_by_year.items():
            print(f"  {y}: n={v['n']:>4}  net_mean={v['net_mean']:+.4f}")
        print(f"\nportfolio 15-slot (full sample): {port}")

    # ---- control (a): matched random settlement-bars, 24h hold ----
    pool = master.dropna(subset=["ret24", "fpnl24"]).copy()
    n_match = int(len(best_ev)) if best is not None else 1000
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(pool), size=min(n_match, len(pool)), replace=False)
    rc = pool.iloc[idx].copy()
    rc["net"] = rc["ret24"] - RT_COST + rc["fpnl24"]
    rc_dev = stats(rc[rc["t"] < SPLIT]["net"].to_numpy())
    rc_hol = stats(rc[rc["t"] >= SPLIT]["net"].to_numpy())
    rc_all = stats(rc["net"].to_numpy())
    print("\n=== CONTROL (a): random matched settlement-bars, 24h hold, incl. funding ===")
    print(f"matched N={n_match}  ALL={rc_all}\n  DEV={rc_dev}\n  HOLDOUT={rc_hol}")

    # ---- control (b): MIRROR, extreme POSITIVE funding, raw forward returns ----
    mir = master.dropna(subset=["apr9", "ret24", "ret72"])
    mir = mir[mir["apr9"] >= 0.30]
    def raw(sub):
        return {"n": int(len(sub)),
                "raw_fwd24_mean": round(float(sub["ret24"].mean()), 4) if len(sub) else None,
                "raw_fwd72_mean": round(float(sub["ret72"].mean()), 4) if len(sub) else None,
                "raw_fwd24_median": round(float(sub["ret24"].median()), 4) if len(sub) else None,
                "raw_fwd72_median": round(float(sub["ret72"].median()), 4) if len(sub) else None}
    mir_dev, mir_hol, mir_all = (raw(mir[mir["t"] < SPLIT]),
                                 raw(mir[mir["t"] >= SPLIT]), raw(mir))
    print("\n=== CONTROL (b): MIRROR extreme POSITIVE funding (K9 APR>=+30%), RAW fwd ===")
    print(f"ALL={mir_all}\n  DEV={mir_dev}\n  HOLDOUT={mir_hol}")

    # ---- persist ----
    ART_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {"K_grid": K_GRID, "X_grid": X_GRID, "stab_grid": STAB_GRID,
                 "hold_grid": HOLD_GRID, "rt_cost": RT_COST, "cooldown_h": COOLDOWN_H,
                 "liq_min_usd": LIQ_MIN, "slots": SLOTS, "split": str(SPLIT),
                 "settles_per_year": SETTLES_PER_YEAR},
        "median_trailing_funding_apr_by_year": {str(k): float(v)
                                                for k, v in med_fund.items()},
        "grid": cells,
        "dev_best": (None if best is None else {
            "config": {k: best[k] for k in ("tag", "K", "X", "stab", "hold")},
            "dev": best["dev"], "holdout_verdict": best["holdout"],
            "by_year_net": best_by_year, "portfolio_15slot": port}),
        "control_random_24h": {"matched_n": n_match, "all": rc_all,
                               "dev": rc_dev, "holdout": rc_hol},
        "control_mirror_positive_funding": {"all": mir_all, "dev": mir_dev,
                                            "holdout": mir_hol},
        "caveats": ["universe picked 2026 -> survivorship bias flatters long reversion",
                    "negative-funding regime: fixed -X threshold fires more in 2025-26",
                    "funding pnl uses realized rates; slippage/borrow/ADL not modelled"],
    }
    (ART_DIR / "results.json").write_text(json.dumps(result, indent=2, default=str),
                                          encoding="utf-8")
    print(f"\nartifacts -> {ART_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
