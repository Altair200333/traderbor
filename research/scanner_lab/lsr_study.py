"""Positioning-ratio extremes -> forward returns: pre-registered bucket study.

============================ PRE-REGISTERED SPEC ============================
Frozen 2026-07-08 BEFORE the first run of this script. Column inspection of
BTCUSDT/ADAUSDT/AGLDUSDT metrics_5m parquets was performed to enumerate the
ratio columns (allowed pre-registration); NO results were looked at first.

HYPOTHESIS (untested, exploratory): positioning-ratio extremes predict forward
returns -- e.g. fade the retail crowd / follow top traders. FIRST establish
whether ANY signal exists (bucket study), only THEN simulate trades.

DATA (UTC): research/data/perp/metrics_5m/{PAIR}.parquet, 5m cadence,
span ~2022-07-01 .. 2026-07-07. 1h klines: research/data/v3/klines/1h/{PAIR}.
Funding (only used if a sim runs): research/data/perp/funding/{PAIR}.parquet.
Universe: 149 coins from `universe.load_universe`.

RATIO COLUMNS studied (all that exist in metrics_5m; float ratios):
  - count_toptrader_long_short_ratio   (top-trader long/short ACCOUNT ratio)
  - sum_toptrader_long_short_ratio     (top-trader long/short POSITION ratio)
  - count_long_short_ratio             (GLOBAL long/short account ratio)
  - sum_taker_long_short_vol_ratio     (taker buy/sell VOLUME ratio)
  (toptrader columns start ~2022-09 on the oldest coins, with scattered gaps;
   handled by the rolling min_periods + NaN drop, never lookahead-filled.)

PIPELINE (per symbol, per ratio column):
  1. resample 5m ratio -> 1h using last(); reindex onto a COMPLETE hourly grid
     spanning the klines (guarantees shift(-h) == exactly h hours ahead), then
     ffill(limit=12h) only to bridge tiny gaps (never across long outages).
  2. pctrank = ratio.rolling(720h=30d, min_periods=360h=15d).rank(pct=True)
     -> trailing-window percentile rank of the CURRENT value, point-in-time.
  3. liquidity filter: daily quote_volume sum -> rolling(30d).median() > $1M,
     reindexed to 1h ffill; observations failing this are dropped.
  4. forward log returns fwd_h = log(close.shift(-h)/close), h in {4,12,24,48}h.
  5. sample every 4h (positional i%4==0 on the complete hourly grid) to reduce
     overlap; report n per bucket.

BUCKET STUDY (core): buckets = fixed quintile edges on pctrank
  [0,.2,.4,.6,.8,1.0] -> Q1..Q5 (pctrank is ~uniform by construction, so fixed
  edges == equal-frequency quintiles and are fully deterministic). Per ratio x
  horizon x window: Q1..Q5 mean forward log return + n, plus 'all' (pooled)
  mean as the per-ratio drift baseline, plus Q5-Q1 spread (bps). Windows:
  DEV (< 2025-01-01), HOLDOUT (>= 2025-01-01), FULL.

CONTROL: random-bars forward means at the same horizons per window = mean fwd
  return over ALL sampled liquid bars of the taker universe (0% NaN coverage),
  for scale.

DECISION RULE (pre-registered, per ratio): a ratio "has signal" IFF, on DEV:
  (a) |Q5-Q1| spread at the 24h horizon >= 50 bps (0.0050 log), AND
  (b) bucket means Q1..Q5 are roughly monotone := >= 3 of the 4 adjacent
      steps move in the direction of sign(Q5-Q1), AND
  (c) the HOLDOUT 24h spread has the SAME SIGN as the DEV 24h spread.
  Otherwise: NO signal, NO simulation for that ratio.

MULTIPLE-COMPARISONS NOTE: ~4 ratio columns x 4 horizons = ~16 tests; the bar
  for "signal" is already Bonferroni-thin. That is exactly why the decision rule
  demands DEV+HOLDOUT sign agreement AND monotonicity -- a single lucky 24h
  spread is not enough. Treat any pass as a candidate to be re-checked live, not
  a confirmed edge.

EVENT SIM (only for a ratio that PASSES the decision rule): trade the extreme
  bucket. Event = pctrank CROSSING into <5th (from >=5th) or into >95th (from
  <=95th) rolling percentile. Direction per DEV 24h spread sign s=sign(Q5-Q1):
  high-cross (>95th) position = +s ; low-cross (<5th) position = -s (high
  pctrank predicts sign(s) return; low pctrank predicts the opposite end).
  24h per-symbol cooldown, entry at NEXT 1h open, hold 24h (exit close of
  entry_bar+23), 25 bps round-trip cost, 15-slot portfolio (liqrev_v2 pattern,
  1/15 equity/slot, slot frees at exit). SHORTS receive realized funding:
  funding_pnl = -pos * sum(fundingRate over [entry_ts, exit_ts)). Report
  DEV/HOLDOUT split + by-year.

Live shadow (small size) remains the final validator regardless of outcome.

Usage: python lsr_study.py   (runtime target < 20 min; prints progress)
=============================================================================
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

METRICS_DIR = REPO_ROOT / "research" / "data" / "perp" / "metrics_5m"
KL_DIR = REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h"
FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
ART_DIR = REPO_ROOT / "research" / "data" / "lsr"

RATIO_COLS = [
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]
HORIZONS = [4, 12, 24, 48]
WIN_SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
ROLL_W, ROLL_MIN = 720, 360          # 30d / 15d in hours
SAMPLE_EVERY = 4                     # sample every 4h
QEDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
QLABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
LIQ_MIN = 1e6
# sim constants
HOLD_BARS, SLOTS, RT_COST = 24, 15, 0.0025
LO_X, HI_X = 0.05, 0.95


# ---------------------------------------------------------------- data load
def load_symbol(pair: str):
    """Return (kframe on complete hourly grid, {ratio: pctrank series}, liq mask,
    {h: fwd log ret series}) or None if data missing."""
    kp, mp = KL_DIR / f"{pair}.parquet", METRICS_DIR / f"{pair}.parquet"
    if not kp.exists() or not mp.exists():
        return None
    k = pd.read_parquet(kp, columns=["open_time", "open", "high", "low",
                                     "close", "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    k = k[~k.index.duplicated(keep="last")]
    grid = pd.date_range(k.index.min(), k.index.max(), freq="1h", tz="UTC")
    k = k.reindex(grid)
    close = k["close"]

    # liquidity mask (30d median daily quote_volume > $1M)
    dvol = k["quote_volume"].resample("1D").sum()
    liq = (dvol.rolling(30, min_periods=20).median() > LIQ_MIN)
    liq_h = liq.reindex(grid, method="ffill").fillna(False)

    # forward log returns
    fwd = {h: np.log(close.shift(-h) / close) for h in HORIZONS}

    # ratio pctranks
    m = pd.read_parquet(mp, columns=["ts_ms"] + RATIO_COLS)
    m.index = pd.to_datetime(m["ts_ms"], unit="ms", utc=True)
    ranks = {}
    for col in RATIO_COLS:
        r1h = m[col].resample("1h").last().reindex(grid).ffill(limit=12)
        ranks[col] = r1h.rolling(ROLL_W, min_periods=ROLL_MIN).rank(pct=True)
    return k, ranks, liq_h, fwd, grid


def build_observations(pairs):
    """Long-format obs per ratio: symbol, ts, pctrank, bucket, fwd_h..."""
    per_ratio = {c: [] for c in RATIO_COLS}
    taker_bars = []  # random-bars control universe (taker = full coverage)
    for i, pair in enumerate(pairs, 1):
        res = load_symbol(pair)
        if res is None:
            continue
        k, ranks, liq_h, fwd, grid = res
        n = len(grid)
        pos = np.arange(n)
        samp = (pos % SAMPLE_EVERY == 0) & liq_h.to_numpy()
        fwd_arr = {h: fwd[h].to_numpy() for h in HORIZONS}
        for col in RATIO_COLS:
            pr = ranks[col].to_numpy()
            mask = samp & np.isfinite(pr)
            # require at least the 24h fwd (primary decision horizon) present
            mask = mask & np.isfinite(fwd_arr[24])
            if not mask.any():
                continue
            d = {"symbol": pair,
                 "ts": grid[mask],
                 "pctrank": pr[mask]}
            for h in HORIZONS:
                d[f"fwd_{h}"] = fwd_arr[h][mask]
            per_ratio[col].append(pd.DataFrame(d))
        # control: taker universe sampled liquid bars (any horizon)
        tmask = samp & np.isfinite(fwd_arr[24])
        if tmask.any():
            td = {"ts": grid[tmask]}
            for h in HORIZONS:
                td[f"fwd_{h}"] = fwd_arr[h][tmask]
            taker_bars.append(pd.DataFrame(td))
        if i % 25 == 0:
            tot = sum(sum(len(x) for x in v) for v in per_ratio.values())
            print(f"[obs {i}/{len(pairs)}] rows so far: {tot:,}", flush=True)
    out = {}
    for col in RATIO_COLS:
        if per_ratio[col]:
            df = pd.concat(per_ratio[col], ignore_index=True)
            df["bucket"] = pd.cut(df["pctrank"], QEDGES, labels=QLABELS,
                                  include_lowest=True)
            out[col] = df
    control = pd.concat(taker_bars, ignore_index=True) if taker_bars else None
    return out, control


# ---------------------------------------------------------------- bucketing
def win_of(ts: pd.Series) -> pd.Series:
    return np.where(ts < WIN_SPLIT, "DEV", "HOLDOUT")


def bucket_table(df: pd.DataFrame) -> dict:
    """{window: {horizon: {Q1..Q5:{mean_bps,n}, all_mean_bps, n_all,
    spread_bps, monotone_up_steps, monotone_dn_steps}}}"""
    df = df.copy()
    df["win"] = win_of(df["ts"])
    res = {}
    for wname, wdf in [("DEV", df[df["win"] == "DEV"]),
                       ("HOLDOUT", df[df["win"] == "HOLDOUT"]),
                       ("FULL", df)]:
        res[wname] = {}
        for h in HORIZONS:
            col = f"fwd_{h}"
            sub = wdf[["bucket", col]].dropna()
            g = sub.groupby("bucket", observed=True)[col]
            means = g.mean()
            ns = g.size()
            qd = {}
            for q in QLABELS:
                if q in means.index:
                    qd[q] = {"mean_bps": round(float(means[q] * 1e4), 2),
                             "n": int(ns[q])}
                else:
                    qd[q] = {"mean_bps": None, "n": 0}
            vals = [means.get(q, np.nan) for q in QLABELS]
            spread = (vals[4] - vals[0]) if np.isfinite(vals[0]) and \
                np.isfinite(vals[4]) else np.nan
            steps = np.diff([v if np.isfinite(v) else np.nan for v in vals])
            up = int(np.nansum(steps > 0))
            dn = int(np.nansum(steps < 0))
            qd["all_mean_bps"] = round(float(sub[col].mean() * 1e4), 2) \
                if len(sub) else None
            qd["n_all"] = int(len(sub))
            qd["spread_bps"] = round(float(spread * 1e4), 2) \
                if np.isfinite(spread) else None
            qd["monotone_up_steps"] = up
            qd["monotone_dn_steps"] = dn
            res[wname][f"{h}h"] = qd
    return res


def decide(tbl: dict) -> dict:
    """Pre-registered decision rule at 24h."""
    dev = tbl["DEV"]["24h"]
    hld = tbl["HOLDOUT"]["24h"]
    ds = dev["spread_bps"]
    hs = hld["spread_bps"]
    if ds is None or hs is None:
        return {"passed": False, "reason": "missing spread",
                "dev_spread_24h_bps": ds, "holdout_spread_24h_bps": hs}
    s = np.sign(ds)
    steps_dir = dev["monotone_up_steps"] if s > 0 else dev["monotone_dn_steps"]
    cond_mag = abs(ds) >= 50.0
    cond_mono = steps_dir >= 3
    cond_sign = np.sign(hs) == s and hs != 0
    passed = bool(cond_mag and cond_mono and cond_sign)
    reason = []
    if not cond_mag:
        reason.append(f"|DEV spread| {abs(ds):.1f}bps < 50")
    if not cond_mono:
        reason.append(f"DEV monotone steps {steps_dir}/4 < 3")
    if not cond_sign:
        reason.append("HOLDOUT sign disagrees")
    return {"passed": passed,
            "dev_spread_24h_bps": ds, "holdout_spread_24h_bps": hs,
            "dev_monotone_steps_in_dir": int(steps_dir),
            "dev_sign": int(s),
            "reason": "PASS" if passed else "; ".join(reason)}


def control_table(control: pd.DataFrame) -> dict:
    if control is None:
        return {}
    control = control.copy()
    control["win"] = win_of(control["ts"])
    res = {}
    for wname, wdf in [("DEV", control[control["win"] == "DEV"]),
                       ("HOLDOUT", control[control["win"] == "HOLDOUT"]),
                       ("FULL", control)]:
        res[wname] = {}
        for h in HORIZONS:
            v = wdf[f"fwd_{h}"].dropna()
            res[wname][f"{h}h"] = {"mean_bps": round(float(v.mean() * 1e4), 2)
                                   if len(v) else None,
                                   "se_bps": round(float(v.std() /
                                   np.sqrt(len(v)) * 1e4), 2) if len(v) else None,
                                   "n": int(len(v))}
    return res


# ---------------------------------------------------------------- event sim
def load_funding(pair: str) -> pd.DataFrame:
    fp = FUND_DIR / f"{pair}.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=["ft", "rate"])
    f = pd.read_parquet(fp, columns=["fundingTime", "fundingRate"])
    f["ft"] = pd.to_datetime(f["fundingTime"], unit="ms", utc=True)
    return f[["ft", "fundingRate"]].rename(columns={"fundingRate": "rate"}) \
        .sort_values("ft")


def sim_ratio(pairs, ratio_col: str, spread_sign: int) -> dict:
    """Event sim for one passing ratio; returns portfolio + splits."""
    trades = []
    for pair in pairs:
        res = load_symbol(pair)
        if res is None:
            continue
        k, ranks, liq_h, fwd, grid = res
        pr = ranks[ratio_col].to_numpy()
        liq = liq_h.to_numpy()
        opn = k["open"].to_numpy()
        cls = k["close"].to_numpy()
        n = len(grid)
        prev = np.r_[np.nan, pr[:-1]]
        hi_cross = (pr > HI_X) & (prev <= HI_X) & np.isfinite(prev)
        lo_cross = (pr < LO_X) & (prev >= LO_X) & np.isfinite(prev)
        fund = None
        last_i = -10**9
        for i in range(n):
            if not (hi_cross[i] or lo_cross[i]):
                continue
            if not liq[i]:
                continue
            if i - last_i < HOLD_BARS:      # 24h symbol cooldown (bars)
                continue
            if i + 1 + HOLD_BARS >= n:
                continue
            entry = opn[i + 1]
            exit_px = cls[i + 1 + HOLD_BARS - 1]
            if not (np.isfinite(entry) and np.isfinite(exit_px)) or entry <= 0:
                continue
            pos = spread_sign if hi_cross[i] else -spread_sign
            entry_ts, exit_ts = grid[i + 1], grid[i + 1 + HOLD_BARS - 1]
            gross = pos * (exit_px / entry - 1.0)
            fpnl = 0.0
            if pos < 0:  # only shorts need funding accrual (load lazily)
                if fund is None:
                    fund = load_funding(pair)
                if len(fund):
                    fr = fund[(fund["ft"] >= entry_ts) & (fund["ft"] < exit_ts)]
                    fpnl = -pos * float(fr["rate"].sum())
            net = gross - RT_COST + fpnl
            trades.append({"symbol": pair, "ts": entry_ts, "exit_ts": exit_ts,
                           "pos": int(pos), "ret": net})
            last_i = i
    if not trades:
        return {"n_trades": 0}
    tr = pd.DataFrame(trades).sort_values("ts").reset_index(drop=True)
    out = {"n_trades": int(len(tr)),
           "n_long": int((tr["pos"] > 0).sum()),
           "n_short": int((tr["pos"] < 0).sum()),
           "net_mean_bps": round(float(tr["ret"].mean() * 1e4), 2),
           "win": round(float((tr["ret"] > 0).mean()), 3),
           "portfolio_15slots": portfolio(tr),
           "DEV": _split(tr[tr["ts"] < WIN_SPLIT]),
           "HOLDOUT": _split(tr[tr["ts"] >= WIN_SPLIT])}
    return out


def _split(tr: pd.DataFrame) -> dict:
    if not len(tr):
        return {"n": 0}
    return {"n": int(len(tr)),
            "net_mean_bps": round(float(tr["ret"].mean() * 1e4), 2),
            "win": round(float((tr["ret"] > 0).mean()), 3),
            "portfolio_15slots": portfolio(tr)}


def portfolio(tr: pd.DataFrame) -> dict:
    t = tr.sort_values("ts")
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
    years = max((c.index[-1] - c.index[0]).days / 365.25, 1e-6)
    yearly = c.groupby(c.index.year).last() / c.groupby(c.index.year).first() - 1
    return {"total": round(float(eq - 1), 4),
            "cagr": round(float(eq ** (1 / years) - 1), 4),
            "maxDD": round(float((c / c.cummax() - 1).min()), 4),
            "n_taken": n_taken,
            "by_year": {str(y): round(float(v), 3) for y, v in yearly.items()}}


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = datetime.now(timezone.utc)
    pairs = [c.pair for c in load_universe()]
    print(f"universe: {len(pairs)} pairs; ratios: {RATIO_COLS}", flush=True)

    obs, control = build_observations(pairs)
    print("observations built:", {c: len(obs[c]) for c in obs}, flush=True)

    # coverage
    coverage = {}
    for col, df in obs.items():
        coverage[col] = {
            "first_ts": str(df["ts"].min()), "last_ts": str(df["ts"].max()),
            "n_obs": int(len(df)), "n_symbols": int(df["symbol"].nunique()),
            "dev_obs": int((df["ts"] < WIN_SPLIT).sum()),
            "holdout_obs": int((df["ts"] >= WIN_SPLIT).sum())}

    buckets, decisions = {}, {}
    for col, df in obs.items():
        buckets[col] = bucket_table(df)
        decisions[col] = decide(buckets[col])
        d = decisions[col]
        print(f"[decide] {col}: DEV24h={d.get('dev_spread_24h_bps')}bps "
              f"HOLD24h={d.get('holdout_spread_24h_bps')}bps "
              f"-> {'PASS' if d['passed'] else 'no'} ({d['reason']})",
              flush=True)

    ctrl = control_table(control)

    sims = {}
    for col, d in decisions.items():
        if d["passed"]:
            print(f"[sim] running event sim for {col} "
                  f"(dir sign={d['dev_sign']}) ...", flush=True)
            sims[col] = sim_ratio(pairs, col, int(d["dev_sign"]))
    if not sims:
        print("[sim] no ratio passed the decision rule -> no simulation.",
              flush=True)

    result = {
        "run_utc": t0.isoformat(),
        "spec": "pre-registered in module docstring (frozen before run)",
        "config": {"ratio_cols": RATIO_COLS, "horizons_h": HORIZONS,
                   "roll_window_h": ROLL_W, "roll_min_h": ROLL_MIN,
                   "sample_every_h": SAMPLE_EVERY, "qedges": QEDGES,
                   "win_split": str(WIN_SPLIT), "liq_min_usd": LIQ_MIN,
                   "decision": "DEV|Q5-Q1|@24h>=50bps AND >=3/4 monotone steps "
                               "AND HOLDOUT 24h sign==DEV sign",
                   "sim": {"hold_bars": HOLD_BARS, "slots": SLOTS,
                           "rt_cost": RT_COST, "extreme": [LO_X, HI_X],
                           "funding": "shorts accrue realized funding"}},
        "coverage": coverage,
        "buckets": buckets,
        "random_bars_control": ctrl,
        "decision": decisions,
        "sim": sims if sims else None,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    outp = ART_DIR / "results.json"
    outp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"\nartifacts -> {outp}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
