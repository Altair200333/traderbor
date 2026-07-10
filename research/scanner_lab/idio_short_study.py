"""Idiosyncratic-cascade SHORT study (one-shot, PRE-REGISTERED, question #13).

CONTEXT. The live lab strategy LONGS liquidation cascades (liqrev_v2.py). An
established conditional structure (research/data/liqrev/ml_dataset.parquet,
column btc_ret_6h) says: cascades triggered while BTC is dumping ("MARKET-WIDE",
btc_ret_6h <= -0.02) bounce hard (+2.3..+5.8%/ev by year); cascades triggered
while BTC is calm ("IDIOSYNCRATIC", btc_ret_6h > -0.02) drift NEGATIVE ~-1% over
the next 24h in 2024/2025/2026 (win 38-41%) but were POSITIVE in 2022-23
(+1.9..+3.9%). HYPOTHESIS UNDER TEST: shorting idiosyncratic cascades is a
deployable ANTI-strategy.

TWO PRE-REGISTERED KILL-QUESTIONS:
  Q1 SPECIFICITY. Does idio-short beat shorting RANDOM alt bars in the same
     period, or is it just alt-bear beta in disguise?
  Q2 REGIME. Is it profitable only in the 2024+ bear, and if so does an
     A-PRIORI regime gate (BTC close < BTC 200d SMA, computed as-of on the
     hourly grid) make it self-switching?

PRE-REGISTERED SPEC (frozen BEFORE the first run; nothing added after):
  - IDIO event: ml_dataset row with btc_ret_6h > -0.02 (the established cut,
    NOT tuned here). MW complement (<= -0.02) used only for control C1.
  - Bar indexing: an ml_dataset row's ts = trigger-bar OPEN (bar i). Trigger
    close = close[i]. "t+24h" = bar i+24 (opens 24 bars after trigger, closes
    t+25h). "t+25h" entry = open[i+25]. This makes S2's parenthetical entry
    time (t+25h) exact.
  - Variants (declared grid, ALL reported):
      S1 immediate short: entry MARKET at next 1h open after trigger
        (open[i+1]); hold {24, 48} bars, exit at close of last held bar;
        costs 25bps RT (10bps sensitivity on the best cell only).
      S2 failed-bounce short: condition close[i+24] < close[i]; if met, entry
        MARKET at open[i+25]; hold {24, 48}; same costs. Avoids the violent
        wick zone, adds confirmation. Report fraction of idio events that
        qualify.
  - Short PnL = entry/exit - 1 - cost + funding_pnl   (frozen convention:
    entry_price / exit_price - 1, the exact inverse-ratio of liqrev's long
    ret = exit/entry - 1). funding_pnl = +sum of funding rates at 8h
    settlements inside (entry_time, exit_time]; shorts RECEIVE positive
    funding, PAY negative (2025-26 alt climate is negative -> a real
    headwind). Loaded per symbol.
  - DISASTER stop mirrored from liqrev: exit if price rises 20% above entry.
    Gap-aware: when breached in a bar, exit at max(open, stop). Report stop
    rate and p5 of returns (bounce tails +15-23% are the short's tail risk).
  - Portfolio: 15 slots x 1/15 equity, slot frees at actual exit
    (liqrev_v2.portfolio pattern).
  - REGIME-GATED variant: S1-24h taken ONLY when BTC < its 200d SMA at trigger
    (a-priori gate; 200d SMA computed as a trailing 4800-hour rolling mean of
    BTC hourly close, evaluated as-of the trigger bar -> lookahead-free).
    Report gated AND ungated, incl. gate behaviour in 2022-23.
  - CONTROLS (mandatory):
      C1  short ALL cascade events incl. market-wide (expect worse; MW bounces
          kill shorts; confirms the conditioning matters).
      C2  RANDOM-SHORT baseline: ~300 random symbol-bars per year (~1500 total)
          passing the same $1M liquidity filter, short 24h, same costs+funding,
          same disaster stop; computed PER YEAR so each year's idio-short is
          compared to same-year random-short. Seed=13.
      C3  the long side of idio events (known ~-1%): simple 24h long
          (entry open[i+1], exit close[i+24], 25bps, no stop/funding) for
          reference/pipeline sanity.
  - Report EVERYTHING by year (2022..2026) plus DEV (<2025) vs 2025-26 split.

PRE-REGISTERED VERDICT RULE. PROMISING only if BOTH hold:
  (a) idio-short beats same-period random-short by >= 50bps/trade net in
      2024-26 (SPECIFICITY);
  (b) EITHER positive across all years, OR positive under the 200d-SMA gate
      including correct gate behaviour in 2022-23 (the gate must have kept us
      out of, or profitable through, the 2022-23 idio-drift-positive regime).
  Otherwise NO (it is alt-short beta or regime luck).

HIGHER EVIDENTIAL BAR (stated honestly): this is question #13 to this dataset
and it was BORN from inspecting recent-year conditionals. Recent-year-tuned
structure invites confirmation; the bar for PROMISING is therefore HIGHER, not
lower. Two-sided specificity + a-priori regime gate + same-year random control
are the guards against fooling ourselves.

Usage: python idio_short_study.py
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
FUND_DIR = REPO_ROOT / "research" / "data" / "perp" / "funding"
ART_DIR = REPO_ROOT / "research" / "data" / "liqrev"
DS_PATH = ART_DIR / "ml_dataset.parquet"

IDIO_CUT = -0.02          # btc_ret_6h > IDIO_CUT  => idiosyncratic
COST_RT = 0.0025          # 25 bps round trip (primary)
STOP_MULT = 1.20          # disaster stop: +20% above entry
SLOTS = 15
SMA_HOURS = 200 * 24      # 200d as-of on the hourly grid
LIQ_MIN = 1e6             # $1M/day 30d-median liquidity filter
N_RAND_PER_YEAR = 300     # ~1500 across 2022..2026
SEED = 13                 # question #13
HOUR_NS = 3_600_000_000_000
YEARS = [2022, 2023, 2024, 2025, 2026]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_klines(pair: str) -> dict | None:
    p = KL_DIR / f"{pair}.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "open", "high", "low",
                                     "close", "quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.set_index("ts").sort_index()
    k = k[~k.index.duplicated(keep="last")]
    # $1M liquidity filter (identical to liqrev: 30d median of daily quote vol)
    dvol30 = k["quote_volume"].resample("1D").sum().rolling(30).median()
    liq_ok = (dvol30.reindex(k.index, method="ffill") > LIQ_MIN).fillna(False)
    return {"idx": k.index.asi8.astype(np.int64),
            "o": k["open"].to_numpy(np.float64),
            "h": k["high"].to_numpy(np.float64),
            "l": k["low"].to_numpy(np.float64),
            "c": k["close"].to_numpy(np.float64),
            "liq": liq_ok.to_numpy(bool),
            "year": k.index.year.to_numpy()}


def load_funding(pair: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = FUND_DIR / f"{pair}.parquet"
    if not p.exists():
        return None
    f = pd.read_parquet(p, columns=["fundingTime", "fundingRate"])
    ts = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("1h")
    s = pd.Series(f["fundingRate"].to_numpy(np.float64), index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.index.asi8.astype(np.int64), s.to_numpy(np.float64)


def funding_sum(fc, entry_ns: int, exit_ns: int) -> float:
    """+sum of funding rates settled in (entry_ns, exit_ns] (short receives +)."""
    if fc is None:
        return 0.0
    settle, rate = fc
    lo = int(np.searchsorted(settle, entry_ns, side="right"))
    hi = int(np.searchsorted(settle, exit_ns, side="right"))
    if hi <= lo:
        return 0.0
    return float(rate[lo:hi].sum())


# --------------------------------------------------------------------------- #
# Short / long simulators
# --------------------------------------------------------------------------- #
def _short_one(kc, fc, i: int, hold: int, cost: float,
               failed_bounce: bool) -> dict | None:
    """Return trade dict, or None if not-qualified (S2) / no data. Uses frozen
    short convention entry/exit - 1 - cost + funding, disaster stop +20%."""
    o, h, c, idx, n = kc["o"], kc["h"], kc["c"], kc["idx"], len(kc["c"])
    if failed_bounce:
        if i + 24 >= n:
            return None
        if not (c[i + 24] < c[i]):
            return {"qualified": False}       # confirmation not met -> no trade
        entry_bar = i + 25
    else:
        entry_bar = i + 1
    if entry_bar >= n or entry_bar + hold - 1 >= n:
        return None
    entry = float(o[entry_bar])
    stop = entry * STOP_MULT
    exit_px, exit_bar, stopped = None, entry_bar + hold - 1, False
    for j in range(entry_bar, entry_bar + hold):
        if o[j] >= stop or h[j] >= stop:          # breached this bar
            exit_px, exit_bar, stopped = float(max(o[j], stop)), j, True
            break
    if exit_px is None:
        exit_px = float(c[entry_bar + hold - 1])
    entry_ns = int(idx[entry_bar])
    exit_ns = int(idx[exit_bar]) + HOUR_NS
    fp = funding_sum(fc, entry_ns, exit_ns)
    ret = entry / exit_px - 1.0 - cost + fp
    return {"qualified": True, "filled": True, "ret": float(ret),
            "stopped": bool(stopped), "funding": float(fp),
            "exit_ts": pd.Timestamp(int(idx[exit_bar]), tz="UTC")}


def sim_short(events, kcache, fcache, hold: int, cost: float,
              failed_bounce: bool = False) -> tuple[pd.DataFrame, dict]:
    rows, n_data, n_qual = [], 0, 0
    for sym, i, ts in events:
        kc = kcache.get(sym)
        if kc is None:
            continue
        t = _short_one(kc, fcache.get(sym), int(i), hold, cost, failed_bounce)
        if t is None:
            continue
        n_data += 1
        if not t["qualified"]:
            continue
        n_qual += 1
        rows.append({"symbol": sym, "ts": ts, "filled": True, "ret": t["ret"],
                     "stopped": t["stopped"], "funding": t["funding"],
                     "exit_ts": t["exit_ts"]})
    df = pd.DataFrame(rows)
    meta = {"n_with_data": n_data, "n_qualified": n_qual,
            "qualify_frac": round(n_qual / n_data, 3) if n_data else None}
    return df, meta


def sim_long(events, kcache, hold: int, cost: float) -> pd.DataFrame:
    """C3 reference: simple long, entry open[i+1], exit close[i+hold], no stop."""
    rows = []
    for sym, i, ts in events:
        kc = kcache.get(sym)
        if kc is None:
            continue
        o, c, n = kc["o"], kc["c"], len(kc["c"])
        i = int(i)
        if i + hold >= n:
            continue
        entry = float(o[i + 1])
        exit_px = float(c[i + hold])          # close of bar i+hold (t+ (hold)h from entry)
        rows.append({"symbol": sym, "ts": ts, "filled": True,
                     "ret": exit_px / entry - 1.0 - cost, "stopped": False,
                     "exit_ts": pd.Timestamp(int(kc["idx"][i + hold]), tz="UTC")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Portfolio (liqrev_v2 pattern) + reporting
# --------------------------------------------------------------------------- #
def portfolio(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {}
    t = tr[tr["filled"]].sort_values("ts")
    eq, busy, curve = 1.0, [], []
    n_taken = 0
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


def _stats(f: pd.DataFrame) -> dict:
    if f.empty:
        return {"n": 0, "net_mean": None, "win": None, "stop_rate": None,
                "p5": None}
    return {"n": int(len(f)),
            "net_mean": round(float(f["ret"].mean()), 4),
            "net_median": round(float(f["ret"].median()), 4),
            "win": round(float((f["ret"] > 0).mean()), 3),
            "stop_rate": round(float(f["stopped"].mean()), 3),
            "p5": round(float(f["ret"].quantile(0.05)), 4),
            "funding_mean": round(float(f["funding"].mean()), 5)
            if "funding" in f else None}


def report(tr: pd.DataFrame, tag: str, meta: dict | None = None) -> dict:
    f = tr[tr["filled"]] if not tr.empty else tr
    by_year = {}
    if not f.empty:
        for y, g in f.groupby(f["ts"].dt.year):
            by_year[str(int(y))] = _stats(g)
    dev = f[f["ts"] < "2025-01-01"] if not f.empty else f
    rec = f[f["ts"] >= "2025-01-01"] if not f.empty else f
    out = {"tag": tag, **_stats(f),
           "dev_pre2025": _stats(dev), "y2025_26": _stats(rec),
           "by_year": by_year, "portfolio_15slots": portfolio(tr)}
    if meta:
        out["meta"] = meta
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(SEED)
    ds = pd.read_parquet(DS_PATH, columns=["symbol", "ts", "btc_ret_6h",
                                           "net_ret", "year"])
    ds = ds.sort_values("ts").reset_index(drop=True)
    idio = ds[ds["btc_ret_6h"] > IDIO_CUT].copy()
    print(f"events: total={len(ds)}  idio={len(idio)}  "
          f"mw={len(ds) - len(idio)}", flush=True)

    # klines + funding caches (all universe pairs, for events + random control)
    pairs = [c.pair for c in load_universe()]
    kcache, fcache = {}, {}
    for n, pair in enumerate(pairs, 1):
        kc = load_klines(pair)
        if kc is not None:
            kcache[pair] = kc
            fcache[pair] = load_funding(pair)
        if n % 40 == 0:
            print(f"  loaded {n}/{len(pairs)} klines", flush=True)
    print(f"klines loaded: {len(kcache)} symbols", flush=True)

    # BTC 200d SMA (as-of, trailing 4800h rolling mean of hourly close)
    btc = kcache["BTCUSDT"]
    btc_close = pd.Series(btc["c"], index=btc["idx"])
    btc_sma = btc_close.rolling(SMA_HOURS, min_periods=SMA_HOURS).mean()
    btc_idx = btc["idx"]

    def below_sma(ts: pd.Timestamp) -> bool | None:
        pos = int(np.searchsorted(btc_idx, ts.value, side="right")) - 1
        if pos < 0:
            return None
        sma = btc_sma.iloc[pos]
        if not np.isfinite(sma):
            return None
        return bool(btc["c"][pos] < sma)

    # map events -> (symbol, bar i, ts); require exact ts hit in that kline grid
    def to_events(frame: pd.DataFrame):
        ev = []
        for _, r in frame.iterrows():
            kc = kcache.get(r["symbol"])
            if kc is None:
                continue
            tv = pd.Timestamp(r["ts"]).value
            pos = int(np.searchsorted(kc["idx"], tv))
            if pos >= len(kc["idx"]) or int(kc["idx"][pos]) != tv:
                continue
            ev.append((r["symbol"], pos, pd.Timestamp(r["ts"], tz="UTC")
                       if pd.Timestamp(r["ts"]).tzinfo is None
                       else pd.Timestamp(r["ts"])))
        return ev

    idio_ev = to_events(idio)
    all_ev = to_events(ds)
    print(f"mapped events: idio={len(idio_ev)}  all={len(all_ev)}", flush=True)

    reports = {}

    # ---- S1 immediate short: hold 24 / 48, 25bps ----
    s1_24, _ = sim_short(idio_ev, kcache, fcache, 24, COST_RT)
    reports["S1_immediate_24_25bps"] = report(s1_24, "S1_immediate_24_25bps")
    s1_48, _ = sim_short(idio_ev, kcache, fcache, 48, COST_RT)
    reports["S1_immediate_48_25bps"] = report(s1_48, "S1_immediate_48_25bps")
    # 10bps sensitivity on the best cell (declared best = S1-24)
    s1_24_10, _ = sim_short(idio_ev, kcache, fcache, 24, 0.0010)
    reports["S1_immediate_24_10bps_sens"] = report(s1_24_10,
                                                    "S1_immediate_24_10bps_sens")

    # ---- S2 failed-bounce short: hold 24 / 48, 25bps ----
    s2_24, m2_24 = sim_short(idio_ev, kcache, fcache, 24, COST_RT,
                             failed_bounce=True)
    reports["S2_failed_bounce_24_25bps"] = report(s2_24,
                                                   "S2_failed_bounce_24_25bps",
                                                   m2_24)
    s2_48, m2_48 = sim_short(idio_ev, kcache, fcache, 48, COST_RT,
                             failed_bounce=True)
    reports["S2_failed_bounce_48_25bps"] = report(s2_48,
                                                   "S2_failed_bounce_48_25bps",
                                                   m2_48)

    # ---- Regime-gated S1-24h (a-priori BTC<200d SMA gate) ----
    gate_flags = {}
    gated_ev, ungated_note = [], []
    for sym, i, ts in idio_ev:
        b = below_sma(ts)
        gate_flags[(sym, ts)] = b
        if b:
            gated_ev.append((sym, i, ts))
    g_24, _ = sim_short(gated_ev, kcache, fcache, 24, COST_RT)
    gated_rep = report(g_24, "S1_gated_below200dSMA_24_25bps")
    # gate coverage / behaviour by year (how many idio events pass the gate)
    gate_by_year = {}
    for y in YEARS:
        yes = sum(1 for (s, t), b in gate_flags.items() if t.year == y and b)
        no = sum(1 for (s, t), b in gate_flags.items()
                 if t.year == y and b is False)
        na = sum(1 for (s, t), b in gate_flags.items()
                 if t.year == y and b is None)
        gate_by_year[str(y)] = {"pass": yes, "block": no, "na_no_sma": na}
    gated_rep["gate_coverage_by_year"] = gate_by_year
    reports["S1_gated_below200dSMA_24_25bps"] = gated_rep
    # ungated reference already = S1_immediate_24

    # ---- C1: short ALL cascade events (idio + market-wide) ----
    c1_24, _ = sim_short(all_ev, kcache, fcache, 24, COST_RT)
    reports["C1_all_events_short_24"] = report(c1_24, "C1_all_events_short_24")

    # ---- C2: RANDOM-SHORT baseline, per year (~300/yr) ----
    # build per-symbol valid entry positions passing $1M liquidity, 24h forward
    pool_sym, pool_pos, pool_year = [], [], []
    for sym, kc in kcache.items():
        n = len(kc["c"])
        if n < 26:
            continue
        valid = np.where(kc["liq"][: n - 25])[0]      # need entry_bar+24 in range
        # entry_bar = pos+1, need pos+25 <= n-1 -> pos <= n-26
        valid = valid[valid <= n - 26]
        if valid.size == 0:
            continue
        pool_sym.append(np.full(valid.size, sym, dtype=object))
        pool_pos.append(valid)
        pool_year.append(kc["year"][valid])
    pool_sym = np.concatenate(pool_sym)
    pool_pos = np.concatenate(pool_pos)
    pool_year = np.concatenate(pool_year)
    rand_ev = []
    rand_year_counts = {}
    for y in YEARS:
        idxs = np.where(pool_year == y)[0]
        if idxs.size == 0:
            rand_year_counts[str(y)] = 0
            continue
        take = min(N_RAND_PER_YEAR, idxs.size)
        pick = rng.choice(idxs, size=take, replace=False)
        rand_year_counts[str(y)] = int(take)
        for p in pick:
            sym = pool_sym[p]
            pos = int(pool_pos[p])
            ts = pd.Timestamp(int(kcache[sym]["idx"][pos]), tz="UTC")
            rand_ev.append((sym, pos, ts))
    c2_24, _ = sim_short(rand_ev, kcache, fcache, 24, COST_RT)
    c2_rep = report(c2_24, "C2_random_short_24")
    c2_rep["sampled_per_year"] = rand_year_counts
    reports["C2_random_short_24"] = c2_rep

    # ---- C3: long side of idio events (reference, ~-1%) ----
    c3 = sim_long(idio_ev, kcache, 24, COST_RT)
    reports["C3_idio_long_24_ref"] = report(c3, "C3_idio_long_24_ref")
    # dataset's own long net_ret for idio (cross-check the established structure)
    ds_idio_by_year = {str(int(y)): round(float(g["net_ret"].mean()), 4)
                       for y, g in idio.groupby(idio["ts"].dt.year)}
    reports["C3_idio_long_24_ref"]["dataset_net_ret_by_year"] = ds_idio_by_year

    # ---- Q1 SPECIFICITY table: S1-24 idio-short vs same-year random-short ----
    def by_year_net(rep):
        return {y: rep["by_year"].get(y, {}).get("net_mean")
                for y in [str(x) for x in YEARS]}
    idio_ny = by_year_net(reports["S1_immediate_24_25bps"])
    rand_ny = by_year_net(reports["C2_random_short_24"])
    spec_table = {}
    for y in [str(x) for x in YEARS]:
        iv, rv = idio_ny.get(y), rand_ny.get(y)
        edge = round(iv - rv, 4) if (iv is not None and rv is not None) else None
        spec_table[y] = {"idio_short_net": iv, "random_short_net": rv,
                         "edge_bps": round(edge * 1e4, 1) if edge is not None
                         else None}

    # pooled 2024-26 edge (Q1 pass = edge >= +50 bps)
    f_idio = s1_24[s1_24["ts"] >= "2024-01-01"]
    f_rand = c2_24[c2_24["ts"] >= "2024-01-01"]
    edge_2426 = None
    if len(f_idio) and len(f_rand):
        edge_2426 = round(float(f_idio["ret"].mean() - f_rand["ret"].mean()), 4)
    q1_pass = bool(edge_2426 is not None and edge_2426 >= 0.005)

    # ---- Q2 REGIME ----
    ungated_by_year = {y: reports["S1_immediate_24_25bps"]["by_year"]
                       .get(y, {}).get("net_mean") for y in
                       [str(x) for x in YEARS]}
    gated_by_year = {y: gated_rep["by_year"].get(y, {}).get("net_mean")
                     for y in [str(x) for x in YEARS]}
    pos_all_years = all(v is not None and v > 0 for v in ungated_by_year.values())
    # gated positive in 2024-26?
    fg = g_24[g_24["ts"] >= "2024-01-01"]
    gated_pos_2426 = bool(len(fg) and fg["ret"].mean() > 0)
    # gate behaviour 2022-23: did it keep us out (few passes) or profitable?
    g_2223 = g_24[g_24["ts"] < "2024-01-01"]
    gate_2223_net = round(float(g_2223["ret"].mean()), 4) if len(g_2223) else None
    gate_2223_n = int(len(g_2223))
    # (b): positive all years OR (gated positive 2024-26 AND gate handled 2022-23:
    #      either it blocked most trades OR gated 2022-23 return >= 0)
    gate_handled_2223 = (gate_2223_n == 0) or (gate_2223_net is not None
                                               and gate_2223_net >= 0)
    q2_pass = bool(pos_all_years or (gated_pos_2426 and gate_handled_2223))

    verdict = "PROMISING" if (q1_pass and q2_pass) else "NO"
    verdict_block = {
        "verdict": verdict,
        "Q1_specificity": {
            "pooled_2024_26_edge": edge_2426,
            "pooled_2024_26_edge_bps": round(edge_2426 * 1e4, 1)
            if edge_2426 is not None else None,
            "threshold_bps": 50, "pass": q1_pass,
            "by_year": spec_table},
        "Q2_regime": {
            "ungated_net_by_year": ungated_by_year,
            "positive_all_years": pos_all_years,
            "gated_net_by_year": gated_by_year,
            "gated_positive_2024_26": gated_pos_2426,
            "gate_2022_23_net": gate_2223_net, "gate_2022_23_n": gate_2223_n,
            "gate_handled_2022_23": gate_handled_2223, "pass": q2_pass},
        "rule": "PROMISING iff (a) edge>=+50bps in 2024-26 AND (b) positive all "
                "years OR gated-positive-2024-26 with correct 2022-23 gate "
                "behaviour",
        "evidential_note": "Question #13, born from recent-year conditionals; "
                           "bar is HIGHER not lower. See docstring."}

    out = {"run_utc": datetime.now(timezone.utc).isoformat(),
           "spec": {"idio_cut": IDIO_CUT, "cost_rt": COST_RT,
                    "stop_mult": STOP_MULT, "slots": SLOTS,
                    "sma_hours": SMA_HOURS, "n_rand_per_year": N_RAND_PER_YEAR,
                    "seed": SEED},
           "n_events": {"total": len(ds), "idio": len(idio),
                        "idio_mapped": len(idio_ev), "all_mapped": len(all_ev)},
           "Q1_specificity_table": spec_table,
           "verdict": verdict_block,
           "reports": reports}

    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "results_idio_short.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    # ---- console summary ----
    print("\n=== Q1 SPECIFICITY: S1-24 idio-short vs same-year random-short ===")
    print(f"{'year':>5} {'idio_net':>10} {'rand_net':>10} {'edge_bps':>9} "
          f"{'idio_n':>7} {'rand_n':>7}")
    for y in [str(x) for x in YEARS]:
        st = spec_table[y]
        iy = reports["S1_immediate_24_25bps"]["by_year"].get(y, {})
        ry = reports["C2_random_short_24"]["by_year"].get(y, {})
        print(f"{y:>5} {str(st['idio_short_net']):>10} "
              f"{str(st['random_short_net']):>10} {str(st['edge_bps']):>9} "
              f"{str(iy.get('n')):>7} {str(ry.get('n')):>7}")
    print(f"pooled 2024-26 edge = {edge_2426}  (Q1 pass={q1_pass})")

    print("\n=== GRID SUMMARY (net_mean / win / stop / p5 / n) ===")
    for tag in ["S1_immediate_24_25bps", "S1_immediate_48_25bps",
                "S1_immediate_24_10bps_sens", "S2_failed_bounce_24_25bps",
                "S2_failed_bounce_48_25bps", "S1_gated_below200dSMA_24_25bps",
                "C1_all_events_short_24", "C2_random_short_24",
                "C3_idio_long_24_ref"]:
        r = reports[tag]
        pf = r.get("portfolio_15slots", {})
        print(f"  {tag:>34}: net={str(r['net_mean']):>8} win={str(r['win']):>5} "
              f"stop={str(r['stop_rate']):>5} p5={str(r['p5']):>8} "
              f"n={str(r['n']):>4}  cagr={str(pf.get('cagr')):>7} "
              f"maxDD={str(pf.get('maxDD')):>7}")

    print("\n=== VERDICT ===")
    print(json.dumps(verdict_block, indent=1, default=str))
    print(f"\nartifacts -> {ART_DIR / 'results_idio_short.json'}")


if __name__ == "__main__":
    main()
