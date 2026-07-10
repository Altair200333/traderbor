"""PRE-REGISTERED FEATURE AUDIT #14 -- order-book DEPTH at the liqrev trigger.

This docstring is the pre-registration: it is fixed BEFORE any results are looked
at. Question: does resting order-book depth at the liquidation-cascade trigger add
conditioning signal for bounce quality BEYOND the BTC-context overlay (btc_ret_6h
mw/idio split) that liqrev_v2 already trades? No strategy is built either way; a
depth feature either clears the pre-registered bar or is reported as descriptive
color only.

------------------------------------------------------------------- DATA / SCHEMA
Raw depth: research/data/binance_um/bookdepth_events/{PAIR}.parquet, produced by
download_bookdepth_events.py from Binance UM daily bookDepth dumps. LONG rows:
  event_ts(ms trigger-bar open), ts(ms snapshot), percentage, depth, notional.
Binance bookDepth `percentage` is a BAND (% distance from mid), NOT a price level:
negative=BID side below price, positive=ASK side above price; depth/notional at |k|
are CUMULATIVE resting size within k% of mid. We use `notional` (quote/USD).
  bid1=notional[-1], bid2=notional[-2], bid5=notional[-5],
  ask2=notional[+2], ask5=notional[+5].
Snapshot cadence ~25-30 s. Labels: research/data/liqrev/ml_dataset.parquet
(filled events only; net_ret = frozen maker-config net return; btc_ret_6h).
Denominator source for f1/f2: spot 1h klines research/data/v3/klines/1h/{PAIR}.parquet.

--------------------------------------------------------- DECLARED FEATURES (f1-f6)
Per event, from the LAST snapshot at or before trigger-bar CLOSE (= event_ts+1h;
snapshot taken as the latest ts in [event_ts, event_ts+1h]):
  f1 bid_near_norm = bid2 / D           D = trailing-30d median DAILY quote_volume
  f2 ask_near_norm = ask2 / D               (spot 1h klines, as-of, COMPLETED days
                                             strictly before the trigger date; median
                                             of the last <=30 such days, need >=5)
  f3 imb_2pct  = (bid2 - ask2) / (bid2 + ask2)
  f4 imb_5pct  = (bid5 - ask5) / (bid5 + ask5)
  f5 wall_pull = bid2(trigger) / bid2(snapshot nearest event_ts-6h)   [NaN if the
                 t-6h snapshot is missing (tol +-15min) or its bid2<=0]
  f6 depth_slope = bid1 / bid5          [NaN if bid5<=0]   near-concentration
SCHEMA ADAPTATION (disclosed): the spec said "notional within 2% below price". The
real feed has no per-order levels, only cumulative band notionals; bid2 = the -2%
band cumulative notional IS "resting notional within 2% below price". f6's near-
concentration uses within-1% / within-5% cumulative notionals (both bid side).

----------------------------------------------------------------- DECLARED WINDOWS
DEV    = trigger ts in 2023-01 .. 2024-12   (primary evidence)
RECENT = trigger ts in 2025-01 .. end       SIGN-CHECK ONLY. NOTE: RECENT was
  already opened at the event-conditional level during the overlay work, so it is
  NOT a pristine holdout -- the bar it must clear is correspondingly higher (sign
  agreement is necessary, not sufficient).

---------------------------------------------------------------- DECLARED ANALYSIS
Filled events only. Per feature:
  (a) DEV quintile means of net_ret (5 equal-count bins).
  (b) DEV Spearman IC(feature, net_ret) with a DAY-CLUSTERED bootstrap 95% CI:
      resample CALENDAR DAYS with replacement (~1000 iters), recompute IC, take the
      2.5/97.5 percentiles. Clustering by day respects the fact that same-day events
      across symbols are correlated (shared BTC move).
  (c) RECENT Spearman IC sign (same sign as DEV or not).
INCREMENTAL TEST (the key one): split DEV by BTC context -- mw = btc_ret_6h < -0.02,
  idio = btc_ret_6h >= -0.02 -- and WITHIN each bucket separately compute tercile
  means of net_ret per feature. Does depth separate bounce quality beyond the
  mw/idio overlay we already trade? Spread = top-tercile mean - bottom-tercile mean.
PHENOMENOLOGY (reported regardless): median imb_2pct, median imb_5pct, median
  wall_pull across events; do books empty out during the fall (trigger bid2 vs the
  t-6h and ~t-12h bid2)? A true "same hour one day earlier" (t-24h) snapshot is NOT
  retained under the pre-registered [t-12h, t+1h] keep-window; the t-6h and ~t-12h
  in-window baselines are used as the closest available proxies (disclosed).

----------------------------------------------------------- DECISION RULE (per feat)
HELPS  iff  DEV |IC| >= 0.10  AND  the day-clustered 95% CI excludes 0
       AND  RECENT IC has the SAME sign as DEV
       AND  within-bucket tercile spread |top-bottom| >= 50 bps in >=1 DEV bucket.
Anything less = NO (descriptive color only). No strategy is built either way.

Output: research/data/liqrev/results_bookdepth.json
Usage:  python bookdepth_audit.py [--iters 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402

DATA = REPO_ROOT / "research" / "data"
DEPTH_DIR = DATA / "binance_um" / "bookdepth_events"
ML_DATASET = DATA / "liqrev" / "ml_dataset.parquet"
KLINES_DIR = DATA / "v3" / "klines" / "1h"
OUT_JSON = DATA / "liqrev" / "results_bookdepth.json"

HOUR = 3_600_000
TOL_6H = 15 * 60 * 1000       # +-15 min around t-6h
TOL_12H = 30 * 60 * 1000      # +-30 min around t-12h
NEEDED_BANDS = (-5, -2, -1, 2, 5)
FEATURES = ["f1_bid_near_norm", "f2_ask_near_norm", "f3_imb_2pct",
            "f4_imb_5pct", "f5_wall_pull", "f6_depth_slope"]
MW_CUT = -0.02


# ------------------------------------------------------------- klines denominator
_daily_cache: dict[str, pd.Series | None] = {}


def daily_qv(pair: str) -> pd.Series | None:
    if pair not in _daily_cache:
        p = KLINES_DIR / f"{pair}.parquet"
        if not p.exists():
            _daily_cache[pair] = None
        else:
            k = pd.read_parquet(p, columns=["open_time", "quote_volume"])
            day = (pd.to_datetime(k["open_time"], unit="ms", utc=True)
                   .dt.floor("D"))
            _daily_cache[pair] = k.groupby(day)["quote_volume"].sum().sort_index()
    return _daily_cache[pair]


def denom_asof(pair: str, ets: int) -> float:
    d = daily_qv(pair)
    if d is None:
        return np.nan
    trig_day = pd.Timestamp(ets, unit="ms", tz="UTC").floor("D")
    hist = d[d.index < trig_day]
    if len(hist) < 5:
        return np.nan
    return float(hist.iloc[-30:].median())


# --------------------------------------------------------------- snapshot picking
def _pick_trigger(idx: np.ndarray, ets: int) -> int | None:
    cand = idx[(idx >= ets - HOUR) & (idx <= ets + HOUR)]
    if len(cand) == 0:
        return None
    inbar = cand[cand >= ets]
    use = inbar if len(inbar) else cand
    return int(use.max())


def _pick_nearest(idx: np.ndarray, target: int, tol: int) -> int | None:
    if len(idx) == 0:
        return None
    diff = np.abs(idx - target)
    i = int(diff.argmin())
    return int(idx[i]) if diff[i] <= tol else None


def event_features(piv: pd.DataFrame, ets: int) -> dict:
    """piv: index=ts, columns=band notional. Return f1..f6 + phenomenology inputs."""
    idx = piv.index.to_numpy()
    out = {f: np.nan for f in FEATURES}
    out.update(bid2_trig=np.nan, bid2_t6=np.nan, bid2_t12=np.nan)
    tt = _pick_trigger(idx, ets)
    if tt is None:
        return out
    row = piv.loc[tt]
    bid1, bid2, bid5 = row.get(-1, np.nan), row.get(-2, np.nan), row.get(-5, np.nan)
    ask2, ask5 = row.get(2, np.nan), row.get(5, np.nan)
    out["bid2_trig"] = bid2
    if bid2 + ask2 > 0:
        out["f3_imb_2pct"] = (bid2 - ask2) / (bid2 + ask2)
    if bid5 + ask5 > 0:
        out["f4_imb_5pct"] = (bid5 - ask5) / (bid5 + ask5)
    if bid5 > 0:
        out["f6_depth_slope"] = bid1 / bid5
    out["_bid2"] = bid2
    out["_ask2"] = ask2
    # t-6h wall pull
    t6 = _pick_nearest(idx, ets - 6 * HOUR, TOL_6H)
    if t6 is not None:
        b26 = piv.loc[t6].get(-2, np.nan)
        out["bid2_t6"] = b26
        if b26 and b26 > 0:
            out["f5_wall_pull"] = bid2 / b26
    # ~t-12h earlier baseline (phenomenology only)
    t12 = _pick_nearest(idx, ets - 12 * HOUR, TOL_12H)
    if t12 is not None:
        out["bid2_t12"] = piv.loc[t12].get(-2, np.nan)
    return out


def build_feature_table() -> pd.DataFrame:
    ml = pd.read_parquet(ML_DATASET)
    ml = ml[(ml["filled"] == True) &  # noqa: E712
            (ml["ts"] >= pd.Timestamp("2023-01-02", tz="UTC"))].copy()
    # epoch ms, resolution-agnostic (matches downloader's Timestamp.value//1e6)
    ml["event_ts"] = ml["ts"].dt.tz_localize(None).to_numpy("datetime64[ms]").astype("int64")
    rows = []
    for sym, g in ml.groupby("symbol"):
        p = DEPTH_DIR / f"{sym}.parquet"
        depth = pd.read_parquet(p) if p.exists() else None
        by_ev = {}
        if depth is not None:
            depth = depth[depth["percentage"].isin(NEEDED_BANDS)].copy()
            depth["percentage"] = depth["percentage"].astype(int)  # clean label lookup
            for ev, gg in depth.groupby("event_ts"):
                piv = gg.pivot_table(index="ts", columns="percentage",
                                     values="notional", aggfunc="last").sort_index()
                by_ev[int(ev)] = piv
        for _, r in g.iterrows():
            ets = int(r["event_ts"])
            rec = {"symbol": sym, "event_ts": ets, "ts": r["ts"],
                   "net_ret": r["net_ret"], "btc_ret_6h": r["btc_ret_6h"],
                   "year": int(r["year"])}
            piv = by_ev.get(ets)
            fe = event_features(piv, ets) if piv is not None else {}
            rec.update({f: fe.get(f, np.nan) for f in FEATURES})
            rec["bid2_trig"] = fe.get("bid2_trig", np.nan)
            rec["bid2_t6"] = fe.get("bid2_t6", np.nan)
            rec["bid2_t12"] = fe.get("bid2_t12", np.nan)
            D = denom_asof(sym, ets)
            b2, a2 = fe.get("_bid2", np.nan), fe.get("_ask2", np.nan)
            rec["f1_bid_near_norm"] = (b2 / D) if D and D > 0 else np.nan
            rec["f2_ask_near_norm"] = (a2 / D) if D and D > 0 else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["day"] = df["ts"].dt.floor("D")
    df["win_2yr"] = np.where(df["year"] <= 2024, "DEV", "RECENT")
    df["bucket"] = np.where(df["btc_ret_6h"] < MW_CUT, "mw", "idio")
    return df


# ------------------------------------------------------------------- statistics
def day_bootstrap_ic(sub: pd.DataFrame, feat: str, iters: int, rng) -> tuple:
    s = sub[[feat, "net_ret", "day"]].dropna(subset=[feat, "net_ret"])
    if len(s) < 20 or s[feat].nunique() < 5:
        return (np.nan, np.nan, np.nan)
    ic = float(spearmanr(s[feat], s["net_ret"])[0])
    groups = [g[[feat, "net_ret"]].to_numpy() for _, g in s.groupby("day")]
    ndays = len(groups)
    boots = np.empty(iters)
    for b in range(iters):
        pick = rng.integers(0, ndays, ndays)
        arr = np.concatenate([groups[i] for i in pick])
        if len(np.unique(arr[:, 0])) < 5:
            boots[b] = np.nan
            continue
        boots[b] = spearmanr(arr[:, 0], arr[:, 1])[0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return (ic, float(lo), float(hi))


def quintile_means(sub: pd.DataFrame, feat: str) -> dict:
    s = sub[[feat, "net_ret"]].dropna()
    if len(s) < 25:
        return {"means": [], "counts": [], "n": len(s)}
    try:
        q = pd.qcut(s[feat], 5, labels=False, duplicates="drop")
    except ValueError:
        return {"means": [], "counts": [], "n": len(s)}
    g = s.groupby(q)["net_ret"]
    return {"means": [round(x, 5) for x in g.mean().tolist()],
            "counts": g.size().tolist(), "n": int(len(s)),
            "nbins": int(q.nunique())}


def tercile_spread(sub: pd.DataFrame, feat: str) -> dict:
    s = sub[[feat, "net_ret"]].dropna()
    if len(s) < 15:
        return {"means": [], "spread_bps": None, "n": len(s)}
    try:
        q = pd.qcut(s[feat], 3, labels=False, duplicates="drop")
    except ValueError:
        return {"means": [], "spread_bps": None, "n": len(s)}
    m = s.groupby(q)["net_ret"].mean()
    if m.nunique() < 2 or len(m) < 2:
        return {"means": [round(x, 5) for x in m.tolist()], "spread_bps": None,
                "n": int(len(s))}
    spread = float(m.iloc[-1] - m.iloc[0]) * 1e4
    return {"means": [round(x, 5) for x in m.tolist()],
            "spread_bps": round(spread, 1), "n": int(len(s)),
            "nbins": int(q.nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = build_feature_table()
    dev = df[df["win_2yr"] == "DEV"]
    rec = df[df["win_2yr"] == "RECENT"]

    # coverage
    cov = {
        "events_filled_total": int(len(df)),
        "dev": int(len(dev)), "recent": int(len(rec)),
        "with_trigger_snap": int(df["f3_imb_2pct"].notna().sum()),
        "feat_nonnull": {f: int(df[f].notna().sum()) for f in FEATURES},
        "dev_feat_nonnull": {f: int(dev[f].notna().sum()) for f in FEATURES},
        "symbols_with_depth": int(sorted(DEPTH_DIR.glob("*.parquet")).__len__()),
        "depth_gb": round(sum(p.stat().st_size for p in DEPTH_DIR.glob("*.parquet"))
                          / 1e9, 4),
    }

    ic_table, incremental, decision = {}, {}, {}
    for f in FEATURES:
        ic, lo, hi = day_bootstrap_ic(dev, f, args.iters, rng)
        rec_ic = np.nan
        rs = rec[[f, "net_ret"]].dropna()
        if len(rs) >= 20 and rs[f].nunique() >= 5:
            rec_ic = float(spearmanr(rs[f], rs["net_ret"])[0])
        ci_excl0 = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
        sign_match = bool(np.isfinite(ic) and np.isfinite(rec_ic)
                          and np.sign(ic) == np.sign(rec_ic))
        ic_table[f] = {
            "ic_dev": None if not np.isfinite(ic) else round(ic, 4),
            "ci_lo": None if not np.isfinite(lo) else round(lo, 4),
            "ci_hi": None if not np.isfinite(hi) else round(hi, 4),
            "ci_excludes_0": ci_excl0,
            "ic_recent": None if not np.isfinite(rec_ic) else round(rec_ic, 4),
            "recent_sign_match": sign_match,
            "n_dev": int(dev[[f, "net_ret"]].dropna().shape[0]),
            "n_recent": int(len(rs)),
            "quintiles_dev": quintile_means(dev, f),
        }
        inc = {}
        max_abs_spread = 0.0
        for bkt in ("mw", "idio"):
            t = tercile_spread(dev[dev["bucket"] == bkt], f)
            inc[bkt] = t
            if t["spread_bps"] is not None:
                max_abs_spread = max(max_abs_spread, abs(t["spread_bps"]))
        incremental[f] = inc
        helps = bool(np.isfinite(ic) and abs(ic) >= 0.10 and ci_excl0
                     and sign_match and max_abs_spread >= 50.0)
        decision[f] = {
            "helps": helps,
            "abs_ic_ge_010": bool(np.isfinite(ic) and abs(ic) >= 0.10),
            "ci_excludes_0": ci_excl0,
            "recent_sign_match": sign_match,
            "max_bucket_tercile_spread_bps": round(max_abs_spread, 1),
            "spread_ge_50bps": bool(max_abs_spread >= 50.0),
        }

    # phenomenology
    def _med(x):
        x = x.dropna()
        return None if x.empty else round(float(x.median()), 4)

    ph = {
        "median_imb_2pct": _med(df["f3_imb_2pct"]),
        "median_imb_5pct": _med(df["f4_imb_5pct"]),
        "median_wall_pull": _med(df["f5_wall_pull"]),
        "median_depth_slope": _med(df["f6_depth_slope"]),
        "frac_walls_pulled_lt1": (None if df["f5_wall_pull"].dropna().empty else
                                  round(float((df["f5_wall_pull"].dropna() < 1).mean()), 3)),
        "median_trigger_vs_t12_bid2": _med(df["bid2_trig"] / df["bid2_t12"]),
        "frac_thinner_at_trigger_vs_t6": (
            None if (df["bid2_trig"].notna() & df["bid2_t6"].notna()).sum() == 0 else
            round(float((df.loc[df["bid2_t6"] > 0, "bid2_trig"]
                         < df.loc[df["bid2_t6"] > 0, "bid2_t6"]).mean()), 3)),
        "median_bid2_trig_usd": _med(df["bid2_trig"]),
        "n_wall_pull_defined": int(df["f5_wall_pull"].notna().sum()),
    }

    results = {
        "schema": {
            "source": "Binance UM daily bookDepth",
            "csv_columns": ["timestamp(str UTC naive)", "percentage(band +-1..+-5, %"
                            " from mid; neg=bid/pos=ask)", "depth(base qty, cumulative"
                            " within band)", "notional(USD, cumulative within band)"],
            "snapshot_cadence_s": "~25-30",
            "note": "percentage is a distance-from-mid BAND, not a price level; "
                    "notional is cumulative within |band|%.",
        },
        "windows": {"DEV": "2023-01..2024-12", "RECENT": "2025-01..2026-07 "
                    "(sign-check, already opened, higher bar)"},
        "coverage": cov,
        "ic_table": ic_table,
        "incremental_terciles": incremental,
        "phenomenology": ph,
        "decision": decision,
        "decision_rule": "HELPS iff DEV|IC|>=0.10 AND day-clustered 95% CI excl 0 "
                         "AND RECENT same IC sign AND >=1 DEV bucket tercile spread "
                         ">=50bps.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # compact console summary
    print(f"events(filled,>=2023-01-02): {cov['events_filled_total']} | DEV "
          f"{cov['dev']} RECENT {cov['recent']} | trigger-snap "
          f"{cov['with_trigger_snap']} | depth {cov['depth_gb']}GB "
          f"{cov['symbols_with_depth']} syms")
    print(f"phenomenology: med imb_2pct={ph['median_imb_2pct']} "
          f"med wall_pull={ph['median_wall_pull']} "
          f"walls_pulled<1={ph['frac_walls_pulled_lt1']} "
          f"thinner_vs_t6={ph['frac_thinner_at_trigger_vs_t6']}")
    print(f"{'feature':<18}{'IC_dev':>8}{'CI_lo':>8}{'CI_hi':>8}{'IC_rec':>8}"
          f"{'mw_bps':>8}{'idio_bps':>9}  HELP")
    for f in FEATURES:
        t = ic_table[f]
        d = decision[f]
        mw = incremental[f]["mw"]["spread_bps"]
        idio = incremental[f]["idio"]["spread_bps"]
        print(f"{f:<18}{str(t['ic_dev']):>8}{str(t['ci_lo']):>8}{str(t['ci_hi']):>8}"
              f"{str(t['ic_recent']):>8}{str(mw):>8}{str(idio):>9}  "
              f"{'YES' if d['helps'] else 'no'}")
    print(f"-> results: {OUT_JSON}")


if __name__ == "__main__":
    main()
