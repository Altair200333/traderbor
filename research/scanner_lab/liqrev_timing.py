"""Liquidation-cascade reversion — TRIGGER-TIMING conditioning study (one-shot).

PRE-REGISTERED spec (frozen in this docstring BEFORE the first run; 2026-07-08).
Reuses liqrev_v2 verbatim: event set + kcache are built exactly as
liqrev_v2.main() does, then simulate(ev, kcache, "market", "none", 0.0025,
False) is run ONCE (v1-comparable long, entry next 1h open, NO stop, exit +24h
close, 25bps RT). We only SLICE the per-trade results by the trigger-bar
timestamp; no re-detection, no alternate exit logic, no per-bucket refitting.

QUESTION. Does the UTC clock time of the trigger bar condition the edge
(session, day-of-week, weekend)? If a bucket beats its complement beyond noise
AND the sign replicates out of sample, a session filter is worth adding live;
otherwise we must report NOISE and add nothing.

BUCKETS (pre-registered; no bucket added after seeing results):
  - UTC session of TRIGGER bar:  ASIA 00-08h, EU 08-16h, US 16-24h  [FILTER CANDIDATES]
  - Weekday-vs-weekend: weekend = Sat+Sun (dayofweek in {5,6})       [FILTER CANDIDATE]
  - Day-of-week: 7 rows                                              [DIAGNOSTIC ONLY]
  - Hour-of-day: 24 rows                                             [DIAGNOSTIC ONLY, tiny n/cell (~55)]
Per bucket we report: n, net mean, median, win rate, standard error se=std/sqrt(n)
(ddof=1). All three windows: FULL 4y, DEV (ts < 2025-01-01), HOLDOUT (ts >= 2025-01-01).

DECISION RULE (pre-registered). A timing filter is declared REAL only if BOTH:
  (a) on the FULL window, |mean_bucket - mean_complement| > 2 * se_diff, where
      se_diff = sqrt(se_bucket^2 + se_complement^2)  (bucket vs everything-else); AND
  (b) sign(mean_bucket - mean_complement) is IDENTICAL in DEV and HOLDOUT.
Applied to FILTER CANDIDATES only (3 sessions + weekend-vs-weekday). DOW /
hour-of-day are diagnostic (multiple-comparison / tiny-n), not decision inputs.

POWER MATH (state explicitly in report). ~1300 events, per-trade net std ~8%.
A 3-way session split => ~430/bucket => se ~ 0.08/sqrt(430) ~ 0.4% per bucket.
Bucket-vs-complement se_diff is of the same order, so 2*se_diff ~ 0.8-1.1pp:
net-mean gaps under ~1.1pp are NOISE and cannot support a filter.

CLUSTERING CAVEAT (pre-registered). Cascades cluster in market-wide crashes, so
same-bucket trades are NOT independent (nominal se understates uncertainty). We
report, per bucket: distinct calendar days (effective-n proxy) and the top-3
calendar days by event count. If a bucket's edge is driven by 1-2 crash days,
flag it — the se and the decision are then untrustworthy.

Live shadow (small size) remains the final validator regardless of outcome.

Usage: python liqrev_timing.py   (a few minutes: event detection over ~149 pairs)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liqrev_v2 import ART_DIR, KL_DIR, detect_events, portfolio, simulate  # noqa: E402
from universe import load_universe  # noqa: E402

HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")


def _stat(r: pd.Series) -> dict:
    """Per-bucket summary stats on a net-return series."""
    n = int(len(r))
    if n == 0:
        return {"n": 0, "net_mean": None, "net_median": None, "win": None,
                "se": None, "std": None}
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    return {"n": n,
            "net_mean": round(float(r.mean()), 4),
            "net_median": round(float(r.median()), 4),
            "win": round(float((r > 0).mean()), 3),
            "se": round(std / np.sqrt(n), 4) if n else None,
            "std": round(std, 4)}


def _clustering(df: pd.DataFrame) -> dict:
    """Effective-n proxy: distinct calendar days + top-3 event-heavy days."""
    days = df["ts"].dt.date.astype(str)
    top3 = Counter(days).most_common(3)
    return {"n": int(len(df)),
            "distinct_days": int(days.nunique()),
            "top3_days": [[d, int(c)] for d, c in top3]}


def _bucket_tables(f: pd.DataFrame) -> dict:
    """All pre-registered bucket tables for one window."""
    sess, wknd, dow, hod = {}, {}, {}, {}
    for name, m in (("ASIA", f["sess"] == "ASIA"),
                    ("EU", f["sess"] == "EU"),
                    ("US", f["sess"] == "US")):
        sess[name] = _stat(f.loc[m, "ret"])
    wknd["weekend"] = _stat(f.loc[f["weekend"], "ret"])
    wknd["weekday"] = _stat(f.loc[~f["weekend"], "ret"])
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d in range(7):
        dow[dow_names[d]] = _stat(f.loc[f["dow"] == d, "ret"])
    for h in range(24):
        hod[f"{h:02d}"] = _stat(f.loc[f["hour"] == h, "ret"])
    return {"sessions": sess, "weekend_vs_weekday": wknd,
            "day_of_week": dow, "hour_of_day_DIAGNOSTIC": hod}


def _decision_one(f_full: pd.DataFrame, f_dev: pd.DataFrame,
                  f_hold: pd.DataFrame, mask_col_val) -> dict:
    """Apply pre-registered rule to one bucket vs its complement.

    mask_col_val: callable(df) -> boolean mask selecting the bucket.
    """
    def diff_and_se(f):
        m = mask_col_val(f)
        rb, rc = f.loc[m, "ret"], f.loc[~m, "ret"]
        if len(rb) < 2 or len(rc) < 2:
            return None, None
        mb, mc = float(rb.mean()), float(rc.mean())
        seb = float(rb.std(ddof=1)) / np.sqrt(len(rb))
        sec = float(rc.std(ddof=1)) / np.sqrt(len(rc))
        return mb - mc, float(np.sqrt(seb ** 2 + sec ** 2))

    d_full, se_full = diff_and_se(f_full)
    d_dev, _ = diff_and_se(f_dev)
    d_hold, _ = diff_and_se(f_hold)
    if d_full is None:
        return {"insufficient": True}
    pass_a = abs(d_full) > 2 * se_full
    pass_b = (d_dev is not None and d_hold is not None
              and np.sign(d_dev) == np.sign(d_hold) != 0)
    return {"full_diff": round(d_full, 4),
            "se_diff": round(se_full, 4),
            "threshold_2se": round(2 * se_full, 4),
            "dev_diff": round(d_dev, 4) if d_dev is not None else None,
            "holdout_diff": round(d_hold, 4) if d_hold is not None else None,
            "pass_a_gt_2se": bool(pass_a),
            "pass_b_sign_replicates": bool(pass_b),
            "REAL_FILTER": bool(pass_a and pass_b)}


def _annotate(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f["hour"] = f["ts"].dt.hour
    f["dow"] = f["ts"].dt.dayofweek  # Mon=0 .. Sun=6
    f["weekend"] = f["dow"].isin([5, 6])
    f["sess"] = np.where(f["hour"] < 8, "ASIA",
                         np.where(f["hour"] < 16, "EU", "US"))
    return f


def main() -> None:
    evs, kcache = [], {}
    pairs = [c.pair for c in load_universe()]
    for i, pair in enumerate(pairs, 1):
        e = detect_events(pair)
        if len(e):
            evs.append(e)
            kc = pd.read_parquet(
                KL_DIR / f"{pair}.parquet",
                columns=["open_time", "open", "high", "low", "close"])
            kc["ts"] = pd.to_datetime(kc["open_time"], unit="ms", utc=True)
            kcache[pair] = kc.set_index("ts").sort_index()
        if i % 40 == 0:
            print(f"[{i}/{len(pairs)}] events {sum(len(x) for x in evs)}", flush=True)
    ev = pd.concat(evs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    print(f"events total: {len(ev)}", flush=True)

    tr = simulate(ev, kcache, "market", "none", 0.0025, False)
    f = tr[tr["filled"]].copy()
    f = _annotate(f)
    print(f"filled trades: {len(f)}  net mean overall: {f['ret'].mean():.4f}", flush=True)

    f_dev = f[f["ts"] < HOLDOUT_START]
    f_hold = f[f["ts"] >= HOLDOUT_START]

    windows = {"full": _bucket_tables(f),
               "dev_pre2025": _bucket_tables(f_dev),
               "holdout_2025plus": _bucket_tables(f_hold)}

    decision = {
        "ASIA_vs_rest": _decision_one(f, f_dev, f_hold, lambda x: x["sess"] == "ASIA"),
        "EU_vs_rest": _decision_one(f, f_dev, f_hold, lambda x: x["sess"] == "EU"),
        "US_vs_rest": _decision_one(f, f_dev, f_hold, lambda x: x["sess"] == "US"),
        "weekend_vs_weekday": _decision_one(f, f_dev, f_hold, lambda x: x["weekend"]),
    }
    any_real = any(v.get("REAL_FILTER") for v in decision.values())

    clustering = {
        "ASIA": _clustering(f[f["sess"] == "ASIA"]),
        "EU": _clustering(f[f["sess"] == "EU"]),
        "US": _clustering(f[f["sess"] == "US"]),
        "weekend": _clustering(f[f["weekend"]]),
        "weekday": _clustering(f[~f["weekend"]]),
    }

    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "trigger-timing conditioning; simulate market/none/25bps/no-deep; "
                "reused liqrev_v2; slice-only; see module docstring for frozen rule",
        "overall": {"n_events": int(len(ev)), "n_filled": int(len(f)),
                    "net_mean": round(float(f["ret"].mean()), 4),
                    "portfolio_15slots": portfolio(tr)},
        "windows": windows,
        "decision_rule": {"candidates": decision, "any_real_filter": bool(any_real)},
        "clustering": clustering,
    }

    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / "results_timing.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # --- console summary ---
    def row(tag, s):
        if s["n"] == 0:
            return f"  {tag:<9} n=0"
        return (f"  {tag:<9} n={s['n']:<4} mean={s['net_mean']:+.4f} "
                f"se={s['se']:.4f} med={s['net_median']:+.4f} win={s['win']:.3f}")
    for wname, w in windows.items():
        print(f"\n== {wname} : SESSIONS ==", flush=True)
        for k, s in w["sessions"].items():
            print(row(k, s), flush=True)
        print(f"== {wname} : WEEKEND ==", flush=True)
        for k, s in w["weekend_vs_weekday"].items():
            print(row(k, s), flush=True)
    print("\n== DOW (full, diagnostic) ==", flush=True)
    for k, s in windows["full"]["day_of_week"].items():
        print(row(k, s), flush=True)
    print("\n== DECISION RULE ==", flush=True)
    for k, v in decision.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nany_real_filter = {any_real}", flush=True)
    print("\n== CLUSTERING ==", flush=True)
    for k, v in clustering.items():
        print(f"  {k}: distinct_days={v['distinct_days']} of n={v['n']} "
              f"top3={v['top3_days']}", flush=True)
    print(f"\nartifacts -> {out}", flush=True)


if __name__ == "__main__":
    main()
