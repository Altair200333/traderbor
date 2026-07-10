"""Q19 CONDITIONING AUDIT — does the token-unlock calendar contaminate liqrev v2?
+ EXPORT: frozen liqrev v2 WITH-overlay daily equity curve (full span).

================================ FROZEN SPEC ================================
Frozen 2026-07-09 BEFORE the first full run. This is an AUDIT of an already
frozen, holdout-spent strategy (liqrev v2 with the mw/idio sizing overlay).
PRE-COMMIT: NO parameter changes to the frozen config regardless of findings.
The only allowed outcome is a documented candidate veto/down-weight
recommendation OR a clean NO CHANGE. No new file is modified; this script only
reads existing artifacts and writes two new artifacts.

HYPOTHESIS. A liqrev cascade that fires on a token inside its pre-cliff window
(0-30d before a >=1% or >=3% unlock) may be ANTICIPATORY / INFORMED selling
rather than forced deleveraging, and may bounce worse (like calm-BTC idio
events do). If true the unlock calendar gives liqrev a cheap veto/down-weight.
If false/underpowered a clean null is also valuable.

DATA / JOIN.
  ml_dataset.parquet : 1308 liqrev events; LABEL = net_ret (frozen maker
      config net return); FILLED events (1280) carry the label.
  unlock_events.parquet : DefiLlama cliffs. JOIN KEY = ml.symbol == unlock.pair
      (both Binance pair strings, e.g. AAVEUSDT). unlock.symbol is the bare
      ticker (AAVE) and does NOT join. Cliff time = unlock.event_date parsed at
      00:00 UTC. (DATA SURPRISE: unlock.event_ts is in SECONDS here, not ms as
      the brief warned; event_date string is unambiguous so we use it.)

TAGGING (per event; mutually exclusive, priority order top->bottom):
  d_next(thr)  = days to the NEXT cliff with frac_supply>=thr strictly after ts
  d_since(thr) = days since the LAST cliff with frac_supply>=thr at/before ts
  PRE30_LARGE : 0 < d_next(3%)  <= 30
  PRE30_SMALL : 0 < d_next(1%)  <= 30            (=> nearest within-30 is <3%)
  POST7       : 0 <= d_since(1%) <= 7
  CLEAN       : symbol HAS unlock data, none of the above
  NODATA      : symbol not in the unlock pair set

ANALYSIS (DEV = ts < 2025-01-01 for estimation; 2025-26 = burnt window, SIGN
CHECK ONLY, higher bar):
  1. Contamination: count-share per category/year (all events) + filled-P&L
     share (sum net_ret) per category/year.
  2. Mean net_ret per category with DAY-CLUSTERED bootstrap CIs (resample
     calendar days, 1000 iters, seed 11 — the lab's standard, mirrors
     liqrev_ml_features.py). Same table on 2025-26 (sign check).
  3. Interaction with the mw/idio axis: terciles of btc_ret_6h on DEV (mw =
     lowest btc_ret_6h = market-wide deleveraging; idio = highest = calm BTC).
     Within mw and idio: PRE30(any) mean net_ret vs CLEAN. Small n expected.
  4. Raw forward returns: PRIMARY = net_ret. ml_dataset has NO independent
     gross-forward column (ret_24h is TRAILING, corr(ret_24h,net_ret)<0), so no
     separate raw-forward table is produced; documented, not hidden.

VERDICT RULE (pre-registered). PRE30_LARGE (or PRE30_SMALL) is a candidate VETO
only if ALL of:
  (a) DEV mean net_ret <= CLEAN mean - 150 bps (0.0150)
  (b) day-clustered CI of the (category - CLEAN) difference excludes 0
  (c) sign of the difference agrees in 2025-26 (both < 0)
  (d) n_category >= 25 on DEV
Anything weaker = NO CHANGE (logged as footnote). Underpower expected; say so.

EXPORT (independent of verdict). Regenerate the frozen liqrev v2 WITH-overlay
portfolio over the full span by REUSING liqrev_ml_model.py's exact code path:
the overlay == M1_dumb x R3_rankw, i.e. score = -btc_ret_6h -> percentile in
the frozen DEV/tr_hold distribution -> slot weight w = min(2, 2*pctl), run
through the 15-slot run_portfolio machinery. Reference distribution for the
percentile = tr_hold = DEV filled events with exit_known < 2024-12-25 (the
exact frozen calibration that produced the deployed holdout numbers; DEV ~=
tr_hold up to the last few Dec-2024 events). Save daily_equity_v2.parquet
[date, ret, equity, n_open]. SANITY: reproduce holdout 2025-26 CAGR ~0.325,
maxDD ~-2.4% and a positive full-period equity; report deltas, do not silently
ship a different config.

Artifacts:
  research/data/liqrev/results_unlock_interaction.json
  research/data/liqrev/daily_equity_v2.parquet
Usage: python liqrev_unlock_interaction.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from liqrev_ml_features import ART_DIR, HOLDOUT_START  # noqa: E402
from liqrev_ml_model import (  # noqa: E402
    make_xy, run_portfolio, apply_rule, fit_score, pctl,
    HOLDOUT_TRAIN_CUT, SLOTS,
)

UNLOCK_PATH = ART_DIR.parent / "unlocks" / "unlock_events.parquet"
SEED = 11
N_BOOT = 1000
LARGE = 0.03
SMALL = 0.01
CATS = ["PRE30_LARGE", "PRE30_SMALL", "POST7", "CLEAN", "NODATA"]
VETO_MARGIN = 0.0150   # 150 bps
MIN_N = 25


def py(o):
    """make numpy types JSON serializable."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 6)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


# ----------------------------------------------------------------- tagging --
def build_tagger(un: pd.DataFrame):
    """returns (tag_fn, pairs_set). tag_fn(symbol, ts_ns) -> category str."""
    un = un.copy()
    # naive UTC wall-clock in int64 NANOSECONDS (must match event ts units)
    un["cliff_ns"] = (pd.to_datetime(un["event_date"]).to_numpy()
                      .astype("datetime64[ns]").astype("int64"))
    pairs = set(un["pair"].unique())
    # per-pair arrays of (cliff_ns, frac)
    per = {}
    for p, g in un.groupby("pair"):
        per[p] = (g["cliff_ns"].to_numpy(), g["frac_supply"].to_numpy())
    NS_DAY = 86_400_000_000_000

    def tag(sym: str, ts_ns: int) -> str:
        if sym not in pairs:
            return "NODATA"
        cliff_ns, frac = per[sym]
        d = (cliff_ns - ts_ns) / NS_DAY          # >0 future, <0 past
        fut = d > 0
        past = d <= 0
        # next large ahead
        m = fut & (frac >= LARGE)
        d_next_large = d[m].min() if m.any() else np.inf
        m = fut & (frac >= SMALL)
        d_next_1 = d[m].min() if m.any() else np.inf
        m = past & (frac >= SMALL)
        d_since_1 = (-d[m]).min() if m.any() else np.inf
        if 0 < d_next_large <= 30:
            return "PRE30_LARGE"
        if 0 < d_next_1 <= 30:
            return "PRE30_SMALL"
        if 0 <= d_since_1 <= 7:
            return "POST7"
        return "CLEAN"

    return tag, pairs


# ------------------------------------------------------- clustered bootstrap --
def cluster_boot_means(sub: pd.DataFrame, cats, n_boot=N_BOOT, seed=SEED):
    """Day-clustered bootstrap of per-category mean net_ret + paired diffs vs
    CLEAN. Returns (point, ci, diff) dicts. Resamples calendar days (lab std)."""
    sub = sub.reset_index(drop=True)
    day = sub["ts"].dt.floor("1D")
    groups = {d: g.index.to_numpy() for d, g in sub.groupby(day)}
    days = np.array(list(groups.keys()))
    net = sub["net_ret"].to_numpy()
    cat = sub["cat"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = {c: np.full(n_boot, np.nan) for c in cats}
    for b in range(n_boot):
        sd = rng.choice(len(days), size=len(days), replace=True)
        idx = np.concatenate([groups[days[j]] for j in sd])
        ni, ci = net[idx], cat[idx]
        for c in cats:
            mask = ci == c
            if mask.any():
                boot[c][b] = ni[mask].mean()
    point, ci = {}, {}
    for c in cats:
        m = cat == c
        n = int(m.sum())
        point[c] = {
            "n": n,
            "n_days": int(day[m].nunique()) if n else 0,
            "mean": float(net[m].mean()) if n else None,
            "ci_lo": float(np.nanpercentile(boot[c], 2.5)) if n else None,
            "ci_hi": float(np.nanpercentile(boot[c], 97.5)) if n else None,
        }
    diff = {}
    base = boot["CLEAN"]
    for c in cats:
        if c == "CLEAN" or (cat == c).sum() == 0:
            continue
        d = boot[c] - base
        lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
        pm = point[c]["mean"]
        cm = point["CLEAN"]["mean"]
        diff[c] = {
            "diff_mean": (pm - cm) if (pm is not None and cm is not None) else None,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_0": bool(lo > 0 or hi < 0),
        }
    return point, diff


# ------------------------------------------------------------------- export --
def frozen_overlay_weights(events: pd.DataFrame, ref: pd.DataFrame):
    """EXACT frozen v2 overlay = liqrev_ml_model M1_dumb x R3_rankw path."""
    s_tr, s_te = fit_score("M1_dumb", ref, events)   # = -btc_ret_6h
    p = pctl(s_tr, s_te)
    keep, w = apply_rule("R3_rankw", p)              # w = min(2, 2*pctl)
    return keep, w


def sim_daily(ev: pd.DataFrame, keep: np.ndarray, w: np.ndarray, slots=SLOTS):
    """Replicates run_portfolio's 15-slot sequential sim but records each taken
    trade's (entry, exit, factor) so we can build a DAILY equity curve. The
    per-trade factor and take-decision logic are identical to run_portfolio."""
    order = np.argsort(ev["ts"].to_numpy())
    ts_arr = ev["ts"].to_numpy()[order]
    ex_arr = ev["exit_slot"].to_numpy()[order]
    r_arr = ev["net_ret"].to_numpy()[order]
    k_arr = keep[order]
    w_arr = w[order]
    busy, trades = [], []
    for i in range(len(ev)):
        busy = [b for b in busy if b > ts_arr[i]]
        if k_arr[i] and len(busy) < slots:
            f = 1.0 + w_arr[i] * r_arr[i] / slots
            busy.append(ex_arr[i])
            trades.append((ts_arr[i], ex_arr[i], f))
    td = pd.DataFrame(trades, columns=["entry", "exit", "factor"])
    td["entry"] = pd.to_datetime(td["entry"], utc=True)
    td["exit"] = pd.to_datetime(td["exit"], utc=True)
    eday = td["entry"].dt.floor("1D")
    start = eday.min()
    end = td["exit"].max().floor("1D")
    days = pd.date_range(start, end, freq="D", tz="UTC")
    dfac = td.groupby(eday)["factor"].prod().reindex(days).fillna(1.0)
    ret = dfac.to_numpy() - 1.0
    equity = np.cumprod(dfac.to_numpy())
    # n_open: count trades whose [entry, exit) intersects each calendar day
    n_open = pd.Series(0, index=days)
    for e, x in zip(td["entry"], td["exit"]):
        d0, d1 = e.floor("1D"), x.floor("1D")
        rng_days = pd.date_range(d0, d1, freq="D", tz="UTC")
        if x == d1 and len(rng_days) > 1:      # exit exactly at midnight
            rng_days = rng_days[:-1]
        n_open.loc[rng_days] += 1
    out = pd.DataFrame({
        "date": days.tz_localize(None),
        "ret": ret,
        "equity": equity,
        "n_open": n_open.to_numpy(),
    })
    return out, td


def curve_metrics(equity: np.ndarray, dates) -> dict:
    dates = pd.DatetimeIndex(dates)
    years = max((dates[-1] - dates[0]).days, 1) / 365.25
    dd = equity / np.maximum.accumulate(equity) - 1.0
    return {"total": float(equity[-1] - 1.0),
            "cagr": float(equity[-1] ** (1 / years) - 1.0),
            "maxDD": float(dd.min()),
            "n_days": int(len(equity))}


# --------------------------------------------------------------------- main --
def main() -> None:
    raw = pd.read_parquet(ART_DIR / "ml_dataset.parquet")
    un = pd.read_parquet(UNLOCK_PATH)
    tag, pairs = build_tagger(un)

    # tag ALL events (count-share) using event ts in int64 NANOSECONDS,
    # naive UTC (ml.ts is datetime64[ms,UTC] -> force ns to match cliff_ns)
    ts_ns = (raw["ts"].to_numpy().astype("datetime64[ns]").astype("int64"))
    raw = raw.copy()
    raw["cat"] = [tag(s, t) for s, t in zip(raw["symbol"].to_numpy(), ts_ns)]

    overlap = len(set(raw["symbol"].unique()) & pairs)
    join = {"key": "ml.symbol == unlock.pair",
            "ml_symbols": int(raw["symbol"].nunique()),
            "unlock_pairs": int(len(pairs)),
            "overlap_symbols": int(overlap),
            "unlock_symbol_join_overlap": int(
                len(set(raw["symbol"].unique()) & set(un["symbol"].unique())))}

    dev_all = raw[raw["ts"] < HOLDOUT_START]
    hold_all = raw[raw["ts"] >= HOLDOUT_START]

    # ---- 1. contamination (count share + filled P&L share) ------------------
    def contam(d):
        cnt = d["cat"].value_counts().reindex(CATS).fillna(0).astype(int)
        share = (cnt / len(d)).round(4)
        f = d[d["filled"]]
        pnl = f.groupby("cat")["net_ret"].sum().reindex(CATS).fillna(0.0)
        pnl_share = (pnl / pnl.sum()).round(4) if pnl.sum() != 0 else pnl * 0
        return {"n": int(len(d)),
                "count": cnt.to_dict(), "count_share": share.to_dict(),
                "pnl_share_filled": pnl_share.round(4).to_dict()}

    contamination = {"overall": contam(raw), "DEV": contam(dev_all),
                     "HOLD2526": contam(hold_all), "by_year": {}}
    for y, g in raw.groupby("year"):
        contamination["by_year"][int(y)] = contam(g)

    # ---- 2. category means + day-clustered CIs (filled) ---------------------
    dev_f = dev_all[dev_all["filled"]].copy()
    hold_f = hold_all[hold_all["filled"]].copy()
    dev_pt, dev_diff = cluster_boot_means(dev_f, CATS)
    hold_pt, hold_diff = cluster_boot_means(hold_f, CATS)

    # ---- 3. interaction with mw/idio axis (DEV filled) ----------------------
    dev_f["terc"] = pd.qcut(dev_f["btc_ret_6h"], 3, labels=["mw", "mid", "idio"])
    dev_f["pre30_any"] = dev_f["cat"].isin(["PRE30_LARGE", "PRE30_SMALL"])
    interaction = {}
    for t in ["mw", "idio", "mid"]:
        sub = dev_f[dev_f["terc"] == t]
        pre = sub[sub["pre30_any"]]["net_ret"]
        cln = sub[sub["cat"] == "CLEAN"]["net_ret"]
        interaction[t] = {
            "btc_ret_6h_range": [round(float(sub["btc_ret_6h"].min()), 4),
                                 round(float(sub["btc_ret_6h"].max()), 4)],
            "pre30_n": int(len(pre)),
            "pre30_mean": float(pre.mean()) if len(pre) else None,
            "clean_n": int(len(cln)),
            "clean_mean": float(cln.mean()) if len(cln) else None,
            "pre30_minus_clean": (float(pre.mean() - cln.mean())
                                  if len(pre) and len(cln) else None)}

    # ---- 5. verdict ---------------------------------------------------------
    verdict = {"rule": "veto iff (a) DEV mean<=CLEAN-150bps AND (b) diff CI "
                       "excludes 0 AND (c) sign agrees 2025-26 AND (d) n>=25",
               "categories": {}}
    any_veto = False
    for c in ["PRE30_LARGE", "PRE30_SMALL"]:
        dp, dd = dev_pt[c], dev_diff.get(c, {})
        hd = hold_diff.get(c, {})
        clean_dev = dev_pt["CLEAN"]["mean"]
        diff_dev = dd.get("diff_mean")
        hold_sign = hd.get("diff_mean")
        a = diff_dev is not None and diff_dev <= -VETO_MARGIN
        b = bool(dd.get("excludes_0", False))
        cc = (hold_sign is not None and diff_dev is not None
              and np.sign(hold_sign) == np.sign(diff_dev) and hold_sign < 0)
        e = dp["n"] >= MIN_N
        decision = "CANDIDATE_VETO" if (a and b and cc and e) else "NO CHANGE"
        any_veto = any_veto or (decision == "CANDIDATE_VETO")
        verdict["categories"][c] = {
            "dev_n": dp["n"], "dev_mean": dp["mean"], "clean_dev_mean": clean_dev,
            "dev_diff_vs_clean": diff_dev,
            "dev_diff_ci": [dd.get("ci_lo"), dd.get("ci_hi")],
            "hold_diff_vs_clean": hold_sign,
            "clause_a_margin_150bps": bool(a),
            "clause_b_ci_excludes_0": bool(b),
            "clause_c_sign_agrees_2526": bool(cc),
            "clause_d_n_ge_25": bool(e),
            "decision": decision}
    verdict["overall"] = ("CANDIDATE_VETO present" if any_veto
                          else "NO CHANGE (calendar does not clear the bar)")

    # ---- EXPORT: frozen v2 WITH-overlay daily equity ------------------------
    df = make_xy(raw)                       # filled, adds exit_slot/exit_known
    dev = df[df["ts"] < HOLDOUT_START].reset_index(drop=True)
    hold = df[df["ts"] >= HOLDOUT_START].reset_index(drop=True)
    ref = dev[dev["exit_known"] < HOLDOUT_TRAIN_CUT]   # tr_hold (frozen calib)

    # sanity: reproduce holdout numbers via the exact frozen path
    keep_h, w_h = frozen_overlay_weights(hold, ref)
    res_hold = run_portfolio(hold, keep_h, w_h)

    # full-span: same overlay/reference applied to ALL filled events
    keep_all, w_all = frozen_overlay_weights(df, ref)
    res_full = run_portfolio(df, keep_all, w_all)
    daily, td = sim_daily(df, keep_all, w_all)

    # daily-curve derived metrics + 2025-26 slice (normalized) cross-check
    dm_full = curve_metrics(daily["equity"].to_numpy(), daily["date"])
    mask26 = daily["date"] >= HOLDOUT_START.tz_localize(None)
    eq26 = daily.loc[mask26, "equity"].to_numpy()
    eq26 = eq26 / eq26[0]                    # renormalize to 1.0 at 2025 start
    dm_26 = curve_metrics(eq26, daily.loc[mask26, "date"])

    export_sanity = {
        "overlay": "M1_dumb x R3_rankw (frozen v2 with-overlay)",
        "ref_distribution": "tr_hold = DEV filled, exit_known < 2024-12-25",
        "holdout_repro_run_portfolio": {k: res_hold[k] for k in
            ["n_events", "n_kept", "kept_frac", "net_mean", "win",
             "total", "cagr", "maxDD"]},
        "holdout_target": {"cagr": 0.325, "maxDD": -0.024},
        "full_run_portfolio": {k: res_full[k] for k in
            ["n_events", "n_kept", "kept_frac", "net_mean", "win",
             "total", "cagr", "maxDD"]},
        "full_daily_curve": dm_full,
        "holdout_daily_curve_slice": dm_26,
        "final_equity_match": {
            "run_portfolio_total": res_full["total"],
            "daily_curve_total": round(dm_full["total"], 4)},
        "n_trades_taken_full": int(len(td))}

    daily.to_parquet(ART_DIR / "daily_equity_v2.parquet", index=False)

    # ---- assemble + write JSON ---------------------------------------------
    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Q19 conditioning audit: unlock calendar x liqrev v2",
        "spec": "see module docstring (frozen pre-registration)",
        "join": join,
        "data_notes": [
            "unlock.event_ts is SECONDS not ms; used event_date (unambiguous).",
            "net_ret is PRIMARY label; ml_dataset has no independent gross "
            "forward-return column (ret_24h is TRAILING, corr with net_ret<0).",
            "liqrev events cluster in crash days -> effective n is crash DAYS; "
            "all CIs are day-clustered (resample calendar days, 1000, seed 11)."],
        "category_defs": {
            "PRE30_LARGE": "0<d_next(>=3%)<=30",
            "PRE30_SMALL": "0<d_next(>=1%)<=30 (nearest within-30 is <3%)",
            "POST7": "0<=d_since(>=1%)<=7", "CLEAN": "has unlock data, none above",
            "NODATA": "symbol not in unlock pair set"},
        "contamination": contamination,
        "category_means": {"DEV": dev_pt, "HOLD2526": hold_pt},
        "diff_vs_clean": {"DEV": dev_diff, "HOLD2526": hold_diff},
        "interaction_mw_idio": interaction,
        "verdict": verdict,
        "export_sanity": export_sanity,
        "artifacts": {
            "results": str(ART_DIR / "results_unlock_interaction.json"),
            "daily_equity": str(ART_DIR / "daily_equity_v2.parquet")}}

    with open(ART_DIR / "results_unlock_interaction.json", "w") as fh:
        json.dump(py(out), fh, indent=2)

    # ---- concise console summary -------------------------------------------
    print("JOIN", json.dumps(join))
    print("\nCONTAM overall count_share:",
          json.dumps(py(contamination["overall"]["count_share"])))
    print("CONTAM overall pnl_share:",
          json.dumps(py(contamination["overall"]["pnl_share_filled"])))
    print("\nDEV category means (n / mean / CI):")
    for c in CATS:
        p = dev_pt[c]
        print(f"  {c:12s} n={p['n']:4d} nd={p['n_days']:3d} "
              f"mean={p['mean']} ci=[{p['ci_lo']},{p['ci_hi']}]")
    print("DEV diff vs CLEAN:", json.dumps(py(dev_diff)))
    print("\nHOLD2526 category means (n / mean / CI):")
    for c in CATS:
        p = hold_pt[c]
        print(f"  {c:12s} n={p['n']:4d} nd={p['n_days']:3d} "
              f"mean={p['mean']} ci=[{p['ci_lo']},{p['ci_hi']}]")
    print("HOLD2526 diff vs CLEAN:", json.dumps(py(hold_diff)))
    print("\nINTERACTION:", json.dumps(py(interaction)))
    print("\nVERDICT:", json.dumps(py(verdict), indent=1))
    print("\nEXPORT SANITY:", json.dumps(py(export_sanity), indent=1))
    print("\nwrote", ART_DIR / "results_unlock_interaction.json")
    print("wrote", ART_DIR / "daily_equity_v2.parquet", "rows", len(daily))


if __name__ == "__main__":
    main()
