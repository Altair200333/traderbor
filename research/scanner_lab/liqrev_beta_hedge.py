"""Q26 — Beta-hedging liqrev v2 (long cascade alt + short BTC perp). Pre-registered.

FROZEN SPEC (registered 2026-07-10 BEFORE the first run; nothing changed after).

QUESTION: does a short-BTC hedge leg on each liqrev v2 trade improve the worst
regime — the 2022-style systemic collapse where bounces don't come (MW events
-4.4%/ev; the frozen overlay upweights INTO exactly that regime) — at an
acceptable cost to total return? Honest tradeoff: part of the cascade bounce IS
market beta (hedge gives back edge), and shorting BTC perp pays carry when
funding is negative (2025-26 climate; cite research/data/carry/results.json).

BASE (frozen v2 with overlay, unchanged):
  Events research/data/liqrev/ml_dataset.parquet, FILLED only (maker at
  trigger-close, TTL 1h fill-if-touched, disaster stop -20%, exit at close of
  bar i+24 = ts+25h, 10bps RT — all already inside net_ret). 15 slots x 1/15
  equity, slot busy [ts, ts+24h). Overlay weight w = min(2, 2*pctl(-btc_ret_6h))
  vs the DEV reference distribution (filled events with exit_known ts+25h <
  2024-12-25; the shipped 839-score bot artifact). eq *= 1 + w*net_ret/15
  (liqrev_ml_model machinery, reused via import).
  PARITY CHECK: reproduced daily curve vs
  research/data/liqrev/daily_equity_v2.parquet — max |equity diff| reported;
  must be < 1e-6 relative or the study aborts.

HEDGE LEG (per TAKEN trade): SHORT BTCUSDT perp, opened at ts+1h (start of the
  fill bar; 1m-path audit 4.3: 98% of fills within 5 min), closed at ts+25h
  (alt exit close). The dataset stores no per-trade exit ts, so stopped trades
  (~5%) keep the hedge to ts+25h — declared mismatch. BTC prices: 1m closes
  from research/data/binance_um/klines_1m/BTCUSDT.parquet, asof lookup.
  hedge_ret (per unit hedge notional) =
      -(btc_exit/btc_entry - 1) + funding_recv - 0.0005
  where 0.0005 = 5bps RT (most liquid instrument) and funding_recv = sum over
  8h settlements T (research/data/perp/funding/BTCUSDT.parquet) of
  fundingRate_T * overlap([ts+1h, ts+25h], (T-8h, T]) / 8h  — the short
  RECEIVES positive funding, PAYS negative funding, pro-rated at both edges.

CELLS (6 = 3 sizes x 2 activations, + unhedged reference; ALL reported):
  size  (a) 1to1: h = 1.0
        (b) beta: h = rolling 60-day daily beta of symbol vs BTC, POINT-IN-TIME
            (daily closes; window 60, min 30 obs; beta known at event = value
            as of the last COMPLETE day before the event day; missing -> 1.0,
            count reported), capped to [0.3, 2.0]
        (c) half: h = 0.5
  activation (i) always; (ii) mw_only: hedge only when overlay weight w >= 1.0.
  Hedge notional = h * alt slot notional = h*w/15 of equity. Trade equity
  factor f = 1 + (w*net_ret + w*h*hedge_ret)/15. The hedge does NOT consume a
  slot; it DOES count toward gross exposure: peak gross = max_t of
  sum over open trades of w*(1+h)/15, exposure window [ts+1h, ts+25h).
  Take-decisions are identical to base in every cell (same trades, same slots).

SPLITS: DEV = ts < 2025-01-01 (selects), LIVE = ts >= 2025-01-01 (verdicts).
  DEV/LIVE curve metrics = daily curve sliced at 2025-01-01, renormalized.

DEV SELECTION (frozen): among the 6 cells pick max DEV CAGR among cells with
  (DEV 2022 calendar-year return >= unhedged + 5pp) AND (DEV maxDD strictly
  smaller in magnitude); if none qualifies, pick max DEV-2022 return and let
  the verdict fail honestly.

VERDICT RULE (frozen numbers; applies to the ONE DEV-selected cell):
  ADOPT-CANDIDATE iff
   (a) DEV 2022 calendar-year return >= unhedged 2022 + 0.05 (5pp)
       [2022 is IN DEV and is a single episode: this clause is in-sample and
        n=1 by construction — stated openly]
   (b) DEV maxDD strictly improves (smaller magnitude) vs unhedged
   (c) LIVE CAGR >= 0.70 * unhedged LIVE CAGR
   (d) LIVE harvest month 2025-02 return >= 0.70 * unhedged 2025-02 return
  else NO — failing clause(s) reported.

REPORT per cell: full/DEV/LIVE CAGR+maxDD; by-year with 2022 explicit;
  per-event mean net (net_ret + h*hedge_ret); hedge P&L split: mean h*hedge_ret
  on alt-losers (beta protection) vs alt-winners (edge give-back); funding cost
  of the hedge per year (sum of w*h*funding_recv/15, equity contribution);
  month returns 2022-11 (FTX), 2024-08, 2025-02 vs unhedged; peak gross.

OUTPUTS: research/data/liqrev/results_beta_hedge.json; iff ADOPT-CANDIDATE also
  research/data/liqrev/daily_equity_hedged.parquet (DEV-selected cell curve).
Usage: python liqrev_beta_hedge.py
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
    HOLDOUT_TRAIN_CUT, SLOTS, apply_rule, fit_score, make_xy, pctl,
)
from universe import REPO_ROOT  # noqa: E402

BTC_1M = REPO_ROOT / "research" / "data" / "binance_um" / "klines_1m" / "BTCUSDT.parquet"
FUND_BTC = REPO_ROOT / "research" / "data" / "perp" / "funding" / "BTCUSDT.parquet"
KL_1H = REPO_ROOT / "research" / "data" / "v3" / "klines" / "1h"
CARRY_JSON = REPO_ROOT / "research" / "data" / "carry" / "results.json"
STORED_CURVE = ART_DIR / "daily_equity_v2.parquet"
OUT_JSON = ART_DIR / "results_beta_hedge.json"
OUT_CURVE = ART_DIR / "daily_equity_hedged.parquet"

HEDGE_RT = 0.0005          # 5 bps RT on the BTC leg
BETA_LO, BETA_HI = 0.3, 2.0
BETA_WIN, BETA_MIN = 60, 30
EIGHT_H = pd.Timedelta("8h")
CELLS = [(s, a) for s in ("1to1", "beta", "half") for a in ("always", "mw_only")]
LIVE_START = HOLDOUT_START.tz_localize(None)   # 2025-01-01, naive for daily dates


def py(o):
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o is None or not np.isfinite(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


# ------------------------------------------------------------- base machinery
def taken_trades(ev: pd.DataFrame, keep: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """EXACT run_portfolio/sim_daily take logic; returns the taken-trade table."""
    order = np.argsort(ev["ts"].to_numpy())
    ts_arr = ev["ts"].to_numpy()[order]
    ex_arr = ev["exit_slot"].to_numpy()[order]
    r_arr = ev["net_ret"].to_numpy()[order]
    sym = ev["symbol"].to_numpy()[order]
    k_arr, w_arr = keep[order], w[order]
    busy, rows = [], []
    for i in range(len(ev)):
        busy = [b for b in busy if b > ts_arr[i]]
        if k_arr[i] and len(busy) < SLOTS:
            busy.append(ex_arr[i])
            rows.append((ts_arr[i], ex_arr[i], sym[i],
                         float(w_arr[i]), float(r_arr[i])))
    td = pd.DataFrame(rows, columns=["ts", "exit_slot", "symbol", "w", "net_ret"])
    td["ts"] = pd.to_datetime(td["ts"], utc=True)
    td["exit_slot"] = pd.to_datetime(td["exit_slot"], utc=True)
    return td


def daily_from_factors(td: pd.DataFrame, fac: np.ndarray) -> pd.DataFrame:
    """sim_daily convention: factors multiply on the ENTRY day."""
    eday = td["ts"].dt.floor("1D")
    days = pd.date_range(eday.min(), td["exit_slot"].max().floor("1D"),
                         freq="D", tz="UTC")
    dfac = (pd.Series(fac, index=td.index).groupby(eday).prod()
            .reindex(days).fillna(1.0))
    return pd.DataFrame({"date": days.tz_localize(None),
                         "ret": dfac.to_numpy() - 1.0,
                         "equity": np.cumprod(dfac.to_numpy())})


def slice_metrics(daily: pd.DataFrame, start=None, end=None) -> dict:
    m = np.ones(len(daily), bool)
    if start is not None:
        m &= (daily["date"] >= start).to_numpy()
    if end is not None:
        m &= (daily["date"] < end).to_numpy()
    r = daily.loc[m, "ret"].to_numpy()
    dates = pd.DatetimeIndex(daily.loc[m, "date"])
    eq = np.cumprod(1.0 + r)
    years = max((dates[-1] - dates[0]).days, 1) / 365.25
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return {"total": round(float(eq[-1] - 1), 4),
            "cagr": round(float(eq[-1] ** (1 / years) - 1), 4),
            "maxDD": round(float(dd.min()), 4)}


def by_year(daily: pd.DataFrame) -> dict:
    r = daily.set_index("date")["ret"]
    yr = (1 + r).groupby(r.index.year).prod() - 1
    return {str(k): round(float(v), 4) for k, v in yr.items()}


def month_ret(daily: pd.DataFrame, ym: str) -> float | None:
    r = daily.set_index("date")["ret"]
    mo = (1 + r).resample("MS").prod() - 1
    t = pd.Timestamp(ym)
    return round(float(mo.loc[t]), 4) if t in mo.index else None


# ------------------------------------------------------------------ BTC data
def load_btc_px() -> pd.Series:
    k = pd.read_parquet(BTC_1M, columns=["open_time", "close"])
    idx = pd.to_datetime(k["open_time"].to_numpy() + 60_000, unit="ms", utc=True)
    return pd.Series(k["close"].to_numpy(), index=idx).sort_index()


def px_asof(px: pd.Series, times: pd.Series) -> tuple[np.ndarray, float]:
    ti = pd.DatetimeIndex(times).tz_convert("UTC")
    pos = px.index.searchsorted(ti, side="right") - 1
    if (pos < 0).any():
        raise RuntimeError("BTC price lookup before data start")
    gap = (ti.asi8 - px.index.asi8[pos]) / 60e9
    return px.to_numpy()[pos], float(np.max(gap))


def load_funding() -> pd.Series:
    f = pd.read_parquet(FUND_BTC)
    ts = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("1h")
    s = pd.Series(f["fundingRate"].to_numpy(), index=ts)
    return s[~s.index.duplicated(keep="last")].sort_index()


def funding_recv(fr: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """short receives +rate; accrual window (T-8h, T] pro-rated vs [t0, t1]."""
    i0 = fr.index.searchsorted(t0, side="right")
    i1 = fr.index.searchsorted(t1 + EIGHT_H, side="left")
    tot = 0.0
    for T, r in fr.iloc[i0:i1].items():
        ov = (min(t1, T) - max(t0, T - EIGHT_H)).total_seconds()
        if ov > 0:
            tot += float(r) * ov / EIGHT_H.total_seconds()
    return tot


# ---------------------------------------------------------------- PIT betas
def pit_betas(td: pd.DataFrame, btc_daily_ret: pd.Series) -> tuple[np.ndarray, int]:
    betas = np.full(len(td), np.nan)
    for sym, g in td.groupby("symbol"):
        p = KL_1H / f"{sym}.parquet"
        if not p.exists():
            continue
        k = pd.read_parquet(p, columns=["open_time", "close"])
        s = pd.Series(k["close"].to_numpy(),
                      index=pd.to_datetime(k["open_time"], unit="ms", utc=True))
        sd = s.resample("1D").last().pct_change()
        d = pd.concat([sd.rename("s"), btc_daily_ret.rename("b")], axis=1).dropna()
        if len(d) < BETA_MIN + 1:
            continue
        cov = d["s"].rolling(BETA_WIN, min_periods=BETA_MIN).cov(d["b"])
        var = d["b"].rolling(BETA_WIN, min_periods=BETA_MIN).var()
        beta = (cov / var).dropna()
        if beta.empty:
            continue
        for i in g.index:
            cut = td.loc[i, "ts"].floor("1D") - pd.Timedelta("1D")
            b = beta.asof(cut)
            if pd.notna(b):
                betas[i] = b
    n_default = int(np.isnan(betas).sum())
    betas = np.clip(np.where(np.isnan(betas), 1.0, betas), BETA_LO, BETA_HI)
    return betas, n_default


# ------------------------------------------------------------------- gross --
def peak_gross(td: pd.DataFrame, h: np.ndarray) -> float:
    expo = (td["w"].to_numpy() * (1.0 + h)) / SLOTS
    t0 = (td["ts"] + pd.Timedelta("1h")).to_numpy()
    t1 = (td["ts"] + pd.Timedelta("25h")).to_numpy()
    ev = sorted([(t, +e) for t, e in zip(t0, expo)]
                + [(t, -e) for t, e in zip(t1, expo)])
    cur = mx = 0.0
    for _, de in ev:
        cur += de
        mx = max(mx, cur)
    return round(mx, 4)


# --------------------------------------------------------------------- main --
def main() -> None:
    raw = pd.read_parquet(ART_DIR / "ml_dataset.parquet")
    df = make_xy(raw)
    ref = df[(df["ts"] < HOLDOUT_START)
             & (df["exit_known"] < HOLDOUT_TRAIN_CUT)]
    s_tr, s_te = fit_score("M1_dumb", ref, df)
    p = pctl(s_tr, s_te)
    keep, w = apply_rule("R3_rankw", p)
    td = taken_trades(df, keep, w)
    print(f"filled events {len(df)}, taken trades {len(td)}, "
          f"ref distribution n={len(ref)}")

    # ---- 1. PARITY: unhedged curve vs stored daily_equity_v2.parquet -------
    base_fac = 1.0 + td["w"].to_numpy() * td["net_ret"].to_numpy() / SLOTS
    base_daily = daily_from_factors(td, base_fac)
    stored = pd.read_parquet(STORED_CURVE)
    parity = {"rows_repro": len(base_daily), "rows_stored": len(stored)}
    if len(stored) == len(base_daily):
        dev_abs = np.abs(base_daily["equity"].to_numpy()
                         - stored["equity"].to_numpy())
        parity["max_abs_equity_dev"] = float(dev_abs.max())
        parity["max_rel_equity_dev"] = float(
            (dev_abs / stored["equity"].to_numpy()).max())
        parity["dates_match"] = bool(
            (base_daily["date"].to_numpy() == stored["date"].to_numpy()).all())
    print("PARITY:", json.dumps(py(parity)))
    if not (parity.get("dates_match") and parity["max_rel_equity_dev"] < 1e-6):
        raise SystemExit("PARITY FAILED — aborting per frozen spec")

    # ---- 2. hedge leg inputs ------------------------------------------------
    px = load_btc_px()
    fr = load_funding()
    t0 = td["ts"] + pd.Timedelta("1h")
    t1 = td["ts"] + pd.Timedelta("25h")
    btc_e, gap_e = px_asof(px, t0)
    btc_x, gap_x = px_asof(px, t1)
    btc_ret = btc_x / btc_e - 1.0
    fund = np.array([funding_recv(fr, a, b) for a, b in zip(t0, t1)])
    hedge_ret = -btc_ret + fund - HEDGE_RT
    btc_daily_ret = px.resample("1D").last().pct_change()
    betas, n_beta_default = pit_betas(td, btc_daily_ret)
    print(f"hedge inputs: max px staleness {max(gap_e, gap_x):.1f} min; "
          f"beta default(=1.0) on {n_beta_default}/{len(td)} trades; "
          f"mean funding/trade {fund.mean()*1e4:+.2f} bps (short receives)")

    # funding climate citation (carry study) + realized BTC funding APR by year
    fund_by_year = {str(y): round(float(g.mean() * 3 * 365), 4)
                    for y, g in fr.groupby(fr.index.year)}
    carry_cite = None
    if CARRY_JSON.exists():
        cj = json.loads(CARRY_JSON.read_text(encoding="utf-8"))
        carry_cite = {"source": str(CARRY_JSON),
                      "climate_by_year": cj.get("climate_by_year")}

    wv, rv = td["w"].to_numpy(), td["net_ret"].to_numpy()
    years_v = td["ts"].dt.year.to_numpy()

    def cell_report(h: np.ndarray, name: str) -> dict:
        fac = 1.0 + (wv * rv + wv * h * hedge_ret) / SLOTS
        daily = daily_from_factors(td, fac)
        act = h > 0
        hr = h * hedge_ret
        losers, winners = act & (rv < 0), act & (rv > 0)
        fcontrib = pd.Series(wv * h * fund / SLOTS, index=years_v)
        rep = {
            "cell": name,
            "full": slice_metrics(daily),
            "DEV": slice_metrics(daily, end=LIVE_START),
            "LIVE": slice_metrics(daily, start=LIVE_START),
            "by_year": by_year(daily),
            "months": {"2022-11_FTX": month_ret(daily, "2022-11-01"),
                       "2024-08": month_ret(daily, "2024-08-01"),
                       "2025-02": month_ret(daily, "2025-02-01")},
            "per_event_mean_net": round(float((rv + hr).mean()), 5),
            "n_hedged": int(act.sum()),
            "hedge_decomp": {
                "mean_hr_on_losers_protection": round(float(hr[losers].mean()), 5)
                if losers.any() else None,
                "n_losers": int(losers.sum()),
                "mean_hr_on_winners_giveback": round(float(hr[winners].mean()), 5)
                if winners.any() else None,
                "n_winners": int(winners.sum()),
                "mean_hr_all_hedged": round(float(hr[act].mean()), 5)
                if act.any() else None,
                "total_equity_contrib_approx": round(
                    float((wv * h * hedge_ret / SLOTS).sum()), 4)},
            "funding_equity_contrib_by_year": {
                str(y): round(float(v), 5)
                for y, v in fcontrib.groupby(level=0).sum().items()},
            "peak_gross": peak_gross(td, h),
        }
        return rep, daily

    # ---- 3. all cells --------------------------------------------------------
    unhedged, _ = cell_report(np.zeros(len(td)), "unhedged")
    reports, curves = [unhedged], {}
    for size, actm in CELLS:
        hb = {"1to1": np.ones(len(td)), "beta": betas,
              "half": np.full(len(td), 0.5)}[size]
        mask = np.ones(len(td)) if actm == "always" else (wv >= 1.0).astype(float)
        name = f"{size}_{actm}"
        rep, daily = cell_report(hb * mask, name)
        reports.append(rep)
        curves[name] = daily
        print(f"cell {name:14s} DEV cagr={rep['DEV']['cagr']:+.4f} "
              f"maxDD={rep['DEV']['maxDD']:+.4f} 2022={rep['by_year'].get('2022')} "
              f"| LIVE cagr={rep['LIVE']['cagr']:+.4f} "
              f"2025-02={rep['months']['2025-02']} gross={rep['peak_gross']}")

    # ---- 4. DEV selection (frozen rule) --------------------------------------
    u = unhedged
    u2022, udd = u["by_year"]["2022"], u["DEV"]["maxDD"]
    qual = [r for r in reports[1:]
            if r["by_year"].get("2022", -9) >= u2022 + 0.05
            and abs(r["DEV"]["maxDD"]) < abs(udd)]
    if qual:
        chosen = max(qual, key=lambda r: r["DEV"]["cagr"])
        sel_note = "max DEV CAGR among cells passing 2022+5pp AND maxDD-improve"
    else:
        chosen = max(reports[1:], key=lambda r: r["by_year"].get("2022", -9))
        sel_note = "NO cell passed (2022+5pp AND maxDD-improve); fallback = max DEV 2022"
    print(f"\nDEV-SELECTED: {chosen['cell']} ({sel_note})")

    # ---- 5. verdict (frozen) --------------------------------------------------
    c2022 = chosen["by_year"]["2022"]
    a = c2022 >= u2022 + 0.05
    b = abs(chosen["DEV"]["maxDD"]) < abs(udd)
    c = chosen["LIVE"]["cagr"] >= 0.70 * u["LIVE"]["cagr"]
    um = u["months"]["2025-02"]
    cm = chosen["months"]["2025-02"]
    d = (cm is not None and um is not None and cm >= 0.70 * um)
    adopt = a and b and c and d
    verdict = {
        "chosen_cell": chosen["cell"], "selection_note": sel_note,
        "clause_a_dev2022_plus5pp": {"pass": bool(a), "chosen": c2022,
                                     "unhedged": u2022, "needed": u2022 + 0.05,
                                     "note": "IN-SAMPLE, single episode (n=1)"},
        "clause_b_dev_maxdd_improves": {"pass": bool(b),
                                        "chosen": chosen["DEV"]["maxDD"],
                                        "unhedged": udd},
        "clause_c_live_cagr_70pct": {"pass": bool(c),
                                     "chosen": chosen["LIVE"]["cagr"],
                                     "unhedged": u["LIVE"]["cagr"],
                                     "needed": round(0.70 * u["LIVE"]["cagr"], 4)},
        "clause_d_live_202502_70pct": {"pass": bool(d), "chosen": cm,
                                       "unhedged": um,
                                       "needed": round(0.70 * um, 4)
                                       if um is not None else None},
        "verdict": "ADOPT-CANDIDATE" if adopt else "NO",
    }
    print("VERDICT:", json.dumps(py(verdict), indent=1))

    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Q26: beta-hedge liqrev v2 (short BTC leg) for the systemic tail",
        "spec": "see module docstring (frozen pre-registration)",
        "parity_check": parity,
        "hedge_inputs": {
            "n_taken": len(td), "btc_px_max_staleness_min": max(gap_e, gap_x),
            "n_beta_default": n_beta_default,
            "mean_funding_recv_per_trade": round(float(fund.mean()), 6),
            "mean_btc_leg_ret": round(float((-btc_ret).mean()), 6),
            "hedge_rt_cost": HEDGE_RT,
            "btc_realized_funding_apr_by_year": fund_by_year,
            "carry_study_citation": carry_cite},
        "cells": reports,
        "dev_selection": {"chosen": chosen["cell"], "note": sel_note},
        "verdict": verdict,
        "honesty_notes": [
            "2022 tail clause is in-sample by construction (2022 is in DEV) and "
            "n=1 systemic episode (FTX; span starts 2022-07 so LUNA not covered).",
            "hedge assumes BTC short always available at 5bps RT.",
            "betas point-in-time only (last complete day before event); "
            f"{n_beta_default} trades defaulted to beta=1.0.",
            "dataset stores no per-trade exit ts: stopped trades (~5%) keep the "
            "hedge to ts+25h (small mismatch, both legs, declared).",
            "mw_only cells isolate what the hedge adds BEYOND the overlay's own "
            "down-weighting (overlay already shrinks idio trades; the hedge "
            "targets the MW trades the overlay upweights)."],
    }
    OUT_JSON.write_text(json.dumps(py(out), indent=2), encoding="utf-8")
    print(f"\nartifacts -> {OUT_JSON}")

    if adopt:
        dcur = curves[chosen["cell"]].copy()
        dcur.to_parquet(OUT_CURVE, index=False)
        print(f"ADOPT-CANDIDATE curve -> {OUT_CURVE}")

    print("\nFINAL_JSON_COMPACT:")
    print(json.dumps(py({"parity": parity, "verdict": verdict,
                         "cells": [{k: r[k] for k in
                                    ("cell", "full", "DEV", "LIVE", "by_year",
                                     "months", "per_event_mean_net",
                                     "hedge_decomp",
                                     "funding_equity_contrib_by_year",
                                     "peak_gross")}
                                   for r in reports]})))


if __name__ == "__main__":
    main()
