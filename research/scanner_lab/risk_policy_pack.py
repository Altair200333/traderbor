"""RISK-POLICY PACK — post-hoc risk-policy measurement on the FROZEN liqrev v2 stream.

CRITICAL FRAMING (read first): this is NOT signal research. The detector, the
thresholds, the 1308-event stream, and the overlay weights (M1_dumb x R3_rankw,
w = min(2, 2*pctl(-btc_ret_6h)) against the frozen tr_hold reference) are all
FROZEN artifacts of prior pre-registered studies. Here we evaluate RISK POLICIES
(portfolio weight caps and drawdown kill-switches) applied ON TOP of that frozen
stream. The safe policy (weight cap at 1.0, i.e. never lever a slot) is adopted
BY DEFAULT for safety reasons (external audit 2026-07-10, defect 1: no gross
cap => theoretical 200% gross reached exactly in market-wide crashes). These
sims only MEASURE THE COST of that default — nothing here selects a policy on
LIVE/holdout performance. The kill-switch section is purely descriptive and
feeds the owner's kill-criteria freeze discussion (audit item 7); no verdict.

Tasks (audit-response items 1, 7, 8, 11):
  1. CAPPED OVERLAY COST — 15-slot sim (exact liqrev_ml_model machinery) under
     (a) w'=min(w,1.0) NEW DEFAULT, (b) w'=w (<=2.0) OLD reference,
     (c) w'=1 M0 equal-slot. Full-span + validation (>=2025) metrics,
     gross-exposure diagnostic (instantaneous open-slot weights / 15),
     showcase months 2022-11 (FTX) and 2025-02. Exports the capped curve.
  2. KILL-SWITCH REPLAY — trailing-peak drawdown kills {-8,-10,-12,-15}% x
     restart {never, resume after 30 flat days, resume after 60 flat days}
     replayed through BOTH the capped and uncapped daily curves. Convention:
     kill fires at the CLOSE of the day that breaches the level (that day's
     loss is taken); while OFF the policy earns 0; on resume the trailing
     peak resets to current policy equity (otherwise it re-kills instantly).
     Descriptive only.
  3. CONCENTRATION ADDENDUM (capped curve) — monthly table, leave-one-month-
     out CAGR for the top-5 months, top-N single-day log-return shares,
     trimmed monthly mean, by-year table.
  4. ACTIVE-DAY CORRELATION — liqrev(capped) x unlock S2 / S1 daily
     correlations restricted to days where BOTH series are non-zero, next to
     the naive full-series correlations (glue-addendum, audit item 11).
  5. PNL ATTRIBUTION BY WEIGHT BUCKET (uncapped policy b, auditor round-2) —
     total portfolio log return decomposed into events with w<1, w==1
     (epsilon band), w>1: full span, >=2025 validation window, and inside
     2022-11 / 2025-02. Answers whether the overlay's gain came from
     DOWN-weighting idio events or from UP-weighting (leveraging) MW crashes.

TERMINOLOGY: the >=2025 window is called VALIDATION (validation/selection
set) throughout — it was consumed by strategy selection across the research
program (audit defect 3) and is NOT a clean holdout.

REUSE, NOT REINVENTION: weights/portfolio/daily-curve code is imported from
liqrev_ml_model.py and liqrev_unlock_interaction.py (the exact code that
produced daily_equity_v2.parquet). The only new sim code is a gross-exposure
extension of sim_daily, and it is cross-checked: its equity/n_open must match
the imported sim_daily bit-for-bit for every policy, and the uncapped curve
must match the stored daily_equity_v2.parquet.

Inputs : research/data/liqrev/ml_dataset.parquet
         research/data/liqrev/daily_equity_v2.parquet (cross-check + kill replay)
         research/data/unlocks/daily_returns_s1.parquet, daily_returns_s2.parquet
Outputs: research/data/liqrev/results_policy_pack.json
         research/data/liqrev/daily_equity_v2_capped.parquet
Usage  : python risk_policy_pack.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from liqrev_ml_features import ART_DIR, HOLDOUT_START            # noqa: E402
from liqrev_ml_model import (                                    # noqa: E402
    make_xy, run_portfolio, HOLDOUT_TRAIN_CUT, SLOTS,
)
from liqrev_unlock_interaction import (                          # noqa: E402
    frozen_overlay_weights, sim_daily, py,
)

FRAMING = ("RISK-POLICY MEASUREMENT on FROZEN events/weights — not signal "
           "research. Weight cap 1.0 adopted BY DEFAULT for safety; these "
           "numbers only measure its cost. Kill-switch table is descriptive; "
           "no policy is selected on live performance.")

UNLOCK_DIR = ART_DIR.parent / "unlocks"
OUT_JSON = ART_DIR / "results_policy_pack.json"
OUT_CAPPED = ART_DIR / "daily_equity_v2_capped.parquet"

KILL_LEVELS = (-0.08, -0.10, -0.12, -0.15)
RESTARTS = (("never", None), ("resume_30d", 30), ("resume_60d", 60))
SHOWCASE = ("2022-11", "2025-02")


# ------------------------------------------------------- gross-aware daily sim
def sim_daily_gross(ev: pd.DataFrame, keep: np.ndarray, w: np.ndarray,
                    slots: int = SLOTS):
    """sim_daily (liqrev_unlock_interaction) + per-trade weight tracking so
    daily GROSS EXPOSURE columns can be built. Take-decision, factor and
    day-attribution logic are IDENTICAL to sim_daily; equality is asserted
    in main(). Two gross measures (both / slots, 1.0 == 100% of equity):
      gross       INSTANTANEOUS daily max — exact step function (+w at entry,
                  -w at exit_slot), max within each calendar day incl. the
                  carry-in level at day start. Bounded by max(w) by design.
      gross_union sum of weights of ALL trades whose [entry, exit) touches
                  the day — inflated by intraday slot turnover; upper-bound
                  diagnostic only, NOT instantaneous exposure."""
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
            trades.append((ts_arr[i], ex_arr[i], f, w_arr[i]))
    td = pd.DataFrame(trades, columns=["entry", "exit", "factor", "w"])
    td["entry"] = pd.to_datetime(td["entry"], utc=True)
    td["exit"] = pd.to_datetime(td["exit"], utc=True)
    eday = td["entry"].dt.floor("1D")
    days = pd.date_range(eday.min(), td["exit"].max().floor("1D"),
                         freq="D", tz="UTC")
    dfac = td.groupby(eday)["factor"].prod().reindex(days).fillna(1.0)
    n_open = pd.Series(0, index=days)
    union_w = pd.Series(0.0, index=days)
    for e, x, wt in zip(td["entry"], td["exit"], td["w"]):
        d0, d1 = e.floor("1D"), x.floor("1D")
        rng_days = pd.date_range(d0, d1, freq="D", tz="UTC")
        if x == d1 and len(rng_days) > 1:          # exit exactly at midnight
            rng_days = rng_days[:-1]
        n_open.loc[rng_days] += 1
        union_w.loc[rng_days] += wt
    # exact instantaneous gross: step function, exits apply BEFORE entries at
    # an identical timestamp (matches the sim's `b > ts` slot-free rule)
    deltas = pd.Series(
        np.concatenate([td["w"].to_numpy(), -td["w"].to_numpy()]),
        index=pd.DatetimeIndex(list(td["entry"]) + list(td["exit"])))
    step = deltas.sort_index(kind="stable").cumsum().groupby(level=0).last()
    carry = step.reindex(days, method="ffill").fillna(0.0)   # level at day start
    inday = step.groupby(step.index.floor("1D")).max().reindex(days)
    gross_inst = pd.concat([carry, inday], axis=1).max(axis=1).fillna(0.0)
    out = pd.DataFrame({
        "date": days.tz_localize(None),
        "ret": dfac.to_numpy() - 1.0,
        "equity": np.cumprod(dfac.to_numpy()),
        "n_open": n_open.to_numpy(),
        "gross": gross_inst.to_numpy() / slots,        # instantaneous daily max
        "gross_union": union_w.to_numpy() / slots,     # turnover-inflated bound
    })
    assert out["gross"].max() <= w[keep].max() + 1e-9  # cannot exceed max w
    return out, td


# ------------------------------------------------------------------- metrics -
def daily_metrics(daily: pd.DataFrame) -> dict:
    dates = pd.DatetimeIndex(daily["date"])
    r = pd.Series(daily["ret"].to_numpy(), index=dates)
    eq = daily["equity"].to_numpy()
    eq = eq / (eq[0] / (1.0 + r.iloc[0]))          # start-of-window basis 1.0
    years = max((dates[-1] - dates[0]).days, 1) / 365.25
    m = (1 + r).resample("ME").prod() - 1
    sd = float(r.std())
    return {"total": round(float(eq[-1] - 1), 4),
            "cagr": round(float(eq[-1] ** (1 / years) - 1), 4),
            "maxDD": round(float((eq / np.maximum.accumulate(eq) - 1).min()), 4),
            "sharpe": round(float(r.mean() / sd * np.sqrt(365)), 2) if sd > 0 else None,
            "worst_month": round(float(m.min()), 4),
            "worst_month_date": str(m.idxmin())[:7],
            "n_days": int(len(r))}


def monthly_returns(daily: pd.DataFrame) -> pd.Series:
    r = pd.Series(daily["ret"].to_numpy(), index=pd.DatetimeIndex(daily["date"]))
    m = (1 + r).resample("ME").prod() - 1
    m.index = m.index.strftime("%Y-%m")
    return m


def gross_diag(daily: pd.DataFrame) -> dict:
    active = daily["n_open"].to_numpy() > 0
    out = {"basis": ("gross = INSTANTANEOUS daily max (exact step function); "
                     "gross_union = all trades touching the day, inflated by "
                     "intraday slot turnover (upper bound only)")}
    for col in ("gross", "gross_union"):
        g = daily[col].to_numpy()
        over = g > 1.0 + 1e-9
        imax = int(np.argmax(g))
        out[col] = {
            "pct_days_over_100_all": round(float(over.mean()) * 100, 2),
            "pct_days_over_100_active": round(float(over[active].mean()) * 100, 2)
            if active.any() else None,
            "n_days_over_100": int(over.sum()),
            "max": round(float(g[imax]), 3),
            "max_date": str(daily["date"].iloc[imax].date()),
            "mean_active_days": round(float(g[active].mean()), 3)
            if active.any() else None}
    return out


# ------------------------------------------------------------- kill replay ---
def kill_replay(daily: pd.DataFrame, level: float, restart_days) -> dict:
    """Trailing-peak drawdown kill on a daily curve. Kill at close of breach
    day; OFF => 0 return; resume (if enabled) once (date - kill_date) >=
    restart_days, resetting the trailing peak to current policy equity.
    DESCRIPTIVE ONLY — feeds the kill-criteria freeze discussion."""
    dates = pd.DatetimeIndex(daily["date"])
    ret = daily["ret"].to_numpy()
    eq, peak, on = 1.0, 1.0, True
    kill_date = None
    triggers, episodes = [], []
    cur = None          # open off-episode: [kill_date, missed_factor]
    off_days = 0
    for i in range(len(ret)):
        if not on:
            if restart_days is not None and (dates[i] - kill_date).days >= restart_days:
                on = True
                peak = eq                       # reset reference on resume
                episodes.append({"kill": str(kill_date.date()),
                                 "resume": str(dates[i].date()),
                                 "strategy_ret_while_off": cur[1] - 1.0})
                cur = None
            else:
                off_days += 1
                cur[1] *= 1.0 + ret[i]
                continue
        eq *= 1.0 + ret[i]
        peak = max(peak, eq)
        if eq / peak - 1.0 <= level:
            on = False
            kill_date = dates[i]
            triggers.append(str(dates[i].date()))
            cur = [kill_date, 1.0]
    if cur is not None:                          # off at end of data
        episodes.append({"kill": str(cur[0].date()), "resume": "EOD(never)",
                         "strategy_ret_while_off": cur[1] - 1.0})
    nokill = float(np.prod(1.0 + ret))
    worst = (max(episodes, key=lambda e: e["strategy_ret_while_off"])
             if episodes else None)
    if worst is not None:
        worst = dict(worst)
        worst["strategy_ret_while_off"] = round(worst["strategy_ret_while_off"], 4)
    return {"kill_level": level,
            "n_triggers": len(triggers), "trigger_dates": triggers,
            "final_eq": round(eq, 4), "nokill_final_eq": round(nokill, 4),
            "ratio_vs_nokill": round(eq / nokill, 4),
            "off_days": off_days,
            "off_share_pct": round(100.0 * off_days / len(ret), 1),
            "worst_missed_episode": worst}


# ------------------------------------------------------------------- main ----
def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    inputs = {
        "ml_dataset": ART_DIR / "ml_dataset.parquet",
        "daily_equity_v2": ART_DIR / "daily_equity_v2.parquet",
        "daily_returns_s1": UNLOCK_DIR / "daily_returns_s1.parquet",
        "daily_returns_s2": UNLOCK_DIR / "daily_returns_s2.parquet",
    }
    checksums = {k: md5(v) for k, v in inputs.items()}

    # ---- frozen stream + frozen overlay weights (NO re-derivation) ---------
    df = make_xy(pd.read_parquet(inputs["ml_dataset"]))
    dev = df[df["ts"] < HOLDOUT_START].reset_index(drop=True)
    hold = df[df["ts"] >= HOLDOUT_START].reset_index(drop=True)
    ref = dev[dev["exit_known"] < HOLDOUT_TRAIN_CUT]     # frozen calibration
    keep_all, w_all = frozen_overlay_weights(df, ref)    # w = min(2, 2*pctl)
    keep_h, w_h = frozen_overlay_weights(hold, ref)
    assert keep_all.all() and keep_h.all()               # R3 keeps everything

    policies = {
        "a_capped_w1_NEW_DEFAULT": (np.minimum(w_all, 1.0), np.minimum(w_h, 1.0)),
        "b_uncapped_w2_OLD": (w_all, w_h),
        "c_M0_equal_slot": (np.ones(len(df)), np.ones(len(hold))),
    }
    weight_stats = {
        "mean_w_raw": round(float(w_all.mean()), 4),
        "share_events_w_gt_1": round(float((w_all > 1.0).mean()), 4),
        "mean_w_capped": round(float(np.minimum(w_all, 1.0).mean()), 4),
        "note": "raw w = min(2, 2*pctl(-btc_ret_6h)) vs frozen tr_hold ref",
    }

    # ---- task 1: policy sims ------------------------------------------------
    task1, curves, tds = {}, {}, {}
    stored = pd.read_parquet(inputs["daily_equity_v2"])
    for name, (wf, wh) in policies.items():
        daily, td = sim_daily_gross(df, keep_all, wf)
        # cross-check vs the imported (original) sim_daily — must be identical
        ref_daily, _ = sim_daily(df, keep_all, wf)
        assert np.allclose(daily["equity"], ref_daily["equity"], atol=0), name
        assert (daily["n_open"].to_numpy() == ref_daily["n_open"].to_numpy()).all()
        curves[name], tds[name] = daily, td
        hslice = daily[daily["date"] >= HOLDOUT_START.tz_localize(None)]
        mon = monthly_returns(daily)
        task1[name] = {
            "event_level_full": run_portfolio(df, keep_all, wf),
            "event_level_validation": run_portfolio(hold, keep_h, wh),
            "daily_full": daily_metrics(daily),
            "daily_validation_slice": daily_metrics(hslice),
            "gross_exposure": gross_diag(daily),
            "showcase_months": {m: round(float(mon.get(m, np.nan)), 4)
                                for m in SHOWCASE},
        }
    # uncapped curve must reproduce the stored artifact bit-for-bit
    b = curves["b_uncapped_w2_OLD"]
    repro = {
        "rows_match": bool(len(b) == len(stored)),
        "max_abs_equity_diff": float(np.max(np.abs(
            b["equity"].to_numpy() - stored["equity"].to_numpy()))),
        "n_open_match": bool((b["n_open"].to_numpy()
                              == stored["n_open"].to_numpy()).all()),
    }
    assert repro["rows_match"] and repro["max_abs_equity_diff"] < 1e-9, repro

    capped = curves["a_capped_w1_NEW_DEFAULT"]
    capped.to_parquet(OUT_CAPPED, index=False)

    # ---- task 2: kill-switch replay (descriptive) ---------------------------
    task2 = {"convention": ("kill at close of breach day (that day's loss is "
                            "taken); OFF earns 0; resume resets trailing peak "
                            "to current policy equity; DESCRIPTIVE, no verdict"),
             "rows": []}
    for curve_name, daily in (("capped", capped), ("uncapped", b)):
        for lvl in KILL_LEVELS:
            for rname, rdays in RESTARTS:
                row = kill_replay(daily, lvl, rdays)
                row = {"curve": curve_name, "restart": rname, **row}
                task2["rows"].append(row)

    # ---- task 3: concentration addendum (capped curve) ----------------------
    mon = monthly_returns(capped)
    r = pd.Series(capped["ret"].to_numpy(),
                  index=pd.DatetimeIndex(capped["date"]))
    dates = pd.DatetimeIndex(capped["date"])
    years_full = max((dates[-1] - dates[0]).days, 1) / 365.25
    top5 = mon.sort_values(ascending=False).head(5)
    lomo = {}
    for m in top5.index:
        rr = r.copy()
        rr[rr.index.strftime("%Y-%m") == m] = 0.0
        eqf = float(np.prod(1 + rr.to_numpy()))
        lomo[m] = {"month_ret": round(float(top5[m]), 4),
                   "cagr_without": round(eqf ** (1 / years_full) - 1, 4)}
    base_cagr = daily_metrics(capped)["cagr"]
    logret = np.log1p(capped["ret"].to_numpy())
    total_log = float(logret.sum())
    order = np.argsort(logret)[::-1]
    topn_share = {f"top{n}": round(float(logret[order[:n]].sum()) / total_log, 4)
                  for n in (1, 3, 5, 10)}
    topdays = [{"date": str(dates[j].date()), "ret": round(float(r.iloc[j]), 4)}
               for j in order[:5]]
    m_sorted = mon.sort_values()
    task3 = {
        "monthly_returns": {k: round(float(v), 4) for k, v in mon.items()},
        "full_cagr_capped": base_cagr,
        "leave_one_month_out_top5": lomo,
        "topN_day_share_of_total_logret": topn_share,
        "top5_days": topdays,
        "monthly_mean": round(float(mon.mean()), 4),
        "trimmed_monthly_mean_drop_best_worst":
            round(float(m_sorted.iloc[1:-1].mean()), 4),
        "by_year": {},
    }
    for y, g in capped.groupby(pd.DatetimeIndex(capped["date"]).year):
        task3["by_year"][str(y)] = {
            **daily_metrics(g),
            "active_days": int((g["n_open"] > 0).sum())}

    # ---- task 4: active-day correlations ------------------------------------
    def load_ret(path):
        d = pd.read_parquet(path)
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
        s = d.set_index("date")["ret"].astype(float).sort_index()
        return s[~s.index.duplicated(keep="first")]

    liq = pd.Series(capped["ret"].to_numpy(), index=dates.normalize())
    grid = pd.date_range(liq.index.min(), liq.index.max(), freq="D")
    task4 = {"note": ("naive corr = glue convention (liqrev span, missing "
                      "sleeve days = 0); active = BOTH series non-zero that "
                      "day — separates 'no overlap' from 'true independence'")}
    for sleeve, path in (("s2_long", inputs["daily_returns_s2"]),
                         ("s1_short", inputs["daily_returns_s1"])):
        o = load_ret(path).reindex(grid).fillna(0.0)
        lq = liq.reindex(grid).fillna(0.0)
        both = (lq != 0) & (o != 0)
        task4[f"liqrev_capped_x_{sleeve}"] = {
            "naive_full_series_corr": round(float(lq.corr(o)), 4),
            "n_days_full": int(len(grid)),
            "n_days_liq_active": int((lq != 0).sum()),
            "n_days_sleeve_active": int((o != 0).sum()),
            "n_days_both_active": int(both.sum()),
            "active_day_corr": round(float(lq[both].corr(o[both])), 4)
            if both.sum() >= 10 else None,
        }

    # ---- task 5: PnL attribution by weight bucket (uncapped policy b) -------
    # per-taken-trade log contribution: portfolio equity is the product of
    # per-trade factors (1 + w*r/15), so log-contributions sum exactly.
    EPS = 1e-6
    tb = tds["b_uncapped_w2_OLD"].copy()
    tb["logf"] = np.log(tb["factor"].to_numpy())
    tb["bucket"] = np.select(
        [tb["w"] < 1.0 - EPS, tb["w"] > 1.0 + EPS], ["w_lt_1", "w_gt_1"],
        default="w_eq_1")
    tb["month"] = tb["entry"].dt.strftime("%Y-%m")

    def attribution(sub: pd.DataFrame) -> dict:
        tot = float(sub["logf"].sum())
        res = {"total_logret": round(tot, 4), "n_trades": int(len(sub))}
        for bkt in ("w_lt_1", "w_eq_1", "w_gt_1"):
            s = sub[sub["bucket"] == bkt]
            ls = float(s["logf"].sum())
            res[bkt] = {"n": int(len(s)), "sum_logret": round(ls, 4),
                        "share_of_total": round(ls / tot, 4) if tot != 0 else None}
        return res

    task5 = {
        "policy": "b_uncapped_w2_OLD (raw frozen overlay weights)",
        "epsilon_band": EPS,
        "note": ("share_of_total = bucket sum of per-trade log(1 + w*r/15) / "
                 "window total; answers DOWN-weighting-idio vs UP-weighting-"
                 "leverage question (mean weight alone does not)"),
        "full_span": attribution(tb),
        "validation_2025plus": attribution(
            tb[tb["entry"] >= HOLDOUT_START]),
        "month_2022_11": attribution(tb[tb["month"] == "2022-11"]),
        "month_2025_02": attribution(tb[tb["month"] == "2025-02"]),
    }

    # ---- assemble ------------------------------------------------------------
    out = {
        "framing": FRAMING,
        "terminology": (">=2025 window = VALIDATION (validation/selection "
                        "set; consumed by cross-study strategy selection, "
                        "audit defect 3). NOT a clean holdout; the true test "
                        "is the post-freeze forward period."),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python research/scanner_lab/risk_policy_pack.py",
        "input_md5": checksums,
        "frozen_spec": {"overlay": "M1_dumb x R3_rankw, ref = DEV filled with "
                        "exit_known < 2024-12-25 (unchanged)",
                        "slots": SLOTS, "validation_start": str(HOLDOUT_START),
                        "n_events_filled": int(len(df)),
                        "uncapped_curve_reproduction": repro},
        "weight_stats": weight_stats,
        "task1_capped_overlay_cost": task1,
        "task2_kill_switch_replay_DESCRIPTIVE": task2,
        "task3_concentration_capped": task3,
        "task4_active_day_correlation": task4,
        "task5_pnl_attribution_by_weight_bucket": task5,
        "artifacts": {"results": str(OUT_JSON), "capped_daily": str(OUT_CAPPED)},
    }
    OUT_JSON.write_text(json.dumps(py(out), indent=1), encoding="utf-8")
    print("wrote", OUT_JSON)
    print("wrote", OUT_CAPPED, "rows", len(capped))
    print(json.dumps(py(out), indent=1))


if __name__ == "__main__":
    main()
