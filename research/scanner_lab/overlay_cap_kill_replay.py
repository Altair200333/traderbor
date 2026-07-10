"""CAPPED-OVERLAY RE-SIM + KILL-SWITCH POLICY REPLAY (external-audit blockers 1 & 3).

PRE-REGISTERED SPEC — frozen 2026-07-10 BEFORE the first run. Everything below
is declared up front; every cell is reported; nothing is hidden. This is NOT
signal research: the detector, the 1280 filled events, and the overlay mapping
(M1_dumb x R3_rankw, w = min(2, 2*pctl(-btc_ret_6h)) against the frozen
tr_hold reference = DEV filled events with exit_known < 2024-12-25) are FROZEN
artifacts of prior pre-registered studies (liqrev_ml_model.py). Here we only
measure RISK POLICIES on top of that frozen stream, per
docs/notes/2026-07-10/external-audit-response.md items 1 (gross cap) and
7/blocker 3 (kill-switch replay).

REUSE, NOT REINVENTION: events/weights/daily-curve machinery is imported from
liqrev_ml_model.py and liqrev_unlock_interaction.py (the exact code that
produced daily_equity_v2.parquet). The only new sim code generalizes the same
loop with (a) a total-deployed-weight cap and (b) per-trade weight records; it
is cross-checked: with the cap disabled it must reproduce run_portfolio's
stats AND sim_daily's daily equity bit-for-bit, and the FROZEN daily curve
must match the stored daily_equity_v2.parquet (max abs equity deviation
reported; parity claimed only if < 1e-9).

===================== ADDENDUM A — CAPPED OVERLAY RE-SIM =====================
Declared variants (weights applied to the SAME frozen event stream, 15 slots,
eq *= 1 + w*net_ret/15, slot frees at ts+24h):
  FROZEN          w = min(2, 2*pctl)          no cash constraint (reference).
  CAPPED_1_0      w = min(1, 2*pctl)          same mapping, clipped at 1.0;
                                              gross mathematically <= 1.0x.
  CASH_CONSTRAINT w = min(2, 2*pctl) per slot, but total deployed weight
                  across concurrently open slots hard-capped at 15.0
                  (= gross <= 1.0x equity). Excess intents clipped in trigger
                  order: new trade gets min(w, 15 - deployed); if the residual
                  is <= 0 the trade is NOT taken (no slot consumed).
  BASELINE        w = 1 all events (no overlay; read-rule reference only).
Reported per variant: event-level full-span and holdout (>=2025, fresh-state
run on holdout events — the convention of results_ml.json / audit-01, which
produced CAGR 32.5% frozen vs ~12.5% baseline) CAGR/maxDD; daily-curve
full-span metrics + holdout slice (secondary); by-year returns and
within-year maxDD (2022 = the systemic tail year, explicitly); mean deployed
weight over taken trades; worst single day (date+ret); for FROZEN only, the
share of HOURS with gross > 1.0x (exact interval sweep over trade
entry/exits, over the full span and over active hours); for CASH_CONSTRAINT,
clip diagnostics (n clipped, n skipped-at-zero, total weight clipped, max
gross — asserted <= 1.0 + 1e-9).

PRE-REGISTERED READ RULE (frozen before the run):
  Conservatism order (most -> least): CAPPED_1_0, CASH_CONSTRAINT, FROZEN.
  uplift_frozen = holdout_cagr(FROZEN) - holdout_cagr(BASELINE)   [event-level]
  A variant QUALIFIES iff
    holdout_cagr(variant) >= holdout_cagr(BASELINE) + 0.70 * uplift_frozen
    AND calendar-2022 daily-curve return(variant) >= calendar-2022
        return(FROZEN) - 1e-9.
  LIVE-DEFAULT = the most conservative qualifying variant (FROZEN qualifies
  trivially and is the fallback). Bot settings implied:
    CAPPED_1_0      -> BOT_OVERLAY_WEIGHT_CAP = 1.0; risk.py gross cap 1.0x
                       equity still added as invariant (belt and braces).
    CASH_CONSTRAINT -> BOT_OVERLAY_WEIGHT_CAP = 2.0 (frozen mapping) + risk.py
                       HARD gross cap: sum(open+pending notional) <= 1.0x
                       equity, new orders clipped to the residual, skipped
                       if residual <= 0.
    FROZEN          -> no cap change (would leave audit defect 1 open).

================= ADDENDUM B — KILL-SWITCH POLICY REPLAY ====================
DESCRIPTIVE ONLY (metrics declared upfront, NO verdict rule — feeds the
owner's kill-criteria freeze). Replayed through the daily curves of FROZEN
and of the ADDENDUM-A recommended variant (deduped if identical).
Policy grid: kill thresholds {-8%, -10%, -12%, -15%} trailing drawdown from
the policy-equity peak x restart rules {after 30 flat days, after 60 flat
days, never}.
Declared conventions:
  * Kill detected at the CLOSE of the breach day t (that day's loss is taken).
  * Day t+1 = exit day: policy return = -0.5% (1-day exit slippage of 0.5% of
    equity, applied unconditionally — conservative), strategy return NOT
    earned that day.
  * Days after: flat, 0%. Restart-N: trading resumes on the first day d with
    (d - t) >= N calendar days; the kill-engine trailing peak resets to the
    policy equity at resume (otherwise it re-kills instantly). "never" = flat
    to end of data.
  * Worst realized DD of a policy = global trailing drawdown of the POLICY
    equity curve (independent of the engine's reset peaks).
Reported per policy (4 x 3 x 2 curves = 24 rows): n kills 2022-2026, kill
dates, policy total return vs no-kill total return and cost in pp, worst
realized DD, share of days off, FTX callout (kills in 2022-10..12, policy vs
strategy return over Nov+Dec 2022, off on 2022-11-08?), famine callout (kills
in 2026, policy vs strategy 2026 return, dead-at-end-of-data flag), harvest
callouts (policy vs strategy return in 2024-08 and 2025-02; for never-restart
policies, PERMANENT-KILL flags: first kill before 2024-08-01 => misses both
harvest months; before 2025-02-01 => misses 2025-02).
Declared frontier highlight (descriptive, not a verdict): policies with
cost < 5pp of total return AND worst realized DD at least 2pp shallower than
the no-kill maxDD of the same curve.

Inputs : research/data/liqrev/ml_dataset.parquet
         research/data/liqrev/daily_equity_v2.parquet   (parity check)
Outputs: research/data/liqrev/results_overlay_cap.json
Usage  : python research/scanner_lab/overlay_cap_kill_replay.py
(md5 checksums of inputs + the exact command are recorded in the JSON,
per the adopted audit-response item 4 practice.)
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
from liqrev_ml_features import ART_DIR, HOLDOUT_START              # noqa: E402
from liqrev_ml_model import (                                      # noqa: E402
    make_xy, run_portfolio, HOLDOUT_TRAIN_CUT, SLOTS,
)
from liqrev_unlock_interaction import (                            # noqa: E402
    frozen_overlay_weights, sim_daily, py,
)

OUT_JSON = ART_DIR / "results_overlay_cap.json"
GROSS_CAP_W = float(SLOTS)          # total deployed weight 15 == 1.0x equity
KILL_LEVELS = (-0.08, -0.10, -0.12, -0.15)
RESTARTS = (("restart_30d", 30), ("restart_60d", 60), ("never", None))
SLIPPAGE = 0.005
UPLIFT_KEEP = 0.70
FRONTIER_COST_PP = 5.0
FRONTIER_DD_CUT = 0.02
CONSERVATISM_ORDER = ["CAPPED_1_0", "CASH_CONSTRAINT", "FROZEN"]
BOT_SETTINGS = {
    "CAPPED_1_0": ("BOT_OVERLAY_WEIGHT_CAP=1.0; risk.py gross cap 1.0x equity "
                   "kept as invariant (unreachable with w<=1, 15 slots)"),
    "CASH_CONSTRAINT": ("BOT_OVERLAY_WEIGHT_CAP=2.0 (frozen mapping) + risk.py "
                        "HARD gross cap sum(open+pending notional) <= 1.0x "
                        "equity, clip new orders to residual, skip at 0"),
    "FROZEN": "no cap (leaves audit defect 1 open — NOT acceptable live)",
}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# --------------------------------------------------------------- capped sim --
def sim_capped(ev: pd.DataFrame, keep: np.ndarray, w: np.ndarray,
               cap_w: float | None = None, slots: int = SLOTS):
    """run_portfolio / sim_daily machinery generalized with an optional cap on
    TOTAL deployed weight across open slots. With cap_w=None it must reproduce
    run_portfolio's stats and sim_daily's trade list exactly (asserted in
    main). Returns (event_stats_dict, trades_df, clip_stats_dict)."""
    order = np.argsort(ev["ts"].to_numpy())
    ts_arr = ev["ts"].to_numpy()[order]
    ex_arr = ev["exit_slot"].to_numpy()[order]
    r_arr = ev["net_ret"].to_numpy()[order]
    k_arr = keep[order]
    w_arr = w[order]
    eq, busy, curve = 1.0, [], []           # busy: list of (exit_ts, w_eff)
    kept_ret, trades = [], []
    n_taken = n_clip = n_skip0 = 0
    clip_w = 0.0
    for i in range(len(ev)):
        busy = [b for b in busy if b[0] > ts_arr[i]]
        if k_arr[i] and len(busy) < slots:
            wi = float(w_arr[i])
            if cap_w is not None:
                room = cap_w - sum(b[1] for b in busy)
                we = min(wi, max(room, 0.0))
            else:
                we = wi
            if cap_w is not None and we <= 1e-12:
                n_skip0 += 1                # fully clipped: no trade, no slot
                curve.append((ts_arr[i], eq))
                continue
            if we < wi - 1e-12:
                n_clip += 1
                clip_w += wi - we
            eq *= (1 + we * r_arr[i] / slots)
            busy.append((ex_arr[i], we))
            n_taken += 1
            kept_ret.append(r_arr[i])
            trades.append((ts_arr[i], ex_arr[i],
                           1.0 + we * r_arr[i] / slots, we))
        curve.append((ts_arr[i], eq))
    c = pd.Series({t: v for t, v in curve})
    years = max((c.index[-1] - c.index[0]).days, 1) / 365.25
    kr = np.array(kept_ret)
    n_kept = int(keep.sum())
    stats = {"n_events": int(len(ev)), "n_kept": n_kept,
             "kept_frac": round(n_kept / len(ev), 3),
             "n_taken": n_taken,
             "net_mean": round(float(kr.mean()), 5) if len(kr) else None,
             "win": round(float((kr > 0).mean()), 4) if len(kr) else None,
             "total": round(float(eq - 1), 4),
             "cagr": round(float(eq ** (1 / years) - 1), 4),
             "maxDD": round(float((c / c.cummax() - 1).min()), 4)}
    td = pd.DataFrame(trades, columns=["entry", "exit", "factor", "w"])
    td["entry"] = pd.to_datetime(td["entry"], utc=True)
    td["exit"] = pd.to_datetime(td["exit"], utc=True)
    clip = {"n_trades_clipped": n_clip, "n_trades_skipped_at_zero": n_skip0,
            "total_weight_clipped": round(clip_w, 4)}
    return stats, td, clip


def daily_from_trades(td: pd.DataFrame) -> pd.DataFrame:
    """EXACT copy of sim_daily's day-attribution (factor -> entry day)."""
    eday = td["entry"].dt.floor("1D")
    days = pd.date_range(eday.min(), td["exit"].max().floor("1D"),
                         freq="D", tz="UTC")
    dfac = td.groupby(eday)["factor"].prod().reindex(days).fillna(1.0)
    n_open = pd.Series(0, index=days)
    for e, x in zip(td["entry"], td["exit"]):
        d0, d1 = e.floor("1D"), x.floor("1D")
        rng_days = pd.date_range(d0, d1, freq="D", tz="UTC")
        if x == d1 and len(rng_days) > 1:
            rng_days = rng_days[:-1]
        n_open.loc[rng_days] += 1
    return pd.DataFrame({"date": days.tz_localize(None),
                         "ret": dfac.to_numpy() - 1.0,
                         "equity": np.cumprod(dfac.to_numpy()),
                         "n_open": n_open.to_numpy()})


# ------------------------------------------------------------ curve metrics --
def curve_stats(daily: pd.DataFrame) -> dict:
    r = daily["ret"].to_numpy()
    dates = pd.DatetimeIndex(daily["date"])
    eq = np.cumprod(1.0 + r)
    years = max((dates[-1] - dates[0]).days, 1) / 365.25
    j = int(np.argmin(r))
    return {"total": round(float(eq[-1] - 1), 4),
            "cagr": round(float(eq[-1] ** (1 / years) - 1), 4),
            "maxDD": round(float((eq / np.maximum.accumulate(eq) - 1).min()), 4),
            "worst_day": {"date": str(dates[j].date()),
                          "ret": round(float(r[j]), 4)},
            "n_days": int(len(r))}


def by_year(daily: pd.DataFrame) -> dict:
    out = {}
    for y, g in daily.groupby(pd.DatetimeIndex(daily["date"]).year):
        r = g["ret"].to_numpy()
        eq = np.cumprod(1.0 + r)
        out[str(y)] = {"ret": round(float(eq[-1] - 1), 4),
                       "maxDD_within": round(
                           float((eq / np.maximum.accumulate(eq) - 1).min()), 4)}
    return out


def gross_sweep(td: pd.DataFrame, slots: int = SLOTS) -> dict:
    """Exact interval sweep of deployed weight (gross = sum w_open / slots).
    Exits release BEFORE entries at the same timestamp (matches busy filter)."""
    pts = ([(t.value, -w) for t, w in zip(td["exit"], td["w"])]
           + [(t.value, +w) for t, w in zip(td["entry"], td["w"])])
    pts.sort()
    cur, prev_t = 0.0, None
    dur_over = dur_active = 0.0
    max_g, max_t = 0.0, None
    t0, t1 = pts[0][0], pts[-1][0]
    for t, dw in pts:
        if prev_t is not None and t > prev_t:
            dt_h = (t - prev_t) / 3.6e12
            if cur / slots > 1.0 + 1e-9:
                dur_over += dt_h
            if cur > 1e-12:
                dur_active += dt_h
        cur += dw
        if cur > max_g:
            max_g, max_t = cur, t
        prev_t = t
    span_h = (t1 - t0) / 3.6e12
    return {"share_hours_gross_gt_1_full_span":
            round(dur_over / span_h, 5) if span_h else None,
            "share_hours_gross_gt_1_of_active":
            round(dur_over / dur_active, 5) if dur_active else None,
            "hours_gross_gt_1": round(dur_over, 1),
            "active_hours": round(dur_active, 1),
            "max_gross": round(max_g / slots, 3),
            "max_gross_ts": str(pd.Timestamp(max_t, tz="UTC"))}


# ------------------------------------------------------------- kill replay ---
def kill_replay(daily: pd.DataFrame, level: float, restart_days) -> dict:
    dates = pd.DatetimeIndex(daily["date"])
    ret = daily["ret"].to_numpy()
    n = len(ret)
    pol = np.zeros(n)
    on_arr = np.zeros(n, bool)
    eq, peak, on = 1.0, 1.0, True
    pending_exit = False
    kill_date = None
    kills, off_days = [], 0
    for i in range(n):
        if not on:
            if pending_exit:                       # exit day: slippage only
                eq *= (1.0 - SLIPPAGE)
                pol[i] = -SLIPPAGE
                pending_exit = False
                off_days += 1
                continue
            if (restart_days is not None
                    and (dates[i] - kill_date).days >= restart_days):
                on = True
                peak = eq                          # reset trailing peak
            else:
                off_days += 1
                continue
        eq *= 1.0 + ret[i]
        pol[i] = ret[i]
        on_arr[i] = True
        peak = max(peak, eq)
        if eq / peak - 1.0 <= level + 1e-12:
            on = False
            pending_exit = True
            kill_date = dates[i]
            kills.append(str(dates[i].date()))
    pol_eq = np.cumprod(1.0 + pol)
    nokill_eq = np.cumprod(1.0 + ret)
    worst_dd = float((pol_eq / np.maximum.accumulate(pol_eq) - 1).min())

    def _window_ret(mask: np.ndarray, arr: np.ndarray) -> float | None:
        return (round(float(np.prod(1.0 + arr[mask]) - 1), 4)
                if mask.any() else None)

    ftx_m = (dates >= "2022-11-01") & (dates < "2023-01-01")
    y26 = dates.year == 2026
    h24 = (dates >= "2024-08-01") & (dates < "2024-09-01")
    h25 = (dates >= "2025-02-01") & (dates < "2025-03-01")
    ftx_day = dates == pd.Timestamp("2022-11-08")
    dead_at_end = (not on) or pending_exit
    first_kill = kills[0] if kills else None
    perm_flags = None
    if restart_days is None and kills:
        perm_flags = {
            "permanently_dead_since": first_kill,
            "misses_2024_08_harvest": first_kill < "2024-08-01",
            "misses_2025_02_harvest": first_kill < "2025-02-01"}
    return {"kill_level": level,
            "restart": "never" if restart_days is None
            else f"restart_{restart_days}d",
            "n_kills": len(kills), "kill_dates": kills,
            "policy_total": round(float(pol_eq[-1] - 1), 4),
            "nokill_total": round(float(nokill_eq[-1] - 1), 4),
            "cost_pp": round(float(nokill_eq[-1] - pol_eq[-1]) * 100, 1),
            "worst_realized_DD": round(worst_dd, 4),
            "off_days": off_days,
            "off_share_pct": round(100.0 * off_days / n, 1),
            "ftx": {"kills_2022_q4": [k for k in kills
                                      if "2022-10" <= k < "2023-01"],
                    "off_on_2022_11_08": bool(ftx_day.any()
                                              and not on_arr[ftx_day][0]),
                    "policy_ret_nov_dec_2022": _window_ret(ftx_m, pol),
                    "strategy_ret_nov_dec_2022": _window_ret(ftx_m, ret)},
            "famine_2026": {"kills_2026": [k for k in kills if k >= "2026-01"],
                            "policy_ret_2026": _window_ret(y26, pol),
                            "strategy_ret_2026": _window_ret(y26, ret),
                            "dead_at_end_of_data": dead_at_end},
            "harvest_2024_08": {"policy": _window_ret(h24, pol),
                                "strategy": _window_ret(h24, ret)},
            "harvest_2025_02": {"policy": _window_ret(h25, pol),
                                "strategy": _window_ret(h25, ret)},
            "permanent_kill_flags": perm_flags}


# ------------------------------------------------------------------- main ----
def main() -> None:
    inputs = {"ml_dataset": ART_DIR / "ml_dataset.parquet",
              "daily_equity_v2": ART_DIR / "daily_equity_v2.parquet"}
    checksums = {k: md5(v) for k, v in inputs.items()}

    df = make_xy(pd.read_parquet(inputs["ml_dataset"]))
    dev = df[df["ts"] < HOLDOUT_START].reset_index(drop=True)
    hold = df[df["ts"] >= HOLDOUT_START].reset_index(drop=True)
    ref = dev[dev["exit_known"] < HOLDOUT_TRAIN_CUT]        # frozen calibration
    keep_all, w_all = frozen_overlay_weights(df, ref)
    keep_h, w_h = frozen_overlay_weights(hold, ref)
    assert keep_all.all() and keep_h.all()
    ones_f, ones_h = np.ones(len(df)), np.ones(len(hold))

    variants = {                     # name -> (w_full, w_hold, cap)
        "FROZEN": (w_all, w_h, None),
        "CAPPED_1_0": (np.minimum(w_all, 1.0), np.minimum(w_h, 1.0), None),
        "CASH_CONSTRAINT": (w_all, w_h, GROSS_CAP_W),
        "BASELINE": (ones_f, ones_h, None),
    }

    # ---- Addendum A -----------------------------------------------------
    A, dailies, hold_cagr, y2022 = {}, {}, {}, {}
    stored = pd.read_parquet(inputs["daily_equity_v2"])
    for name, (wf, wh, cap) in variants.items():
        ev_full, td, clip = sim_capped(df, keep_all, wf, cap)
        ev_hold, _, _ = sim_capped(hold, keep_h, wh, cap)
        if cap is None:               # cross-check vs the original machinery
            assert ev_full == run_portfolio(df, keep_all, wf), name
            assert ev_hold == run_portfolio(hold, keep_h, wh), name
            ref_daily, ref_td = sim_daily(df, keep_all, wf)
            assert np.allclose(daily_from_trades(td)["equity"],
                               ref_daily["equity"], atol=0), name
        daily = daily_from_trades(td)
        dailies[name] = daily
        hslice = daily[daily["date"] >= HOLDOUT_START.tz_localize(None)]
        yr = by_year(daily)
        row = {"event_full": ev_full, "event_holdout": ev_hold,
               "daily_full": curve_stats(daily),
               "daily_holdout_slice": curve_stats(hslice),
               "by_year": yr,
               "mean_weight_deployed": round(float(td["w"].mean()), 4),
               "n_trades_taken": int(len(td))}
        if name == "FROZEN":
            row["gross_exposure_hours"] = gross_sweep(td)
        if cap is not None:
            sweep = gross_sweep(td)
            assert sweep["max_gross"] <= 1.0 + 1e-9, sweep
            row["clip_stats"] = {**clip, "max_gross": sweep["max_gross"]}
        A[name] = row
        hold_cagr[name] = ev_hold["cagr"]
        y2022[name] = yr.get("2022", {}).get("ret")

    parity = {"rows_match": bool(len(dailies["FROZEN"]) == len(stored)),
              "max_abs_equity_deviation": float(np.max(np.abs(
                  dailies["FROZEN"]["equity"].to_numpy()
                  - stored["equity"].to_numpy())))
              if len(dailies["FROZEN"]) == len(stored) else None,
              "n_open_match": bool(
                  (dailies["FROZEN"]["n_open"].to_numpy()
                   == stored["n_open"].to_numpy()).all())
              if len(dailies["FROZEN"]) == len(stored) else None}

    # ---- pre-registered read rule ---------------------------------------
    uplift = hold_cagr["FROZEN"] - hold_cagr["BASELINE"]
    thr = hold_cagr["BASELINE"] + UPLIFT_KEEP * uplift
    evals = {}
    winner = "FROZEN"
    for v in CONSERVATISM_ORDER:
        q_cagr = hold_cagr[v] >= thr - 1e-12
        q_2022 = (y2022[v] is not None and y2022["FROZEN"] is not None
                  and y2022[v] >= y2022["FROZEN"] - 1e-9)
        evals[v] = {"holdout_cagr": hold_cagr[v],
                    "cagr_threshold": round(thr, 4),
                    "passes_cagr": bool(q_cagr),
                    "ret_2022": y2022[v], "frozen_2022": y2022["FROZEN"],
                    "passes_2022": bool(q_2022),
                    "qualifies": bool(q_cagr and q_2022)}
    for v in CONSERVATISM_ORDER:
        if evals[v]["qualifies"]:
            winner = v
            break
    read_rule = {
        "declared": ("LIVE-DEFAULT = most conservative variant with holdout "
                     "cagr >= baseline + 0.70*uplift(FROZEN) AND 2022 return "
                     ">= FROZEN's; order CAPPED_1_0 > CASH_CONSTRAINT > "
                     "FROZEN (fallback)"),
        "uplift_frozen": round(uplift, 4),
        "per_variant": evals,
        "winner": winner,
        "bot_settings": BOT_SETTINGS[winner]}

    # ---- Addendum B ------------------------------------------------------
    b_curves = ["FROZEN"] + ([winner] if winner != "FROZEN" else [])
    B = {"convention": ("kill at close of breach day; next day = exit day "
                        "with -0.5% slippage and no strategy P&L; flat 0% "
                        "after; restart-N resumes >= N days after kill, peak "
                        "resets to policy equity; DESCRIPTIVE, no verdict"),
         "curves": {}}
    frontier = []
    for cn in b_curves:
        daily = dailies[cn]
        nokill = curve_stats(daily)
        rows = [kill_replay(daily, lvl, rd)
                for lvl in KILL_LEVELS for _, rd in RESTARTS]
        for r in rows:
            # positive == policy DD shallower than no-kill by that many pp
            r["dd_cut_pp"] = round(
                (r["worst_realized_DD"] - nokill["maxDD"]) * 100, 1)
            r["on_frontier"] = bool(
                r["cost_pp"] < FRONTIER_COST_PP
                and r["worst_realized_DD"] >= nokill["maxDD"] + FRONTIER_DD_CUT)
            if r["on_frontier"]:
                frontier.append({"curve": cn, "kill_level": r["kill_level"],
                                 "restart": r["restart"],
                                 "cost_pp": r["cost_pp"],
                                 "worst_realized_DD": r["worst_realized_DD"],
                                 "nokill_maxDD": nokill["maxDD"]})
        B["curves"][cn] = {"nokill": nokill, "policies": rows}
    B["frontier_cost_lt_5pp_and_DD_cut_ge_2pp"] = frontier

    out = {"framing": ("Risk-policy measurement on the FROZEN liqrev v2 "
                       "stream (audit blockers 1 & 3); spec + read rules "
                       "frozen in the module docstring BEFORE the first run"),
           "run_utc": datetime.now(timezone.utc).isoformat(),
           "command": "python research/scanner_lab/overlay_cap_kill_replay.py",
           "input_md5": checksums,
           "n_events_filled": int(len(df)),
           "n_dev": int(len(dev)), "n_holdout": int(len(hold)),
           "parity_vs_daily_equity_v2": parity,
           "addendum_A_variants": A,
           "addendum_A_read_rule": read_rule,
           "addendum_B_kill_replay": B}
    OUT_JSON.write_text(json.dumps(py(out), indent=1), encoding="utf-8")
    print("wrote", OUT_JSON)
    print(json.dumps(py(out), indent=1))


if __name__ == "__main__":
    main()
