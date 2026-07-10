"""Q28 — can a REAL-TIME universe-selection rule reproduce curated-149 liqrev?

Decisive follow-up to Q23 (liqrev_pit_validation.py / results_pit.json): classic
survivorship was acquitted (15 delisted symbols, 119 events, +1.12%/ev in line
with survivors), but the full $1M-gated PIT universe FAILED (4,926 extra events,
LIVE -0.55%/ev at 25bps, PIT portfolio LIVE -28.6% vs survivors +39.9%): the
gated universe exploded 90 -> 472 names via mass junk listings, and the curated
149-pair list carries real, possibly hindsight-tinged selection alpha.

QUESTION: does any DECLARED, POINT-IN-TIME weekly universe rule (no future
data) reproduce the curated-149 performance?  If yes -> the strategy claim is
solid and the bot's universe becomes a validated refreshable rule.  If no ->
the curated backtest must be marked down to the best rule's numbers for
deployment honesty.

================================ FROZEN SPEC ==================================
(written before the first analysis run; nothing below changes afterwards)

HONESTY, UP FRONT: these rules were chosen AFTER seeing Q23's failure mechanism
(junk = young, low-volume-rank listings).  This is one round of informed
selection on the same data.  A marginal pass is therefore suspect; the real
confirmation is the bot's forward performance.

STUDY WINDOW: triggers 2022-07-01 <= ts < 2026-07-06 UTC.
DEV = ts < 2025-01-01, LIVE = ts >= 2025-01-01 (repo protocol).

EVENT SET (byte-identical to Q23; NO re-tuning):
  - survivors-149: liqrev_v2.detect_events VERBATIM (spot 1h close/OI/liq gate),
    parity target 1308 events vs ml_dataset.parquet (|diff| <= 26 required).
  - PIT-extra 625 symbols: liqrev_pit_validation.detect_pit VERBATIM
    (imported, not copied), expected 4,926 events (drift vs results_pit.json
    reported; > +-2% flags the study unreliable).
  - Detection keeps its own hourly $1M/30d-median liq gate; the rules below
    only FILTER events by weekly universe membership.

WEEKLY POINT-IN-TIME MEMBERSHIP: evaluated at each Monday 00:00 UTC from
  2022-06-27 to 2026-06-29 using ONLY data from days strictly before the
  Monday; membership applies to events with trigger ts in [Mon, next Mon).
  An event is kept iff its symbol is a member for its week.
  CONVENTION DELTA (declared): weekly snap vs research's static universe and
  vs detection's hourly gate; a symbol can trigger while gated-in intraweek
  yet be a non-member at the Monday snap (and vice versa for rule c).

VOLUME PANEL: daily dollar volume = daily sum of 1h quote_volume.
  Survivors: SPOT klines (research/data/v3/klines/1h — research convention);
  PIT-extra: FUTURES klines (research/data/binance_um/pit_extra/klines_1h) —
  the same declared, unavoidable cross-source deviation as Q23.  Ranking mixes
  the two sources; direction of bias unquantified — declared.
  Trailing 90d median at Monday M = rolling(90d calendar, min 30 daily obs)
  .median() valued at day M-1.  Rule-c $1M gate at M = rolling(30d, min 30)
  .median() at day M-1 > 1e6 (Q23 gate math, weekly-snapped).
  PIT-extra kline history is clipped at 2022-05 (Q23 download window): in the
  earliest weeks some old listings have < 90d of observable volume (min 30
  obs applies); declared.  8 pre-window-dead candidates have no data and are
  absent from the ranking pool (they produce no events in-window anyway).

AGE: days since first kline.  PIT-extra: first day of the TRUE first S3
  monthly-zip month (klines_manifest.json first_month — not clipped).
  Survivors: first local SPOT kline ts (futures listing dates not held
  locally; spot history can predate the perp listing -> the age filter is
  generous to survivors; declared).

CANDIDATE RULES (5) + REFERENCES (2):
  top100 / top150 / top200 : TOP-N by trailing 90d median daily dollar volume
                             (rank over ALL panel symbols with a valid value;
                             ties broken by symbol name, deterministic).
  top150_age180            : TOP-150 AND age >= 180d.
  age180_gate1m            : age >= 180d AND weekly-snapped $1M 30d gate
                             (Q23's gate + age only).
  REF survivors149         : hindsight curated list (all survivor events).
  REF pit_full             : Q23's full $1M-gate PIT (all events; known FAIL).

TRADES (frozen deploy convention, = Q23 task 5): maker limit at trigger-bar
  close, TTL 1 bar (filled iff next-bar low < limit); disaster stop -20% from
  entry, gap-aware; else exit at close of bar i+24 (~+24h); 10bps RT for the
  verdict cells and portfolio; 25bps per-event LIVE mean reported as
  sensitivity.  1h-sim caveat (declared): research's 1m-path validation
  retained 93.8% of the 1h edge (results_1m.json / audit 4.3a); cited, not
  re-run.

PORTFOLIO (= Q23 task 6 machinery): filled events in ts order, 15 slots x
  1/15 equity, slot busy 24h, overlay ON (w = min(2, 2*P) from frozen
  bot/artifacts/liqrev_overlay.json scores vs -btc_ret_6h at trigger; missing
  BTC bar -> w = 1, counted), 10bps.  Windows: full span, DEV (ts < 2025-01-01),
  LIVE (ts >= 2025-01-01).

REPORTED PER RULE: kept events + events/year + by-year counts; fill rate;
  per-event net mean/median/win/stop pooled + DEV + LIVE (10bps) with
  day-clustered SE and t on the LIVE mean (clusters = UTC event days,
  se = sqrt(sum_c S_c^2)/n, S_c = within-day sum of demeaned rets);
  25bps LIVE mean; portfolio full/DEV/LIVE (total, CAGR, maxDD, by-year);
  overlap vs curated-149 (share of the 1308 curated events captured; share of
  kept events that are on curated names); membership sizes.

PRE-REGISTERED VERDICT RULE: a candidate rule VALIDATES iff ALL of
   (i)  full-span portfolio CAGR >= 0.60 * survivors149 full-span CAGR,
   (ii) LIVE portfolio total return > 0,
   (iii) LIVE per-event net mean (10bps, filled) > 0.
  Best validating rule = highest LIVE CAGR.  If NONE validates: the honest
  markdown = the best candidate rule's (by LIVE CAGR) numbers become the
  deployment-honest expectation; said plainly in the output.

MECHANISM TABLE (descriptive, best rule — validating or not): among pit_full
  filled events NOT kept by the rule: age-at-event and Monday volume-rank
  distributions (quartiles, share unranked/rank>150, share age<180d), LIVE
  net mean excluded vs kept; and for top150_age180 the exclusion decomposition
  (age-only / rank-only / both) to attribute which filter does the work.

Artifacts: research/data/liqrev/results_pit_rule.json (all tables), printed.
Usage:  python liqrev_pit_rule.py
================================================================================
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import load_universe  # noqa: E402
from liqrev_v2 import detect_events, KL_DIR  # noqa: E402
from liqrev_pit_validation import (  # noqa: E402  (byte-identical Q23 code)
    detect_pit, simulate_deploy, run_portfolio, load_overlay,
    btc_ret6_series, overlay_weight, load_pit_klines, _ev_stats,
    PIT_DIR, PIT_KL, ART_DIR, SPAN_START, SPAN_END, LIVE_START)

CAL_START = pd.Timestamp("2022-01-01", tz="UTC")
CAL_END = pd.Timestamp("2026-07-05", tz="UTC")
MONDAYS = pd.date_range("2022-06-27", "2026-06-29", freq="W-MON", tz="UTC")
YEARS_SPAN = (SPAN_END - SPAN_START).days / 365.25
AGE_DAYS = 180
RULE_NAMES = ["top100", "top150", "top200", "top150_age180", "age180_gate1m"]


# ------------------------------------------------------------ volume panel ---
def build_panel() -> tuple[pd.DataFrame, dict[str, pd.Timestamp], list[str]]:
    """Daily dollar-volume panel (calendar-aligned) + first-kline age refs."""
    survivors = [c.pair for c in load_universe()]
    man = json.loads((PIT_DIR / "klines_manifest.json").read_text(encoding="utf-8"))
    first_month = {r["symbol"]: r.get("first_month") for r in man["results"]
                   if r["status"] == "ok"}
    cal = pd.date_range(CAL_START, CAL_END, freq="D", tz="UTC")
    cols: dict[str, pd.Series] = {}
    first_ts: dict[str, pd.Timestamp] = {}
    for pair in survivors:
        p = KL_DIR / f"{pair}.parquet"
        if not p.exists():
            continue
        k = pd.read_parquet(p, columns=["open_time", "quote_volume"])
        idx = pd.to_datetime(k["open_time"], unit="ms", utc=True)
        s = pd.Series(k["quote_volume"].to_numpy(), index=idx).sort_index()
        first_ts[pair] = s.index[0]
        cols[pair] = s.resample("1D").sum().reindex(cal)
    for sym in sorted(p.stem for p in PIT_KL.glob("*.parquet")):
        k = load_pit_klines(sym)
        first_ts[sym] = pd.Timestamp(first_month[sym] + "-01", tz="UTC")
        cols[sym] = k["quote_volume"].resample("1D").sum().reindex(cal)
    panel = pd.DataFrame(cols, index=cal)
    return panel, first_ts, survivors


def build_memberships(panel: pd.DataFrame, first_ts: dict[str, pd.Timestamp]
                      ) -> tuple[dict[str, dict], dict[pd.Timestamp, dict]]:
    roll90 = panel.rolling(90, min_periods=30).median()
    roll30 = panel.rolling(30, min_periods=30).median()
    members: dict[str, dict] = {r: {} for r in RULE_NAMES}
    ranks_at: dict[pd.Timestamp, dict[str, int]] = {}
    for m in MONDAYS:
        day = m - pd.Timedelta("1D")
        v90 = roll90.loc[day].dropna()
        order = (pd.DataFrame({"v": v90})
                 .assign(sym=v90.index)
                 .sort_values(["v", "sym"], ascending=[False, True]))
        ranked = list(order["sym"])
        ranks_at[m] = {s: i + 1 for i, s in enumerate(ranked)}
        age_ok = {s for s in panel.columns if (m - first_ts[s]).days >= AGE_DAYS}
        gate1m = set(roll30.loc[day][roll30.loc[day] > 1e6].index)
        members["top100"][m] = set(ranked[:100])
        members["top150"][m] = set(ranked[:150])
        members["top200"][m] = set(ranked[:200])
        members["top150_age180"][m] = set(ranked[:150]) & age_ok
        members["age180_gate1m"][m] = gate1m & age_ok
    return members, ranks_at


def monday_of(ts: pd.Series) -> pd.Series:
    return ts.dt.normalize() - pd.to_timedelta(ts.dt.dayofweek, unit="D")


# ------------------------------------------------------------------- stats ---
def clustered_live(f: pd.DataFrame) -> dict:
    """Day-clustered SE/t of the LIVE net mean (clusters = UTC days)."""
    live = f[f["ts"] >= LIVE_START]
    if len(live) < 2:
        return {"n": int(len(live))}
    mu = float(live["ret"].mean())
    resid = live["ret"] - mu
    sc = resid.groupby(live["ts"].dt.floor("D")).sum()
    se = float(np.sqrt((sc ** 2).sum()) / len(live))
    return {"n": int(len(live)), "mean": round(mu, 4), "se_day_clustered":
            round(se, 4), "t": round(mu / se, 2) if se else None,
            "n_days": int(sc.size)}


def rule_report(keep: pd.Series, tr10: pd.DataFrame, tr25: pd.DataFrame,
                w: pd.Series, surv_keys: set) -> dict:
    ev = tr10[keep]
    f = ev[ev["filled"]]
    f25 = tr25[keep & tr25["filled"]]
    by_year = {str(y): int(n) for y, n in
               ev.groupby(ev["ts"].dt.year).size().items()}
    kept_keys = set(zip(ev["symbol"], ev["ts"]))
    cur_captured = len(kept_keys & surv_keys)
    on_curated = int((~ev["is_new"].astype(bool)).sum())
    dev = tr10[keep & (tr10["ts"] < LIVE_START)]
    return {
        "n_events": int(len(ev)),
        "events_per_year": round(len(ev) / YEARS_SPAN, 1),
        "by_year_events": by_year,
        "fill_rate": round(float(ev["filled"].mean()), 3) if len(ev) else None,
        "per_event_10bps": {
            "pooled": _ev_stats(f),
            "DEV": _ev_stats(f[f["ts"] < LIVE_START]),
            "LIVE": _ev_stats(f[f["ts"] >= LIVE_START]),
            "LIVE_day_clustered": clustered_live(f)},
        "per_event_25bps_LIVE_mean":
            round(float(f25[f25["ts"] >= LIVE_START]["ret"].mean()), 4)
            if len(f25[f25["ts"] >= LIVE_START]) else None,
        "portfolio_10bps_overlay": {
            "full": run_portfolio(ev, w),
            "DEV": run_portfolio(dev, w),
            "LIVE": run_portfolio(ev, w, LIVE_START)},
        "overlap_curated": {
            "curated_events_captured": cur_captured,
            "share_of_curated_1308": round(cur_captured / len(surv_keys), 3),
            "share_rule_events_on_curated_names":
                round(on_curated / len(ev), 3) if len(ev) else None}}


# -------------------------------------------------------------------- main ---
def main() -> None:
    print("building volume panel...", flush=True)
    panel, first_ts, survivors = build_panel()
    print(f"panel: {panel.shape[1]} symbols x {panel.shape[0]} days", flush=True)
    members, ranks_at = build_memberships(panel, first_ts)
    memb_stats = {r: {"avg_members": round(np.mean([len(members[r][m])
                                                    for m in MONDAYS]), 1),
                      "members_last_monday": len(members[r][MONDAYS[-1]]),
                      "avg_survivor_members":
                          round(np.mean([len(members[r][m] & set(survivors))
                                         for m in MONDAYS]), 1)}
                  for r in RULE_NAMES}

    # ---- detection (byte-identical Q23 code paths) ----------------------
    print("detecting survivors (parity)...", flush=True)
    surv_parts, kcache = [], {}
    for idx, pair in enumerate(survivors, 1):
        e = detect_events(pair)
        if len(e):
            e = e[(e["ts"] >= SPAN_START) & (e["ts"] < SPAN_END)]
        if len(e):
            e = e.copy()
            e["is_new"] = False
            surv_parts.append(e)
            k = pd.read_parquet(KL_DIR / f"{pair}.parquet",
                                columns=["open_time", "open", "high", "low",
                                         "close"])
            k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
            kcache[pair] = k.set_index("ts").sort_index()
        if idx % 40 == 0:
            print(f"  [{idx}/{len(survivors)}]", flush=True)
    surv_ev = pd.concat(surv_parts, ignore_index=True)
    canon = pd.read_parquet(ART_DIR / "ml_dataset.parquet",
                            columns=["symbol", "ts"])
    surv_keys = set(zip(surv_ev["symbol"], surv_ev["ts"]))
    parity = {"n_canonical": len(canon), "n_this_run": len(surv_keys),
              "n_match": len(set(zip(canon["symbol"], canon["ts"]))
                            & surv_keys)}
    parity_ok = abs(len(surv_keys) - 1308) <= 26

    print("detecting PIT-extra...", flush=True)
    pit_syms = sorted(p.stem for p in PIT_KL.glob("*.parquet"))
    new_parts = []
    for idx, sym in enumerate(pit_syms, 1):
        e = detect_pit(sym)
        if len(e):
            e["is_new"] = True
            new_parts.append(e)
            kcache[sym] = load_pit_klines(sym)[["open", "high", "low", "close"]]
        if idx % 100 == 0:
            print(f"  [{idx}/{len(pit_syms)}]", flush=True)
    new_ev = pd.concat(new_parts, ignore_index=True)
    q23_new_total = 4926
    new_drift = abs(len(new_ev) - q23_new_total) / q23_new_total

    all_ev = pd.concat([surv_ev, new_ev], ignore_index=True)
    print(f"events: survivors {len(surv_ev)}, new {len(new_ev)}", flush=True)

    # ---- trades + overlay weights (once) ---------------------------------
    print("simulating trades...", flush=True)
    tr10 = simulate_deploy(all_ev, kcache, 0.0010)
    tr25 = simulate_deploy(all_ev, kcache, 0.0025)
    scores, n_sc = load_overlay()
    btc6 = btc_ret6_series()
    w = pd.Series([overlay_weight(t, btc6, scores, n_sc) for t in tr10["ts"]],
                  index=tr10.index)
    n_w_fallback = int(sum(1 for t in tr10["ts"]
                           if not np.isfinite(btc6.get(t, np.nan))))

    # ---- per-rule keep masks + reports -----------------------------------
    mon = monday_of(tr10["ts"])
    reports: dict[str, dict] = {}
    keeps: dict[str, pd.Series] = {}
    for r in RULE_NAMES:
        keeps[r] = pd.Series([s in members[r][m] for s, m in
                              zip(tr10["symbol"], mon)], index=tr10.index)
    keeps["survivors149"] = ~tr10["is_new"].astype(bool)
    keeps["pit_full"] = pd.Series(True, index=tr10.index)
    for r in RULE_NAMES + ["survivors149", "pit_full"]:
        print(f"portfolio: {r}", flush=True)
        reports[r] = rule_report(keeps[r], tr10, tr25, w, surv_keys)

    # ---- verdict ----------------------------------------------------------
    base_cagr = reports["survivors149"]["portfolio_10bps_overlay"]["full"]["cagr"]
    verdicts = {}
    for r in RULE_NAMES:
        rep = reports[r]
        cagr = rep["portfolio_10bps_overlay"]["full"].get("cagr", 0.0)
        live_tot = rep["portfolio_10bps_overlay"]["LIVE"].get("total", None)
        live_mean = rep["per_event_10bps"]["LIVE"].get("net_mean", None)
        i_ok = cagr >= 0.60 * base_cagr
        ii_ok = live_tot is not None and live_tot > 0
        iii_ok = live_mean is not None and live_mean > 0
        verdicts[r] = {"i_full_cagr": cagr, "i_pass": bool(i_ok),
                       "ii_live_total": live_tot, "ii_pass": bool(ii_ok),
                       "iii_live_ev_mean": live_mean, "iii_pass": bool(iii_ok),
                       "VALIDATES": bool(i_ok and ii_ok and iii_ok)}
    validating = [r for r in RULE_NAMES if verdicts[r]["VALIDATES"]]

    def live_cagr(r: str) -> float:
        return reports[r]["portfolio_10bps_overlay"]["LIVE"].get("cagr", -9.9)

    best = (max(validating, key=live_cagr) if validating
            else max(RULE_NAMES, key=live_cagr))
    markdown = None
    if not validating:
        b = reports[best]
        markdown = {
            "statement": ("NO real-time rule validates. Deployment-honest "
                          "expectation = best candidate rule "
                          f"({best}) numbers below; the curated-149 backtest "
                          "CAGR is NOT a deployable claim."),
            "best_rule": best,
            "full_cagr": b["portfolio_10bps_overlay"]["full"].get("cagr"),
            "live_total": b["portfolio_10bps_overlay"]["LIVE"].get("total"),
            "live_cagr": b["portfolio_10bps_overlay"]["LIVE"].get("cagr"),
            "live_ev_mean_10bps": b["per_event_10bps"]["LIVE"].get("net_mean"),
            "vs_survivors149_full_cagr": base_cagr}

    # ---- mechanism table for the best rule --------------------------------
    fb = tr10[tr10["filled"]].copy()
    fb["kept"] = keeps[best][fb.index]
    fb["age_days"] = [(t - first_ts[s]).days
                      for s, t in zip(fb["symbol"], fb["ts"])]
    mon_f = monday_of(fb["ts"])
    fb["rank"] = [ranks_at[m].get(s, np.nan)
                  for s, m in zip(fb["symbol"], mon_f)]

    def dist(x: pd.Series) -> dict:
        x = x.dropna()
        if not len(x):
            return {"n": 0}
        q = x.quantile([0.25, 0.5, 0.75])
        return {"n": int(len(x)), "q25": round(float(q.iloc[0]), 1),
                "median": round(float(q.iloc[1]), 1),
                "q75": round(float(q.iloc[2]), 1)}

    exc, kep = fb[~fb["kept"]], fb[fb["kept"]]
    mech = {
        "best_rule": best,
        "excluded_vs_gate_universe": {
            "n_excluded_filled": int(len(exc)),
            "age_days_dist": dist(exc["age_days"]),
            "vol_rank_dist": dist(exc["rank"]),
            "share_unranked": round(float(exc["rank"].isna().mean()), 3),
            "share_rank_gt150":
                round(float((exc["rank"] > 150).mean()), 3),
            "share_age_lt180": round(float((exc["age_days"] < AGE_DAYS).mean()), 3),
            "LIVE_net_mean_excluded":
                round(float(exc[exc["ts"] >= LIVE_START]["ret"].mean()), 4)
                if len(exc[exc["ts"] >= LIVE_START]) else None},
        "kept": {
            "n_kept_filled": int(len(kep)),
            "age_days_dist": dist(kep["age_days"]),
            "vol_rank_dist": dist(kep["rank"]),
            "LIVE_net_mean_kept":
                round(float(kep[kep["ts"] >= LIVE_START]["ret"].mean()), 4)
                if len(kep[kep["ts"] >= LIVE_START]) else None}}
    # attribution on top150_age180 regardless of best (the composite rule)
    kc = keeps["top150_age180"][fb.index]
    exc_c = fb[~kc]
    age_fail = exc_c["age_days"] < AGE_DAYS
    rank_fail = exc_c["rank"].isna() | (exc_c["rank"] > 150)
    mech["top150_age180_exclusion_attribution"] = {
        "n_excluded_filled": int(len(exc_c)),
        "age_only": int((age_fail & ~rank_fail).sum()),
        "rank_only": int((~age_fail & rank_fail).sum()),
        "both": int((age_fail & rank_fail).sum()),
        "note": "membership snapped Mondays; rank/age recomputed identically"}

    out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "frozen in liqrev_pit_rule.py docstring (Q28)",
        "honesty": {
            "informed_selection": "rules chosen AFTER Q23 failure mechanism "
                                  "was known; one selection round; marginal "
                                  "passes are suspect",
            "weekly_snap_convention": "Monday-00:00 snap vs hourly gate in "
                                      "detection (declared delta)",
            "volume_source_mix": "survivors ranked on SPOT dollar volume, "
                                 "PIT-extra on FUTURES (Q23 deviation)",
            "survivor_age_source": "first local SPOT kline (predates perp "
                                   "listing for some names)",
            "pit_volume_history_clipped_at": "2022-05",
            "1h_sim_citation": "1m-path validation retained 93.8% of 1h edge "
                               "(results_1m.json / audit 4.3a)",
            "overlay_weight_fallbacks": n_w_fallback},
        "detector_parity": {**parity, "parity_ok": bool(parity_ok),
                            "new_events": int(len(new_ev)),
                            "q23_new_events": q23_new_total,
                            "new_drift_frac": round(new_drift, 4),
                            "new_ok": bool(new_drift <= 0.02)},
        "membership_stats": memb_stats,
        "rules": reports,
        "verdicts": verdicts,
        "validating_rules": validating,
        "best_rule": best,
        "honest_markdown": markdown,
        "mechanism": mech}
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "results_pit_rule.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    print(f"\nartifacts -> {ART_DIR / 'results_pit_rule.json'}")


if __name__ == "__main__":
    main()
