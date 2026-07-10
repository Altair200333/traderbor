"""
Q21 -- Crypto Variance Risk Premium (VRP) on BTC/ETH. PRE-REGISTERED ONE-SHOT STUDY.

Question #21 in the research ledger. Protocol: this docstring is the frozen spec.
It is written BEFORE the first run; the script is then run once and everything is
reported honestly (no peeking-then-tuning). Failures are reported as failures.

QUESTION
--------
Does the crypto variance risk premium (implied vol > subsequent realized vol) exist
on BTC/ETH; how large is it by year 2021-2026; and did it compress after US spot-ETF
options launched (Nov 2024) -- i.e. is it still alive in 2025-26?

DATA (all on disk)
------------------
- research/data/options/dvol_{btc,eth}_1h.parquet : Deribit DVOL index, hourly OHLC,
  cols [ts(ms epoch, open of bar), open, high, low, close]. UNITS: annualized 30d
  implied vol in PERCENTAGE POINTS (56 = 56% ann.). Daily close = close of the last
  hourly bar of each UTC day (bar ts = day 23:00, close ~= day 24:00).
- research/data/binance_um/klines_1m/{BTCUSDT,ETHUSDT}.parquet : Binance USD-M PERP
  1m klines, cols [open_time(ms), open, high, low, close, ...], 2020-01-01..~2026-07-07.
  This is the realized-vol source. CAVEAT: realized vol is measured on Binance perp
  prices while IV is Deribit-option-derived; BTC/ETH spot/perp price paths are ~identical
  across venues so RV is essentially venue-agnostic, but a small venue/basis mismatch
  exists and is acknowledged (not corrected).

PRE-REGISTERED DESIGN
---------------------
Timekeeping: everything UTC. For entry UTC day t, T_t = day-t daily-close instant = (t+1) 00:00 UTC.

1. IV_t = DVOL daily close / 100 (annualized, fractional). RV_t (fwd 30d) from hourly
   log returns: build hourly close by resampling 1m close with label='right',closed='right'
   (hourly stamp = close instant). Return r_h = ln(c_h/c_{h-1}). Forward window = returns
   with timestamp in (T_t, T_t+30d]. RV_t = sqrt( sum(r^2) * (365*24 / n_hours) ). Require
   n_hours >= 648 (90% of the 720 expected) else RV_t = NaN and day t is dropped (this is
   how the data-end truncation of the forward window is handled -- reported, not silently cut).
   Robustness RV_daily_t: daily close-to-close log returns over the SAME (t, t+30d] window,
   sqrt( sum(r^2) * (365/n_days) ), require n_days>=27.
2. Core series (per currency, per day t):
      VRP_t   = IV_t - RV_t                       (vol-point spread)
      VarP_t  = IV_t^2 - RV_t^2                    (variance premium)
      LR_t    = ln( RV_t^2 / IV_t^2 )             (log variance ratio; <0 means IV>RV)
      pnl_t   = (IV_t^2 - RV_t^2) / (2*IV_t)       (per-vega 30d short-variance-swap proxy)
3. Report per currency: full-sample mean/median VRP and LR; BY YEAR (2021..2026) mean VRP,
   mean IV, mean RV, share of days VRP>0, n; monthly NON-OVERLAPPING table (one obs per
   calendar month = VRP of that month's first UTC day) as the low-autocorrelation view.
4. Statistics: overlapping 30d windows are heavily autocorrelated. CI on mean VRP via
   MOVING-BLOCK BOOTSTRAP on the daily VRP series (block length L=45 days, 2000 resamples,
   seed=21): draw ceil(N/L) blocks of length L with random start in [0,N-L], concat, truncate
   to N, take mean; report 90% (5,95) and 95% (2.5,97.5) percentile CIs for full/DEV/LIVE.
   Also the non-overlapping monthly-sample mean and its plain t-stat = mean/(std/sqrt(n)).
   Reported effective-n caveat: overlapping n_days ~ n_days/30 independent 30d blocks.
5. DEV/LIVE split (lab convention): DEV = days with t < 2025-01-01; LIVE = t >= 2025-01-01.
   PRE-REGISTERED READ RULE (on VRP vol-point spread, hourly-RV definition):
     ALIVE                    if LIVE mean VRP > 0 AND LIVE 90% block-bootstrap CI excludes 0
                                 AND LIVE mean VRP >= 50% of DEV mean VRP.
     COMPRESSED-BUT-POSITIVE  if LIVE 90% CI excludes 0 but LIVE level < 50% of DEV level.
     DEAD                     otherwise.
6. Blowup-day table (descriptive) for 2021-05-19, 2022-11-08, 2022-11-09, 2022-11-10,
   2024-08-05, 2025-10-10: DVOL close at t-1d vs max(DVOL close at t, t+1d); the fwd-30d
   RV a short SEATED 30d before experienced = RV over window (T_t-30d, T_t] (i.e. holding
   INTO the blowup); and the WORST single per-vega short-vol proxy pnl over entry days in
   [t-45d, t+45d] (the entry that got maximally hurt). Prices "the day insurance pays out".
7. Regime/context (descriptive, FULL sample): corr(VRP_t, IV_t) [do sellers get paid more
   when vol is high?]; mean VRP conditional on IV terciles (labeled purely descriptive).
8. Q21b (descriptive event study, NO verdict): does BTC DVOL spike after liquidation
   cascades? Use canonical artifact research/data/liqrev/events.parquet (per-symbol events).
   Aggregate to event-DAYS = UTC dates with >=1 event; storm-DAYS = UTC dates with >=3 events.
   Table: mean BTC DVOL (pct pts, daily close) at day offsets -1,0,+1,+2,+3,+7 relative to
   the event day, vs the unconditional mean DVOL over the full DVOL sample; same for storms.
9. Outputs: research/data/options/results_vrp.json (all tables); research/data/options/
   vrp_daily.parquet (date, currency, iv, rv30_fwd, vrp, pnl_proxy [+varp, lr]). Print full JSON.

HONESTY NOTES (emitted in JSON meta): kline venue vs Deribit mismatch; overlapping-window
effective-n; DVOL is a model-derived index (not tradeable P&L) so pnl_t is a variance-swap
approximation -- real short-option P&L differs (path dependence, margin, discrete strikes,
skew); actual joint VRP span is reported and year tables reflect it (no silent truncation).
"""
import json
import numpy as np
import pandas as pd

BASE = r"F:\projects\traderbor\research\data"
OPT = BASE + r"\options"
KL = BASE + r"\binance_um\klines_1m"
SEED = 21
BLOCK = 45
NBOOT = 2000
EXP_HOURS = 720  # 30d * 24h
MIN_HOURS = 648  # 90%

rng = np.random.default_rng(SEED)


def load_dvol(path):
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
    s = pd.Series(df["close"].values, index=ts).sort_index()
    # daily close = close of last hourly bar of each UTC day
    daily = s.groupby(s.index.normalize()).last()
    daily.index = daily.index.tz_convert("UTC")
    return daily  # pct points, indexed by UTC midnight of day t


def load_hourly_close(path):
    df = pd.read_parquet(path)
    ct = pd.to_datetime(df["open_time"] + 60_000, unit="ms", utc=True)  # bar close instant
    c = pd.Series(df["close"].values, index=ct).sort_index()
    hourly = c.resample("1h", label="right", closed="right").last().dropna()
    return hourly


def load_daily_close(path):
    df = pd.read_parquet(path)
    ct = pd.to_datetime(df["open_time"] + 60_000, unit="ms", utc=True)
    c = pd.Series(df["close"].values, index=ct).sort_index()
    daily = c.resample("1D", label="right", closed="right").last().dropna()
    return daily


def fwd_rv_series(entry_days, hourly, horizon_days=30):
    """RV over (T_t, T_t+30d], T_t = day t 24:00 = (t+1) 00:00. entry_days: DatetimeIndex of day-midnights."""
    lret = np.log(hourly.values[1:] / hourly.values[:-1])
    r2 = lret ** 2
    rt = hourly.index.values[1:]  # return timestamp = close instant of the hour
    cum = np.concatenate([[0.0], np.cumsum(r2)])
    T = (entry_days + pd.Timedelta(days=1)).values  # (t+1) 00:00 = T_t
    hi = T + np.timedelta64(horizon_days, "D")
    i_lo = np.searchsorted(rt, T, side="right")
    i_hi = np.searchsorted(rt, hi, side="right")
    n = i_hi - i_lo
    ssq = cum[i_hi] - cum[i_lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.sqrt(ssq * (365.0 * 24.0 / n))
    rv = np.where(n >= MIN_HOURS, rv, np.nan)
    return rv, n


def fwd_rv_daily(entry_days, dclose, horizon_days=30):
    lret = np.log(dclose.values[1:] / dclose.values[:-1])
    r2 = lret ** 2
    rt = dclose.index.values[1:]
    cum = np.concatenate([[0.0], np.cumsum(r2)])
    T = (entry_days + pd.Timedelta(days=1)).values
    hi = T + np.timedelta64(horizon_days, "D")
    i_lo = np.searchsorted(rt, T, side="right")
    i_hi = np.searchsorted(rt, hi, side="right")
    n = i_hi - i_lo
    ssq = cum[i_hi] - cum[i_lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.sqrt(ssq * (365.0 / n))
    rv = np.where(n >= 27, rv, np.nan)
    return rv


def rv_window(hourly, lo, hi):
    """RV over (lo, hi] hourly."""
    lret = np.log(hourly.values[1:] / hourly.values[:-1])
    r2 = lret ** 2
    rt = hourly.index.values[1:]
    m = (rt > np.datetime64(lo)) & (rt <= np.datetime64(hi))
    n = int(m.sum())
    if n < 24:
        return float("nan"), n
    return float(np.sqrt(r2[m].sum() * (365.0 * 24.0 / n))), n


def block_bootstrap_mean_ci(x, block=BLOCK, nboot=NBOOT):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    N = len(x)
    if N < block + 5:
        return None
    nblocks = int(np.ceil(N / block))
    starts_max = N - block
    means = np.empty(nboot)
    for b in range(nboot):
        starts = rng.integers(0, starts_max + 1, size=nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:N]
        means[b] = x[idx].mean()
    return {
        "mean": float(x.mean()),
        "ci90": [float(np.percentile(means, 5)), float(np.percentile(means, 95))],
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_days": int(N),
        "eff_n_approx": round(N / 30.0, 1),
    }


def build_currency(cur, dvol_path, kline_path):
    iv_daily_pct = load_dvol(dvol_path)              # pct pts, indexed UTC midnight
    hourly = load_hourly_close(kline_path)
    dclose = load_daily_close(kline_path)

    days = iv_daily_pct.index  # candidate entry days (every DVOL day)
    iv = iv_daily_pct.values / 100.0
    rv, nh = fwd_rv_series(days, hourly)
    rv_d = fwd_rv_daily(days, dclose)

    df = pd.DataFrame({
        "date": days,
        "iv": iv,
        "rv30_fwd": rv,
        "rv30_fwd_daily": rv_d,
        "n_hours": nh,
    })
    df["vrp"] = df["iv"] - df["rv30_fwd"]
    df["varp"] = df["iv"] ** 2 - df["rv30_fwd"] ** 2
    df["lr"] = np.log((df["rv30_fwd"] ** 2) / (df["iv"] ** 2))
    df["pnl_proxy"] = (df["iv"] ** 2 - df["rv30_fwd"] ** 2) / (2.0 * df["iv"])
    df = df.dropna(subset=["vrp"]).reset_index(drop=True)
    df["currency"] = cur
    return df, iv_daily_pct, hourly


def summarize(df, cur):
    out = {}
    out["span"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    out["n_days"] = int(len(df))
    out["full_sample"] = {
        "mean_vrp": float(df["vrp"].mean()),
        "median_vrp": float(df["vrp"].median()),
        "mean_lr": float(df["lr"].mean()),
        "median_lr": float(df["lr"].median()),
        "mean_iv": float(df["iv"].mean()),
        "mean_rv": float(df["rv30_fwd"].mean()),
        "share_vrp_pos": float((df["vrp"] > 0).mean()),
        "mean_vrp_dailyRV": float((df["iv"] - df["rv30_fwd_daily"]).mean()),
    }
    yr = df["date"].dt.year
    byyear = {}
    for y in range(2021, 2027):
        sub = df[yr == y]
        if len(sub) == 0:
            continue
        byyear[str(y)] = {
            "mean_vrp": float(sub["vrp"].mean()),
            "mean_iv": float(sub["iv"].mean()),
            "mean_rv": float(sub["rv30_fwd"].mean()),
            "share_vrp_pos": float((sub["vrp"] > 0).mean()),
            "n": int(len(sub)),
        }
    out["by_year"] = byyear

    # monthly non-overlapping: first UTC day of each month
    m = df.set_index("date").copy()
    m["ym"] = m.index.to_period("M")
    first = m.groupby("ym").first()
    mv = first["vrp"].values
    n = len(mv)
    tmean = float(np.mean(mv))
    tstd = float(np.std(mv, ddof=1))
    tstat = tmean / (tstd / np.sqrt(n)) if tstd > 0 else float("nan")
    out["monthly_nonoverlap"] = {
        "mean_vrp": tmean, "std": tstd, "t_stat": float(tstat), "n_months": int(n),
        "share_pos": float((mv > 0).mean()),
    }

    # bootstrap full/DEV/LIVE
    dev = df[df["date"] < pd.Timestamp("2025-01-01", tz="UTC")]
    live = df[df["date"] >= pd.Timestamp("2025-01-01", tz="UTC")]
    bfull = block_bootstrap_mean_ci(df["vrp"].values)
    bdev = block_bootstrap_mean_ci(dev["vrp"].values)
    blive = block_bootstrap_mean_ci(live["vrp"].values)
    out["bootstrap"] = {"full": bfull, "dev": bdev, "live": blive}

    # verdict
    dev_mean = bdev["mean"]
    live_mean = blive["mean"]
    live_ci90 = blive["ci90"]
    ci_excl_0 = live_ci90[0] > 0 or live_ci90[1] < 0
    ratio = live_mean / dev_mean if dev_mean != 0 else float("nan")
    if live_mean > 0 and ci_excl_0 and live_mean >= 0.5 * dev_mean:
        verdict = "ALIVE"
    elif ci_excl_0 and live_mean < 0.5 * dev_mean and live_mean > 0:
        verdict = "COMPRESSED-BUT-POSITIVE"
    else:
        verdict = "DEAD"
    out["verdict"] = {
        "label": verdict, "dev_mean_vrp": float(dev_mean), "live_mean_vrp": float(live_mean),
        "live_ci90": live_ci90, "live_over_dev_ratio": float(ratio),
        "live_ci90_excludes_0": bool(ci_excl_0),
    }

    # regime / terciles (full sample descriptive)
    corr = float(np.corrcoef(df["vrp"].values, df["iv"].values)[0, 1])
    q = df["iv"].quantile([1/3, 2/3]).values
    lo = df[df["iv"] <= q[0]]["vrp"].mean()
    mid = df[(df["iv"] > q[0]) & (df["iv"] <= q[1])]["vrp"].mean()
    hi = df[df["iv"] > q[1]]["vrp"].mean()
    out["regime_descriptive"] = {
        "corr_vrp_iv": corr,
        "iv_tercile_thresholds_frac": [float(q[0]), float(q[1])],
        "mean_vrp_low_iv": float(lo), "mean_vrp_mid_iv": float(mid), "mean_vrp_high_iv": float(hi),
    }
    return out


def blowup_table(df, iv_daily_pct, hourly):
    days = ["2021-05-19", "2022-11-08", "2022-11-09", "2022-11-10", "2024-08-05", "2025-10-10"]
    rows = {}
    dpct = iv_daily_pct  # pct pts by day midnight
    dfi = df.set_index(df["date"].dt.normalize())
    for d in days:
        t = pd.Timestamp(d, tz="UTC")
        tm1 = t - pd.Timedelta(days=1)
        tp1 = t + pd.Timedelta(days=1)
        dvol_pre = float(dpct.get(tm1, np.nan))
        after_vals = [dpct.get(t, np.nan), dpct.get(tp1, np.nan)]
        dvol_post_max = float(np.nanmax(after_vals)) if not all(np.isnan(after_vals)) else float("nan")
        # RV a short seated 30d before experienced: RV over (T_t-30d, T_t], T_t=(t+1)00:00
        Tt = t + pd.Timedelta(days=1)
        rv_into, n_into = rv_window(hourly, Tt - pd.Timedelta(days=30), Tt)
        # worst per-vega pnl over entry days [t-45, t+45]
        lo = (t - pd.Timedelta(days=45)).normalize()
        hh = (t + pd.Timedelta(days=45)).normalize()
        win = dfi[(dfi.index >= lo) & (dfi.index <= hh)]
        if len(win):
            worst = win.loc[win["pnl_proxy"].idxmin()]
            worst_pnl = float(worst["pnl_proxy"])
            worst_date = str(pd.Timestamp(worst["date"]).date())
        else:
            worst_pnl, worst_date = float("nan"), None
        rows[d] = {
            "dvol_close_tm1": dvol_pre,
            "dvol_close_max_t_tp1": dvol_post_max,
            "jump": (dvol_post_max - dvol_pre) if not np.isnan(dvol_pre) else float("nan"),
            "rv30_into_blowup_ann": rv_into,
            "worst_pervega_pnl_pm45d": worst_pnl,
            "worst_pnl_entry_date": worst_date,
        }
    return rows


def q21b_cascade(iv_daily_pct_btc):
    ev = pd.read_parquet(BASE + r"\liqrev\events.parquet")
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    ev["day"] = ev["ts"].dt.normalize()
    per_day = ev.groupby("day").size()
    event_days = per_day.index
    storm_days = per_day[per_day >= 3].index
    dvol = iv_daily_pct_btc  # pct pts by day midnight
    uncond = float(dvol.mean())
    offsets = [-1, 0, 1, 2, 3, 7]

    def offset_table(day_index):
        tbl = {}
        for k in offsets:
            vals = []
            for d in day_index:
                v = dvol.get(d + pd.Timedelta(days=k), np.nan)
                if not np.isnan(v):
                    vals.append(v)
            tbl[str(k)] = {"mean_dvol": float(np.mean(vals)) if vals else float("nan"), "n": len(vals)}
        return tbl

    return {
        "events_span": [str(ev["ts"].min().date()), str(ev["ts"].max().date())],
        "n_events": int(len(ev)),
        "n_event_days": int(len(event_days)),
        "n_storm_days": int(len(storm_days)),
        "unconditional_mean_btc_dvol_pct": uncond,
        "event_day_dvol_by_offset": offset_table(event_days),
        "storm_day_dvol_by_offset": offset_table(storm_days),
        "note": "BTC DVOL levels in percentage points (annualized 30d IV); offsets in UTC days relative to the event/storm day.",
    }


def main():
    results = {"question": "Q21 crypto variance risk premium BTC/ETH 2021-2026", "meta": {}}
    results["meta"]["kline_source"] = "research/data/binance_um/klines_1m/{BTCUSDT,ETHUSDT}.parquet (Binance USD-M perpetual, 1m)"
    results["meta"]["iv_source"] = "research/data/options/dvol_{btc,eth}_1h.parquet (Deribit DVOL index, hourly, pct pts)"
    results["meta"]["caveats"] = [
        "Venue mismatch: RV on Binance perp vs Deribit-option IV; price paths ~identical across venues, small basis mismatch acknowledged, not corrected.",
        "Overlapping 30d windows: daily VRP is heavily autocorrelated; effective independent n ~ n_days/30. Block bootstrap (L=45d) and non-overlapping monthly t-stat address this.",
        "DVOL is a model-derived index (from the option book), NOT tradeable P&L. pnl_proxy is a variance-swap approximation; real short-option P&L differs (path dependence, margin, discrete strikes, skew).",
        "Forward-30d RV requires a full window; days whose forward window is <648h (90% of 720) are dropped, so VRP span ends ~30d before kline end. Actual span reported below.",
    ]

    dfs = []
    btc_iv_daily = None
    for cur, dv, kl in [("BTC", OPT + r"\dvol_btc_1h.parquet", KL + r"\BTCUSDT.parquet"),
                        ("ETH", OPT + r"\dvol_eth_1h.parquet", KL + r"\ETHUSDT.parquet")]:
        df, iv_daily_pct, hourly = build_currency(cur, dv, kl)
        if cur == "BTC":
            btc_iv_daily = iv_daily_pct
        res = summarize(df, cur)
        res["blowup_days"] = blowup_table(df, iv_daily_pct, hourly)
        results[cur] = res
        dfs.append(df)

    results["Q21b_cascade_dvol"] = q21b_cascade(btc_iv_daily)

    allf = pd.concat(dfs, ignore_index=True)
    results["meta"]["joint_span"] = [str(allf["date"].min().date()), str(allf["date"].max().date())]

    # write vrp_daily.parquet (spec cols + varp, lr extras)
    outp = allf[["date", "currency", "iv", "rv30_fwd", "vrp", "pnl_proxy", "varp", "lr"]].copy()
    outp["date"] = outp["date"].dt.tz_convert("UTC")
    outp.to_parquet(OPT + r"\vrp_daily.parquet", index=False)

    with open(OPT + r"\results_vrp.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nWROTE:", OPT + r"\results_vrp.json")
    print("WROTE:", OPT + r"\vrp_daily.parquet", "rows=", len(outp))


if __name__ == "__main__":
    main()
