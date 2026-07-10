"""Q24 — Do on-chain network-activity factors add cross-sectional predictive power
beyond price/volume, on our tradable Binance USDT-perp universe?

FROZEN SPEC 2026-07-10 (written after Stage-0 data feasibility, BEFORE any signal or
return computation; no edits after first run except bug fixes that do not change the
declared design).

STAGE-0 RESULT (data feasibility, drives the declared substitutions):
  Source: CoinMetrics Community API (keyless, community-api.coinmetrics.io/v4); github
  CSV mirror was ~6 weeks stale so API used; data through 2026-07-08.
  Universe mapping: 145 local perp pairs -> 141 name-map to CM assets (unmapped: IOTA,
  MET, SKY, TAO — CM ids differ/absent), but only 25 assets expose network metrics
  (AdrActCnt/TxCnt/TxTfrCnt) in the community set: aave ada algo bch bnb btc comp crv
  dash doge dot etc eth icp ldo link ltc mana snx trx uni xlm xrp xtz zec.
  FeeTotNtv: only 14 (ada algo bch btc dash doge etc eth icp ltc xlm xrp xtz zec).
  CapMrktCurUSD: 24 (trx missing). TfrValUSD and FeeTotUSD do NOT exist in the
  community set -> declared substitutions below. VERDICT: DATA-LIMITED (25 < 40
  symbols); claims shrunk accordingly.

PANEL:
  Weekly Monday 00:00 UTC formations, 2020-02-03 .. last Monday with a complete
  forward week. Prices/volumes: research/data/binance_um/klines_1m/{SYM}USDT.parquet
  aggregated to UTC daily bars (close = last 1m close of day, $vol = sum quote_volume).
  Formation uses daily data through Sunday (m-1). Gate at formation: 30d median daily
  $vol >= $1M and >= 35 prior daily bars. Fwd 1w gross return = dayclose(m+6) /
  dayclose(m-1) - 1 (Sunday close to Sunday close). DEV: formation < 2025-01-01;
  LIVE: >= 2025-01-01.

ON-CHAIN PUBLICATION LAG: 2 days (CM community daily metrics publish T+1..T+2; for a
  Monday formation we use CM data through Friday). Exchange price/volume: no lag.

FEATURES (sm7(x,d) = trailing 7d mean ending d; chg28(x,d) = ln(sm7(x,d)/sm7(x,d-28));
  d_lag = formation - 2d):
  a) act_adr  = chg28(AdrActCnt, d_lag)
  b) act_tx   = chg28(TxCnt, d_lag)
  c) tfr_vs_vol = chg28(TxTfrCnt, d_lag) - chg28(binance $vol, m-1)
     [declared substitute for TfrValUSD/$vol ratio-change: TfrValUSD not in community set]
  d) fee_chg  = chg28(FeeTotUSD, d_lag), FeeTotUSD = FeeTotNtv * PriceUSD  [14 assets]
  e) nvt_inv  = -( sm7(CapMrktCurUSD,d_lag) / sm7(AdrActCnt,d_lag) )  level, inverted
     (cap per active address, low = cheap; declared substitute for Cap/TfrValUSD NVT)
  f) composite = mean of available cross-sectional pct-ranks of a-e (need >= 3)
  CONTROLS (price/volume only): mom4w = ln(close(m-1)/close(m-29));
  dvol4w = chg28(binance $vol, m-1).

TEST (per feature): weekly cross-section = gated symbols with finite feature; skip
  week if n < 6. Terciles by feature rank (top = highest). Metrics:
  - IC: weekly Spearman(feature, fwd gross ret); mean, Newey-West t (Bartlett, L=4).
  - Top-tercile net alpha: net_top_t = EW gross top ret - 20bps * turnover_t
    (turnover = fraction of top names replaced vs prior week; first week = 1;
    10bps maker-ish per side). alpha_t = net_top_t - EW universe gross ret.
    Report mean bps/wk with NW t; gross variant too.
  - Monotonicity: mean gross ret by tercile.
  - Beyond-price clause: per week OLS-residualize feature pct-rank on [1, mom4w
    pct-rank, dvol4w pct-rank]; residual IC = Spearman(resid, fwd ret);
    retention = mean resid IC / mean raw IC (NA if |mean raw IC| < 0.005).

BAR (frozen): DEV INTERESTING iff |meanIC| >= 0.05 AND NW-t >= 2 AND top-tercile net
  alpha >= +25 bps/wk AND retention >= 60%. DEV-best single feature (max |meanIC|
  among INTERESTING) + composite go to LIVE. LIVE PASS iff net alpha > 0 AND same-sign
  IC AND net alpha >= 40% of DEV net alpha. Everything else = NO. If nothing is DEV-
  INTERESTING, LIVE numbers are reported for the best-by-|IC| feature + composite as
  DESCRIPTIVE only, verdict NO.

DESCRIPTIVE EXTRAS (no verdict): gross top-minus-bottom tercile spread (shorting alts
  pays funding; 2025-26 avg funding negative — reference only); by-year mean IC; top-
  tercile symbol concentration for DEV-best + composite.

HONESTY: current-vintage CM data (community tier has no point-in-time vintages;
  restatements would flatter the backtest). 25-symbol cross-section -> terciles of
  ~5-8 names; ~250 DEV / ~78 LIVE weeks -> wide CIs. 6 declared features, NW-t>=2 is
  lenient for 6 tries -> marginal passes are fragile. Weekly returns are Sunday-close
  aligned from perp 1m closes, no funding PnL included (long-only cost model).

Outputs: research/data/onchain/results_onchain.json (+ printed final JSON).
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
KL = os.path.join(ROOT, "binance_um", "klines_1m")
OC = os.path.join(ROOT, "onchain")
ASSETS = ['aave', 'ada', 'algo', 'bch', 'bnb', 'btc', 'comp', 'crv', 'dash', 'doge',
          'dot', 'etc', 'eth', 'icp', 'ldo', 'link', 'ltc', 'mana', 'snx', 'trx',
          'uni', 'xlm', 'xrp', 'xtz', 'zec']
FEATS = ["act_adr", "act_tx", "tfr_vs_vol", "fee_chg", "nvt_inv", "composite"]
LAG_DAYS = 2
COST_BPS = 10.0          # per side
GATE_DVOL = 1_000_000.0
MIN_XS = 6
DEV_END = pd.Timestamp("2025-01-01")


def load_daily(asset):
    """Daily UTC bars from 1m klines: close=last, dvol=sum(quote_volume)."""
    f = os.path.join(KL, asset.upper() + "USDT.parquet")
    df = pd.read_parquet(f, columns=["open_time", "close", "quote_volume"])
    d = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.floor("D").dt.tz_localize(None)
    g = pd.DataFrame({"date": d,
                      "close": pd.to_numeric(df["close"], errors="coerce"),
                      "qv": pd.to_numeric(df["quote_volume"], errors="coerce")})
    agg = g.groupby("date").agg(close=("close", "last"), dvol=("qv", "sum"))
    return agg


def load_onchain(asset):
    df = pd.read_csv(os.path.join(OC, asset + ".csv"), parse_dates=["time"])
    df = df.set_index("time").apply(pd.to_numeric, errors="coerce")
    if "FeeTotNtv" in df.columns and "PriceUSD" in df.columns:
        df["FeeTotUSD"] = df["FeeTotNtv"] * df["PriceUSD"]
    return df


def sm7(s):
    return s.rolling(7, min_periods=5).mean()


def chg28(sm):
    return np.log(sm / sm.shift(28))


def nw_mean_t(x, L=4):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan, np.nan, n
    m = x.mean()
    e = x - m
    s = e @ e / n
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        s += 2 * w * (e[:-l] @ e[l:]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return m, m / se, n


def spearman(a, b):
    a = pd.Series(a).rank()
    b = pd.Series(b).rank()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    # ---------- build per-symbol daily merged frames ----------
    daily = {}
    for a in ASSETS:
        px = load_daily(a)
        oc = load_onchain(a)
        f = pd.DataFrame(index=px.index.union(oc.index))
        f["close"], f["dvol"] = px["close"], px["dvol"]
        for c in ["AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotUSD", "CapMrktCurUSD"]:
            f[c] = oc[c] if c in oc.columns else np.nan
        # smoothed series
        f["sm_adr"], f["sm_tx"], f["sm_tfr"] = sm7(f["AdrActCnt"]), sm7(f["TxCnt"]), sm7(f["TxTfrCnt"])
        f["sm_fee"], f["sm_cap"] = sm7(f["FeeTotUSD"]), sm7(f["CapMrktCurUSD"])
        f["sm_dvol"] = sm7(f["dvol"])
        f["c_adr"], f["c_tx"], f["c_tfr"], f["c_fee"] = (chg28(f["sm_adr"]), chg28(f["sm_tx"]),
                                                         chg28(f["sm_tfr"]), chg28(f["sm_fee"]))
        f["c_dvol"] = chg28(f["sm_dvol"])
        f["nvt"] = f["sm_cap"] / f["sm_adr"]
        f["gate_dvol30"] = f["dvol"].rolling(30, min_periods=30).median()
        f["mom4w"] = np.log(f["close"] / f["close"].shift(28))
        daily[a] = f

    # ---------- weekly panel ----------
    start = pd.Timestamp("2020-02-03")
    last_data = min(max(d.index[d["close"].notna()][-1] for d in daily.values()),
                    max(d.index[d["AdrActCnt"].notna()][-1] for d in daily.values()))
    mondays = pd.date_range(start, last_data - pd.Timedelta(days=6), freq="W-MON")
    rows = []
    for m in mondays:
        s_prev, s_fwd, d_lag = m - pd.Timedelta(days=1), m + pd.Timedelta(days=6), m - pd.Timedelta(days=LAG_DAYS)
        for a, f in daily.items():
            if s_prev not in f.index or s_fwd not in f.index or d_lag not in f.index:
                continue
            c0, c1 = f.at[s_prev, "close"], f.at[s_fwd, "close"]
            gate = f.at[s_prev, "gate_dvol30"]
            nbars = f.loc[:s_prev, "close"].notna().sum()
            if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(gate)
                    and gate >= GATE_DVOL and nbars >= 35):
                continue
            rows.append({
                "m": m, "sym": a, "ret": c1 / c0 - 1.0,
                "act_adr": f.at[d_lag, "c_adr"], "act_tx": f.at[d_lag, "c_tx"],
                "tfr_vs_vol": f.at[d_lag, "c_tfr"] - f.at[s_prev, "c_dvol"],
                "fee_chg": f.at[d_lag, "c_fee"],
                "nvt_inv": -f.at[d_lag, "nvt"],
                "mom4w": f.at[s_prev, "mom4w"], "dvol4w": f.at[s_prev, "c_dvol"],
            })
    panel = pd.DataFrame(rows)
    # composite = mean of available pct-ranks of a-e (>=3 needed), per week
    base_feats = FEATS[:5]
    def comp_fn(g):
        rk = pd.DataFrame({c: g[c].rank(pct=True) for c in base_feats})
        out = rk.mean(axis=1)
        out[rk.notna().sum(axis=1) < 3] = np.nan
        return out
    panel["composite"] = panel.groupby("m", group_keys=False).apply(comp_fn, include_groups=False)

    # ---------- per-feature weekly stats ----------
    def run_feature(feat, sub):
        ics, rics, alphas_net, alphas_gr, ls, terc_sums = [], [], [], [], [], np.zeros((3, 2))
        by_year, weeks = {}, []
        prev_top = None
        top_hits, top_contrib = {}, {}
        for m, g in sub.groupby("m"):
            g = g.dropna(subset=[feat, "ret"])
            if len(g) < MIN_XS:
                continue
            ic = spearman(g[feat], g["ret"])
            # residualization on controls
            gc = g.dropna(subset=["mom4w", "dvol4w"])
            ric = np.nan
            if len(gc) >= MIN_XS:
                y = gc[feat].rank(pct=True).values
                X = np.column_stack([np.ones(len(gc)), gc["mom4w"].rank(pct=True).values,
                                     gc["dvol4w"].rank(pct=True).values])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                ric = spearman(y - X @ beta, gc["ret"].values)
            r = g.sort_values(feat)
            k = len(g) // 3
            bot, top = r.iloc[:k], r.iloc[-k:]
            mid = r.iloc[k:-k]
            uni_ret = g["ret"].mean()
            top_ret, bot_ret = top["ret"].mean(), bot["ret"].mean()
            tset = set(top["sym"])
            turn = 1.0 if prev_top is None else (len(tset - prev_top) / max(len(tset), 1))
            prev_top = tset
            net_top = top_ret - 2 * COST_BPS / 1e4 * turn
            ics.append(ic); rics.append(ric)
            alphas_net.append(net_top - uni_ret); alphas_gr.append(top_ret - uni_ret)
            ls.append(top_ret - bot_ret)
            terc_sums += np.array([[bot_ret, 1], [mid["ret"].mean() if len(mid) else np.nan, 1 if len(mid) else 0], [top_ret, 1]])
            by_year.setdefault(m.year, []).append(ic)
            weeks.append(m)
            for s in tset:
                top_hits[s] = top_hits.get(s, 0) + 1
                top_contrib[s] = top_contrib.get(s, 0.0) + float(g.set_index("sym").at[s, "ret"]) / max(k, 1)
        if not weeks:
            return None
        mic, tic, nic = nw_mean_t(ics)
        mric = np.nanmean(rics) if np.isfinite(np.nanmean(rics)) else np.nan
        retention = float(mric / mic) if (np.isfinite(mic) and abs(mic) >= 0.005 and np.isfinite(mric)) else None
        an, tan, _ = nw_mean_t(alphas_net)
        ag, tag, _ = nw_mean_t(alphas_gr)
        lsm, lst, _ = nw_mean_t(ls)
        terc = [float(terc_sums[i, 0] / terc_sums[i, 1]) * 1e4 if terc_sums[i, 1] else None for i in range(3)]
        nw_ = len(weeks)
        conc = sorted(((s, top_hits[s] / nw_, top_contrib[s]) for s in top_hits),
                      key=lambda x: -x[2])[:8]
        return {"n_weeks": nw_, "mean_IC": round(float(mic), 4), "IC_nw_t": round(float(tic), 2),
                "resid_IC": None if not np.isfinite(mric) else round(float(mric), 4),
                "retention": None if retention is None else round(retention, 3),
                "top_alpha_net_bps": round(float(an) * 1e4, 1), "alpha_net_nw_t": round(float(tan), 2),
                "top_alpha_gross_bps": round(float(ag) * 1e4, 1),
                "ls_gross_bps": round(float(lsm) * 1e4, 1), "ls_nw_t": round(float(lst), 2),
                "terciles_gross_bps": [None if t is None else round(t, 1) for t in terc],
                "by_year_IC": {int(y): round(float(np.nanmean(v)), 3) for y, v in sorted(by_year.items())},
                "top_concentration": [(s, round(h, 2), round(c * 1e4, 0)) for s, h, c in conc]}

    dev = panel[panel["m"] < DEV_END]
    live = panel[panel["m"] >= DEV_END]
    res = {"stage0": {
               "source": "CoinMetrics Community API v4 (keyless), data through 2026-07-08, current vintage (no PIT)",
               "pairs_local": 145, "name_mapped": 141, "with_network_metrics": len(ASSETS),
               "assets": ASSETS, "fee_assets": 14, "verdict": "DATA-LIMITED (25 symbols < 40 bar)"},
           "panel": {"weeks_total": int(panel["m"].nunique()),
                     "span": [str(panel["m"].min().date()), str(panel["m"].max().date())],
                     "dev_weeks": int(dev["m"].nunique()), "live_weeks": int(live["m"].nunique()),
                     "median_xs_size": int(panel.groupby("m")["sym"].count().median())},
           "dev": {}, "live": {}, "controls_dev": {}}
    for feat in FEATS:
        r = run_feature(feat, dev)
        if r:
            r["INTERESTING"] = bool(abs(r["mean_IC"]) >= 0.05 and r["IC_nw_t"] >= 2
                                    and r["top_alpha_net_bps"] >= 25
                                    and (r["retention"] or 0) >= 0.6)
            res["dev"][feat] = r
    for c in ["mom4w", "dvol4w"]:
        r = run_feature(c, dev)
        if r:
            res["controls_dev"][c] = {k: r[k] for k in ["n_weeks", "mean_IC", "IC_nw_t", "top_alpha_net_bps"]}

    interesting = [f for f in FEATS if res["dev"].get(f, {}).get("INTERESTING")]
    best = (max(interesting, key=lambda f: abs(res["dev"][f]["mean_IC"])) if interesting
            else max((f for f in FEATS[:5] if f in res["dev"]), key=lambda f: abs(res["dev"][f]["mean_IC"])))
    res["dev_best"] = {"feature": best, "interesting_set": interesting,
                       "selection": "bar-passing" if interesting else "descriptive best-by-|IC| (nothing passed DEV bar)"}
    for feat in {best, "composite"}:
        r = run_feature(feat, live)
        if r:
            d = res["dev"].get(feat, {})
            same_sign = np.sign(r["mean_IC"]) == np.sign(d.get("mean_IC", 0))
            r["PASS"] = bool(interesting and feat in ([best] + ["composite"]) and r["top_alpha_net_bps"] > 0
                             and same_sign and d and r["top_alpha_net_bps"] >= 0.4 * d.get("top_alpha_net_bps", 1e9))
            res["live"][feat] = r
    res["verdict"] = ("PASS" if any(v.get("PASS") for v in res["live"].values())
                      else ("NO (nothing DEV-interesting)" if not interesting else "NO (failed LIVE)"))
    res["caveats"] = [
        "current-vintage CM community data; no point-in-time vintages published for community tier — restatements would flatter results",
        "25-symbol cross-section (DATA-LIMITED), terciles of ~5-8 names; wide CIs",
        "TfrValUSD/FeeTotUSD unavailable in community set; declared substitutes TxTfrCnt, FeeTotNtv*PriceUSD, Cap/AdrActCnt used",
        "fee_chg covers only 14 assets; nvt_inv 24 (trx no mktcap)",
        "6 features declared; NW-t>=2 lenient for 6 tries — marginal passes fragile",
        "big symbols missing from on-chain coverage: SOL, AVAX, NEAR, ATOM, APT, SUI, TON, PEPE, SHIB, WIF, etc.",
    ]
    out = os.path.join(OC, "results_onchain.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
