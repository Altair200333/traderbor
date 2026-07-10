"""Liqrev ML — STAGE 1: zero-lookahead feature dataset + DEV-only EDA.

Scores liquidation-cascade-reversion events at TRIGGER-BAR CLOSE so stage 2 can
size/filter them better than equal slots. This module ONLY builds the labelled
feature dataset and runs exploratory analysis on the DEV split. It fits NO model
and never inspects the holdout.

HONESTY PROTOCOL
  DEV     = events with ts <  2025-01-01 (all EDA/statistics printed here).
  HOLDOUT = events with ts >= 2025-01-01 (features/labels stored, NEVER analysed).
Features + labels are computed for ALL events (stage 2 needs them); every EDA
number is asserted to come from DEV rows only.

LOOKAHEAD DISCIPLINE
  Trigger bar has open_time = ts and closes at the instant ts+1h. Everything
  known strictly before ts+1h is fair game (the trigger bar itself is complete —
  detection uses its close). All feature functions take RAW inputs and slice
  as-of via `searchsorted(cutoff='ts+1h', side='left')`, so the exact same code
  path is used by production and by the lookahead checker (checker feeds
  physically-truncated inputs and asserts identical values).

Label = frozen maker config: simulate(ev, kcache, "maker", "none", 0.0010, False)
  (post-only limit buy at trigger close, next-bar trade-through fill, NO stop,
   exit +24h close, 10bps round-trip). ~+1.66% net mean/event, ~98% fill.

Artifacts:
  research/data/liqrev/ml_dataset.parquet
  research/data/liqrev/eda_summary.json

Usage: python liqrev_ml_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT, load_universe  # noqa: E402
from liqrev_v2 import detect_events, simulate  # noqa: E402

# ---------------------------------------------------------------- paths / const
DATA = REPO_ROOT / "research" / "data"
SPOT_1H_DIR = DATA / "v3" / "klines" / "1h"
FUT_1M_DIR = DATA / "binance_um" / "klines_1m"
OI_5M_DIR = DATA / "perp" / "metrics_5m"
FUND_DIR = DATA / "perp" / "funding"
ART_DIR = DATA / "liqrev"

HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
BTC_PAIR = "BTCUSDT"
N_UNIVERSE = 149                      # mkt_share_6h denominator (main universe)

H1 = pd.Timedelta("1h")
# hourly-bar rolling windows (bars)
W_6H, W_24H, W_7D, W_30D, W_50D = 6, 24, 168, 720, 1200
# 5m positioning window: 30 days of 5-minute bars
W_OI_30D = 288 * 30

FEATURES = [
    # severity / shape
    "ret_6h", "ret_1h", "ret_24h", "dist_low_30d", "wick_frac",
    # OI
    "doi6", "doi24", "oi_turnover",
    # volume / flow
    "vol_spike", "taker_buy_share",
    # volatility
    "rv_7d", "vol_ratio", "atr24_norm",
    # funding
    "funding_last", "funding_trail3",
    # market context
    "btc_ret_6h", "btc_ret_24h", "btc_regime", "mkt_events_24h", "mkt_share_6h",
    # symbol history
    "sym_past_n", "sym_past_mean_ret",
    # microstructure (futures 1m)
    "basis_bps", "m1_maxdrop", "m1_low_pos",
    # positioning
    "lsr_global_pctl", "lsr_toptrader_pctl",
]


# =============================================================== io helpers ===
def _ts_index(df: pd.DataFrame, col: str) -> pd.DatetimeIndex:
    return pd.to_datetime(df[col].to_numpy(), unit="ms", utc=True)


def load_spot_1h(pair: str) -> pd.DataFrame | None:
    p = SPOT_1H_DIR / f"{pair}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = _ts_index(df, "open_time")
    return df.sort_index()


def load_oi_5m(pair: str) -> pd.DataFrame | None:
    p = OI_5M_DIR / f"{pair}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["ts_ms", "sum_open_interest",
                                     "sum_open_interest_value",
                                     "count_long_short_ratio",
                                     "sum_toptrader_long_short_ratio"])
    df.index = _ts_index(df, "ts_ms")
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]


def load_funding(pair: str) -> pd.Series | None:
    p = FUND_DIR / f"{pair}.parquet"
    if not p.exists():
        return None
    f = pd.read_parquet(p, columns=["fundingTime", "fundingRate"])
    s = pd.Series(f["fundingRate"].to_numpy(),
                  index=pd.to_datetime(f["fundingTime"].to_numpy(),
                                       unit="ms", utc=True)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_fut_1m(pair: str) -> pd.DataFrame | None:
    p = FUT_1M_DIR / f"{pair}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close"])
    df.index = _ts_index(df, "open_time")
    return df.sort_index()


def _pos(idx: pd.DatetimeIndex, cutoff: pd.Timestamp) -> int:
    """count of index entries strictly before cutoff (== as-of slice length)."""
    return int(idx.searchsorted(cutoff, side="left"))


# ============================================================ feature blocks ==
# Every block takes RAW inputs + the event and slices as-of internally. Returns
# a plain dict. NaN wherever history/coverage is insufficient.

def hourly_feats(ev: pd.Series, kf: pd.DataFrame) -> dict:
    cutoff = ev["ts"] + H1
    n = _pos(kf.index, cutoff)                       # trigger bar is kf.iloc[n-1]
    out = {k: np.nan for k in
           ["ret_6h", "ret_1h", "ret_24h", "dist_low_30d", "wick_frac",
            "vol_spike", "taker_buy_share", "rv_7d", "vol_ratio", "atr24_norm"]}
    if n < 2:
        return out
    close = kf["close"].to_numpy()[:n]
    high = kf["high"].to_numpy()[:n]
    low = kf["low"].to_numpy()[:n]
    qv = kf["quote_volume"].to_numpy()[:n]
    c0 = close[-1]
    out["ret_1h"] = c0 / close[-2] - 1.0
    if n > W_6H:
        out["ret_6h"] = c0 / close[-1 - W_6H] - 1.0
    if n > W_24H:
        out["ret_24h"] = c0 / close[-1 - W_24H] - 1.0
    if n >= W_30D:
        lo30 = low[-W_30D:].min()
        out["dist_low_30d"] = c0 / lo30 - 1.0 if lo30 > 0 else np.nan
        med30 = np.median(qv[-W_30D:])
        out["vol_spike"] = qv[-1] / med30 if med30 > 0 else np.nan
    hl = high[-1] - low[-1]
    out["wick_frac"] = (c0 - low[-1]) / hl if hl > 0 else np.nan
    if "taker_buy_base" in kf.columns:
        vol = kf["volume"].to_numpy()[:n][-1]
        tbb = kf["taker_buy_base"].to_numpy()[:n][-1]
        out["taker_buy_share"] = tbb / vol if vol > 0 else np.nan
    if n >= W_7D + 1:
        lr = np.diff(np.log(close))                  # n-1 log returns
        out["rv_7d"] = float(np.std(lr[-W_7D:], ddof=1))
        if len(lr) >= W_30D:
            s24 = np.std(lr[-W_24H:], ddof=1)
            s30 = np.std(lr[-W_30D:], ddof=1)
            out["vol_ratio"] = s24 / s30 if s30 > 0 else np.nan
    if n >= W_24H:
        out["atr24_norm"] = float(np.mean(high[-W_24H:] - low[-W_24H:]) / c0)
    return out


def _oi_at(oi_s: np.ndarray, idx: pd.DatetimeIndex, hour_close: pd.Timestamp) -> float:
    """last OI 5m obs strictly before `hour_close` (== resample('1h').last+ffill)."""
    p = _pos(idx, hour_close)
    return float(oi_s[p - 1]) if p >= 1 else np.nan


def oi_feats(ev: pd.Series, kf: pd.DataFrame, oi: pd.DataFrame) -> dict:
    cutoff = ev["ts"] + H1
    out = {k: np.nan for k in
           ["doi6", "doi24", "oi_turnover", "lsr_global_pctl", "lsr_toptrader_pctl"]}
    if oi is None or len(oi) == 0:
        return out
    idx = oi.index
    oi_amt = oi["sum_open_interest"].to_numpy()
    oi_val = oi["sum_open_interest_value"].to_numpy()
    now = _oi_at(oi_amt, idx, ev["ts"] + H1)
    v6 = _oi_at(oi_amt, idx, ev["ts"] - pd.Timedelta(hours=W_6H) + H1)
    v24 = _oi_at(oi_amt, idx, ev["ts"] - pd.Timedelta(hours=W_24H) + H1)
    if np.isfinite(now) and np.isfinite(v6) and v6 > 0:
        out["doi6"] = now / v6 - 1.0
    if np.isfinite(now) and np.isfinite(v24) and v24 > 0:
        out["doi24"] = now / v24 - 1.0
    # oi_turnover = last OI value / 30d-median daily spot quote_volume (as-of)
    p = _pos(idx, cutoff)
    oiv_now = float(oi_val[p - 1]) if p >= 1 else np.nan
    nk = _pos(kf.index, cutoff)
    if nk >= 24 and np.isfinite(oiv_now):
        qv = pd.Series(kf["quote_volume"].to_numpy()[:nk], index=kf.index[:nk])
        daily = qv.resample("1D").sum()
        dvol30 = daily.rolling(30, min_periods=10).median()
        d = float(dvol30.iloc[-1]) if len(dvol30) else np.nan
        if np.isfinite(d) and d > 0:
            out["oi_turnover"] = oiv_now / d
    # positioning percentiles over trailing 30d of 5m bars
    for col, key in [("count_long_short_ratio", "lsr_global_pctl"),
                     ("sum_toptrader_long_short_ratio", "lsr_toptrader_pctl")]:
        vals = oi[col].to_numpy()[:p]
        if p >= 1 and np.isfinite(vals[-1]):
            win = vals[max(0, p - W_OI_30D):p]
            win = win[np.isfinite(win)]
            if len(win) >= 100:
                out[key] = float(np.mean(win < vals[-1]))
    return out


def funding_feats(ev: pd.Series, fund: pd.Series) -> dict:
    out = {"funding_last": np.nan, "funding_trail3": np.nan}
    if fund is None or len(fund) == 0:
        return out
    p = _pos(fund.index, ev["ts"] + H1)
    if p >= 1:
        v = fund.to_numpy()[:p]
        out["funding_last"] = float(v[-1])
        out["funding_trail3"] = float(np.mean(v[-3:]))
    return out


def micro_feats(ev: pd.Series, fut: pd.DataFrame) -> dict:
    out = {"basis_bps": np.nan, "m1_maxdrop": np.nan, "m1_low_pos": np.nan,
           "covered_1m": False}
    if fut is None or len(fut) == 0:
        return out
    lo = _pos(fut.index, ev["ts"])          # first 1m bar of the trigger hour
    hi = _pos(fut.index, ev["ts"] + H1)     # end of trigger hour (exclusive)
    m = fut.iloc[lo:hi]
    if len(m) == 0:
        return out
    out["covered_1m"] = True
    o = m["open"].to_numpy(); c = m["close"].to_numpy(); low = m["low"].to_numpy()
    out["basis_bps"] = (c[-1] / ev["trig_close"] - 1.0) * 1e4
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(o > 0, c / o - 1.0, np.nan)
    if np.isfinite(r).any():
        out["m1_maxdrop"] = float(np.nanmin(r))
    out["m1_low_pos"] = float(int(np.argmin(low))) / 60.0
    return out


def btc_feats(ev: pd.Series, btc_close: pd.Series) -> dict:
    out = {"btc_ret_6h": np.nan, "btc_ret_24h": np.nan, "btc_regime": np.nan}
    if btc_close is None:
        return out
    n = _pos(btc_close.index, ev["ts"] + H1)
    if n < 2:
        return out
    c = btc_close.to_numpy()[:n]
    if n > W_6H:
        out["btc_ret_6h"] = c[-1] / c[-1 - W_6H] - 1.0
    if n > W_24H:
        out["btc_ret_24h"] = c[-1] / c[-1 - W_24H] - 1.0
    if n >= W_50D:
        out["btc_regime"] = 1.0 if c[-1] > float(np.mean(c[-W_50D:])) else 0.0
    return out


def mkt_feats(ev: pd.Series, ev_ts: np.ndarray, ev_sym: np.ndarray) -> dict:
    """counts over ALL events; window (t-w, t] (self included). Lookahead-safe:
    every counted event triggered at ts<=t, known by trigger close."""
    t = ev["ts"].value
    lo24 = t - int(24 * 3.6e12)
    lo6 = t - int(6 * 3.6e12)
    m24 = (ev_ts > lo24) & (ev_ts <= t)
    m6 = (ev_ts > lo6) & (ev_ts <= t)
    return {"mkt_events_24h": int(m24.sum()),
            "mkt_share_6h": len(np.unique(ev_sym[m6])) / float(N_UNIVERSE)}


def symhist_feats(ev: pd.Series, hist: pd.DataFrame) -> dict:
    """strictly-prior same-symbol events. sym_past_n = prior triggers (occurrence
    known at their ts). sym_past_mean_ret = mean net_ret of prior FILLED events
    whose 24h hold fully closed by trigger close (exit_ts < cutoff)."""
    cutoff = ev["ts"] + H1
    prior = hist[(hist["symbol"] == ev["symbol"]) & (hist["ts"] < ev["ts"])]
    n = int(len(prior))
    realized = prior[prior["filled"] & (prior["exit_ts"] < cutoff)]
    mean_ret = float(realized["net_ret"].mean()) if len(realized) else np.nan
    return {"sym_past_n": n, "sym_past_mean_ret": mean_ret}


# ================================================ build the full dataset ======
def build_dataset() -> pd.DataFrame:
    pairs = [c.pair for c in load_universe()]

    # ---- Phase 1: detect all events (global table) --------------------------
    ev_frames = []
    for i, pair in enumerate(pairs, 1):
        e = detect_events(pair)
        if len(e):
            ev_frames.append(e)
        if i % 40 == 0:
            print(f"  detect [{i}/{len(pairs)}] "
                  f"events={sum(len(x) for x in ev_frames)}", flush=True)
    events = (pd.concat(ev_frames, ignore_index=True)
              .sort_values("ts").reset_index(drop=True))
    print(f"detected events: {len(events)} across "
          f"{events['symbol'].nunique()} symbols "
          f"({events['ts'].min()} .. {events['ts'].max()})")

    # ---- Phase 2: labels + per-symbol market-data features ------------------
    ev_sym_ts = np.array([t.value for t in events["ts"]])  # (for later)
    label_rows, feat_rows = [], []
    ev_by_sym = {s: g for s, g in events.groupby("symbol")}
    for si, (pair, g) in enumerate(ev_by_sym.items(), 1):
        kf = load_spot_1h(pair)
        if kf is None:
            continue
        # labels via frozen maker config (per-symbol simulate -> memory light)
        tr = simulate(g, {pair: kf}, "maker", "none", 0.0010, False)
        tr = tr.rename(columns={"ret": "net_ret"})
        label_rows.append(tr)
        oi = load_oi_5m(pair)
        fund = load_funding(pair)
        fut = load_fut_1m(pair)
        for _, ev in g.iterrows():
            row = {"symbol": pair, "ts": ev["ts"]}
            row.update(hourly_feats(ev, kf))
            row.update(oi_feats(ev, kf, oi))
            row.update(funding_feats(ev, fund))
            row.update(micro_feats(ev, fut))
            feat_rows.append(row)
        del kf, oi, fund, fut
        if si % 25 == 0:
            print(f"  feats [{si}/{len(ev_by_sym)}] rows={len(feat_rows)}",
                  flush=True)

    labels = pd.concat(label_rows, ignore_index=True)
    labels["filled"] = labels["filled"].astype(bool)
    if "exit_ts" not in labels.columns:
        labels["exit_ts"] = pd.NaT
    feats = pd.DataFrame(feat_rows)

    df = feats.merge(
        labels[["symbol", "ts", "filled", "net_ret", "exit_ts"]],
        on=["symbol", "ts"], how="left")

    # ---- Phase 3: cross-symbol features (btc / market / symbol history) -----
    btc_kf = load_spot_1h(BTC_PAIR)
    btc_close = btc_kf["close"] if btc_kf is not None else None
    ev_ts_arr = np.array([t.value for t in df["ts"]])
    ev_sym_arr = df["symbol"].to_numpy()
    hist = df[["symbol", "ts", "filled", "net_ret", "exit_ts"]].copy()
    cross = []
    for _, ev in df.iterrows():
        r = {}
        r.update(btc_feats(ev, btc_close))
        r.update(mkt_feats(ev, ev_ts_arr, ev_sym_arr))
        r.update(symhist_feats(ev, hist))
        cross.append(r)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(cross)], axis=1)

    # ---- finalize ------------------------------------------------------------
    df["year"] = df["ts"].dt.year
    df["win"] = np.where(df["filled"], (df["net_ret"] > 0).astype(float), np.nan)
    df["covered_1m"] = df["covered_1m"].fillna(False).astype(bool)
    keep = (["symbol", "ts", "year", "filled", "net_ret", "win", "covered_1m"]
            + FEATURES)
    df_out = df[keep].sort_values("ts").reset_index(drop=True)
    # `hist` (with exit_ts) is retained for the lookahead checker only; it is
    # NOT part of the saved dataset (exit_ts is not a declared column).
    return df_out, btc_close, events, hist


# ============================================================ lookahead check =
def _feature_vector(ev: pd.Series, kf, oi, fund, fut, btc_close, hist,
                    ev_ts_arr, ev_sym_arr) -> dict:
    v = {}
    v.update(hourly_feats(ev, kf))
    v.update(oi_feats(ev, kf, oi))
    v.update(funding_feats(ev, fund))
    v.update(micro_feats(ev, fut))
    v.update(btc_feats(ev, btc_close))
    v.update(mkt_feats(ev, ev_ts_arr, ev_sym_arr))
    v.update(symhist_feats(ev, hist))
    return v


def lookahead_check(df: pd.DataFrame, btc_close: pd.Series,
                    hist_full: pd.DataFrame, seed: int = 7) -> str:
    """Rebuild 5 random events from PHYSICALLY TRUNCATED inputs (< trigger close)
    and assert equality with stored production values."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=5, replace=False)
    ev_ts_full = np.array([t.value for t in df["ts"]])
    ev_sym_full = df["symbol"].to_numpy()
    passes = 0
    fails = []
    for k in idx:
        ev = df.iloc[k]
        cutoff = ev["ts"] + H1
        pair = ev["symbol"]
        kf = load_spot_1h(pair)
        kf = kf[kf.index < cutoff]                       # truncate
        # trig_close is the spot close of the trigger bar (last row after
        # truncation) — reconstructed here, not read from the stored dataset.
        ev = ev.copy()
        ev["trig_close"] = float(kf["close"].iloc[-1])
        oi = load_oi_5m(pair)
        if oi is not None:
            oi = oi[oi.index < cutoff]
        fund = load_funding(pair)
        if fund is not None:
            fund = fund[fund.index < cutoff]
        fut = load_fut_1m(pair)
        if fut is not None:
            fut = fut[fut.index < cutoff]
        btc = btc_close[btc_close.index < cutoff] if btc_close is not None else None
        # truncate event-derived inputs too (strictly-prior only)
        m = ev_ts_full < cutoff.value
        # rebuild hist with truncated realized outcomes handled inside symhist
        hist = hist_full[hist_full["ts"] < cutoff]
        v = _feature_vector(ev, kf, oi, fund, fut, btc, hist,
                            ev_ts_full[m], ev_sym_full[m])
        ok = True
        for f in FEATURES:
            a, b = ev[f], v.get(f, np.nan)
            if pd.isna(a) and pd.isna(b):
                continue
            if not (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-9):
                ok = False
                fails.append(f"{pair}@{ev['ts']} {f}: prod={a} rebuilt={b}")
        passes += int(ok)
    if passes == 5:
        return "lookahead check: PASS 5/5"
    return "lookahead check: FAIL " + f"{passes}/5\n  " + "\n  ".join(fails[:20])


# ==================================================================== EDA =====
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def run_eda(df: pd.DataFrame, n_boot: int = 1000, seed: int = 11) -> dict:
    dev = df[df["ts"] < HOLDOUT_START].copy()
    assert (dev["ts"] < HOLDOUT_START).all(), "EDA leaked holdout rows!"
    fil = dev[dev["filled"]].copy()
    nr = fil["net_ret"].to_numpy()
    print(f"\n=== EDA (DEV only: {len(dev)} events, {len(fil)} filled) ===")

    # ---- label ---------------------------------------------------------------
    s = pd.Series(nr)
    label = {"n_events_dev": int(len(dev)), "n_filled_dev": int(len(fil)),
             "fill_rate": round(float(dev["filled"].mean()), 4),
             "net_ret_mean": round(float(s.mean()), 5),
             "net_ret_median": round(float(s.median()), 5),
             "net_ret_std": round(float(s.std()), 5),
             "net_ret_skew": round(float(s.skew()), 4),
             "win_rate": round(float((s > 0).mean()), 4),
             "p1": round(float(s.quantile(.01)), 5),
             "p5": round(float(s.quantile(.05)), 5),
             "p95": round(float(s.quantile(.95)), 5),
             "p99": round(float(s.quantile(.99)), 5)}
    by_year = (fil.groupby("year")["net_ret"]
               .agg(n="count", mean="mean",
                    win=lambda x: float((x > 0).mean())).round(5))
    print("label:", json.dumps(label, indent=1))
    print("by-year (DEV):\n", by_year.to_string())

    # ---- day-clustered bootstrap day-index sets (shared across features) ----
    fil = fil.reset_index(drop=True)
    fil["day"] = fil["ts"].dt.floor("1D")
    day_groups = {d: g.index.to_numpy() for d, g in fil.groupby("day")}
    days = np.array(list(day_groups.keys()))
    rng = np.random.default_rng(seed)
    boot_idx = []
    for _ in range(n_boot):
        sd = rng.choice(len(days), size=len(days), replace=True)
        boot_idx.append(np.concatenate([day_groups[days[j]] for j in sd]))
    nr_f = fil["net_ret"].to_numpy()

    # ---- per-feature IC + clustered CI --------------------------------------
    ic_rows = []
    for f in FEATURES:
        xf = fil[f].to_numpy()
        ic = _spearman(xf, nr_f)
        bs = np.array([_spearman(xf[b], nr_f[b]) for b in boot_idx])
        bs = bs[np.isfinite(bs)]
        lo = float(np.percentile(bs, 2.5)) if len(bs) else np.nan
        hi = float(np.percentile(bs, 97.5)) if len(bs) else np.nan
        miss = float(dev[f].isna().mean())
        ic_rows.append({"feature": f, "ic": ic, "ci_lo": lo, "ci_hi": hi,
                        "abs_ic": abs(ic) if np.isfinite(ic) else 0.0,
                        "missing": round(miss, 4),
                        "sig": bool(np.isfinite(lo) and np.isfinite(hi)
                                    and (lo > 0 or hi < 0))})
    ic_tbl = pd.DataFrame(ic_rows).sort_values("abs_ic", ascending=False)
    print("\nIC table (DEV filled, sorted |IC|):")
    print(ic_tbl[["feature", "ic", "ci_lo", "ci_hi", "sig", "missing"]]
          .round(4).to_string(index=False))

    # ---- quintile means for top-6 by |IC| -----------------------------------
    quint = {}
    for f in ic_tbl["feature"].head(6):
        sub = fil[[f, "net_ret"]].dropna()
        if sub[f].nunique() < 5:
            continue
        try:
            q = pd.qcut(sub[f], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        means = sub.groupby(q)["net_ret"].mean().round(5)
        quint[f] = {int(k): float(v) for k, v in means.items()}
    print("\nquintile net_ret means (top-6 |IC|):")
    for f, m in quint.items():
        print(f"  {f:18s} Q0..Q4 = {[round(v,4) for v in m.values()]}")

    # ---- feature-feature redundancy (|rho|>0.7) -----------------------------
    corr = fil[FEATURES].corr(method="spearman")
    redun = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) > 0.7:
                redun.append({"a": FEATURES[i], "b": FEATURES[j],
                              "rho": round(float(r), 3)})
    redun.sort(key=lambda d: -abs(d["rho"]))
    print("\nredundancy pairs |rho|>0.7:")
    for d in redun:
        print(f"  {d['a']:18s} ~ {d['b']:18s} rho={d['rho']:+.3f}")

    # ---- event-per-day clustering (DEV) -------------------------------------
    perday = dev.groupby(dev["ts"].dt.floor("1D")).size()
    big = perday[perday >= 5]
    clustering = {"n_days": int(len(perday)),
                  "max_events_per_day": int(perday.max()),
                  "p95_events_per_day": float(np.percentile(perday, 95)),
                  "share_events_on_days_ge5":
                      round(float(dev.groupby(dev["ts"].dt.floor("1D")).size()
                                  .reindex(dev["ts"].dt.floor("1D")).ge(5).mean()),
                            4),
                  "n_days_ge5": int(len(big))}
    print("\nclustering (DEV):", json.dumps(clustering))

    # ---- missingness ---------------------------------------------------------
    missing = {f: round(float(dev[f].isna().mean()), 4) for f in FEATURES}
    print("missingness:", json.dumps({k: v for k, v in missing.items() if v > 0}))

    return {"label": label,
            "by_year": {int(y): r for y, r in
                        by_year.reset_index().to_dict(orient="index").items()},
            "ic_table": ic_tbl.drop(columns="abs_ic").round(5)
                .to_dict(orient="records"),
            "quintiles": quint, "redundancy": redun,
            "clustering": clustering, "missing": missing}


# ==================================================================== main ====
def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    df, btc_close, _events, hist_full = build_dataset()

    out_pq = ART_DIR / "ml_dataset.parquet"
    df.to_parquet(out_pq)
    print(f"\ndataset: {df.shape} -> {out_pq}")
    print(f"filled={int(df['filled'].sum())}/{len(df)} "
          f"({df['filled'].mean():.1%})  covered_1m={df['covered_1m'].mean():.1%}")

    print("\n" + lookahead_check(df, btc_close, hist_full))

    eda = run_eda(df)
    out_json = ART_DIR / "eda_summary.json"
    out_json.write_text(json.dumps(eda, indent=2, default=str), encoding="utf-8")
    print(f"\neda summary -> {out_json}")


if __name__ == "__main__":
    main()
