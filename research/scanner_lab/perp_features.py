"""Perp funding + OI/metrics features (scanner-v3 Phase A2/A3).

Per-symbol HOURLY feature frame indexed by ``as_of`` (the 1h bar CLOSE, ms
epoch), computable independently per symbol. Integration entry point:

    perp_feature_frame(symbol, perp_dir) -> pd.DataFrame   # has an `as_of` col

which dataset.py later merges onto events via merge_asof / join on
(symbol, as_of). ``as_of`` matches candidates.py: as_of = open_time + 3_600_000
(candidates.py:208), i.e. the close timestamp of the 1h trigger bar; the bar
START is therefore as_of - 1h == open_time. Every feature is strictly causal
w.r.t. as_of.

--------------------------------------------------------------------------
Funding features (A2) -- TIMESTAMP RULE
--------------------------------------------------------------------------
At as_of t, use ONLY settlements with ``fundingTime <= t``. Settlements have a
~8h nominal cadence but it is VARIABLE (SOL has 2h/4h episodes) and every
fundingTime carries a +1ms jitter, so a nominal 08:00 settlement
(fundingTime = 08:00:00.001) is EXCLUDED at the 08:00 bar close and first
enters from the 09:00 bar close -- the conservative causal choice. ``markPrice``
is NOT used (NaN before ~mid-2024). Features are computed once per settlement
(each row using only settlements up to & including itself) then AS-OF joined
(backward, no tolerance) onto the hourly grid: the value at t is the feature
as of the most recent settlement with fundingTime <= t. Trailing windows
(3d / 30d / 90d) are measured back from that settlement. Because settlements
are the update points, variable-cadence episodes are handled automatically:
funding_cum_3d sums MORE settlements during a 2h/4h episode, it does not assume
8h.

Funding columns:
- funding_rate_last  : most recent settled fundingRate (level).
- funding_z_30d      : (last - mean) / std of the trailing-30d settlements
                       (ddof=0, inclusive of the current settlement).
- funding_cum_3d     : sum of settled rates over the trailing 3 days (carry).
- funding_pctile_90d : rank of funding_rate_last within its own trailing-90d
                       settlements, in [0,1]. Chosen over a binary extreme flag
                       because a percentile keeps magnitude/relative crowding
                       information and needs no arbitrary threshold; a flag is
                       recoverable downstream as (pctile > 0.9).

--------------------------------------------------------------------------
Metrics / OI features (A3) -- TIMESTAMP RULE
--------------------------------------------------------------------------
Snapshots are 5-minute cadence but may LAG, so at as_of t use the latest
snapshot with ``ts_ms <= t - 1h`` (== open_time, the bar START value). If that
latest snapshot is more than 2h older than the reference time the feature is
NaN -- NO forward-fill across gaps larger than 2h. Window changes / rolling
stats that require a lagged snapshot which is itself missing within the 2h
tolerance are also NaN. Rolling stats are time-based (window by wall-clock,
not row count) so real gaps shrink the sample correctly instead of leaking
across them.

Metrics columns:
- oi_chg_1h / oi_chg_4h / oi_chg_24h : % change of sum_open_interest between the
                       reference snapshot and the snapshot 1h/4h/24h earlier.
- oi_z_7d            : z-score of sum_open_interest vs its trailing 7d (ddof=0).
- toptrader_ls       : sum_toptrader_long_short_ratio level.
- toptrader_ls_z_7d  : z-score of that ratio vs its trailing 7d (ddof=0).
- taker_ratio_24h    : 24h trailing mean of sum_taker_long_short_vol_ratio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MS_H = 3_600_000
MS_D = 86_400_000
BAR_LAG_MS = MS_H                 # metrics reference = as_of - 1h (bar start)
METRIC_MAX_STALE_MS = 2 * MS_H    # >2h stale -> NaN (no fwd-fill across gaps)

# rolling min_periods (5m cadence): 288 = 1 day, 48 = 4h
_OI_Z_MINP = 288
_TAKER_MINP = 48

FUNDING_COLS = [
    "funding_rate_last", "funding_z_30d", "funding_cum_3d", "funding_pctile_90d",
]
METRIC_COLS = [
    "oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_z_7d",
    "toptrader_ls", "toptrader_ls_z_7d", "taker_ratio_24h",
]
ALL_FEATURES = FUNDING_COLS + METRIC_COLS


def _empty_frame() -> pd.DataFrame:
    cols = {"as_of": pd.Series([], dtype="int64"),
            "symbol": pd.Series([], dtype="object")}
    for c in ALL_FEATURES:
        cols[c] = pd.Series([], dtype="float64")
    return pd.DataFrame(cols)


def _funding_features(fund: pd.DataFrame) -> pd.DataFrame:
    """Funding features at SETTLEMENT resolution (each row uses only settlements
    up to & including itself). Returns frame with a `fundingTime` (int64 ms)
    column + FUNDING_COLS. markPrice is ignored."""
    f = (fund[["fundingTime", "fundingRate"]]
         .dropna(subset=["fundingTime"])
         .drop_duplicates("fundingTime")
         .sort_values("fundingTime")
         .reset_index(drop=True))
    ft = f["fundingTime"].to_numpy(dtype="int64")
    fr = f["fundingRate"].to_numpy(dtype="float64")
    idx = pd.to_datetime(ft, unit="ms", utc=True)
    s = pd.Series(fr, index=idx)

    m30 = s.rolling("30D", min_periods=3).mean()
    sd30 = s.rolling("30D", min_periods=3).std(ddof=0).replace(0.0, np.nan)
    z30 = (s - m30) / sd30
    cum3 = s.rolling("3D", min_periods=1).sum()
    # trailing-90d percentile of the current (last) settled rate vs the window
    pct90 = s.rolling("90D", min_periods=5).apply(
        lambda w: float((w <= w[-1]).mean()), raw=True)

    return pd.DataFrame({
        "fundingTime": ft,
        "funding_rate_last": fr,
        "funding_z_30d": z30.to_numpy(),
        "funding_cum_3d": cum3.to_numpy(),
        "funding_pctile_90d": pct90.to_numpy(),
    })


def _metric_rolls(m: pd.DataFrame) -> pd.DataFrame:
    """Time-based rolling metric aggregates at 5m snapshot resolution. Returns
    frame with `ts_ms` (int64) + the roll columns (each valid AT its snapshot)."""
    ts = m["ts_ms"].to_numpy(dtype="int64")
    idx = pd.to_datetime(ts, unit="ms", utc=True)
    oi = pd.Series(m["sum_open_interest"].to_numpy("float64"), index=idx)
    tt = pd.Series(m["sum_toptrader_long_short_ratio"].to_numpy("float64"), index=idx)
    tk = pd.Series(m["sum_taker_long_short_vol_ratio"].to_numpy("float64"), index=idx)

    oi_s7 = oi.rolling("7D", min_periods=_OI_Z_MINP).std(ddof=0).replace(0.0, np.nan)
    oi_z = (oi - oi.rolling("7D", min_periods=_OI_Z_MINP).mean()) / oi_s7
    tt_s7 = tt.rolling("7D", min_periods=_OI_Z_MINP).std(ddof=0).replace(0.0, np.nan)
    tt_z = (tt - tt.rolling("7D", min_periods=_OI_Z_MINP).mean()) / tt_s7
    tk_24 = tk.rolling("24h", min_periods=_TAKER_MINP).mean()

    return pd.DataFrame({
        "ts_ms": ts,
        "oi_z_7d": oi_z.to_numpy(),
        "toptrader_ls": tt.to_numpy(),
        "toptrader_ls_z_7d": tt_z.to_numpy(),
        "taker_ratio_24h": tk_24.to_numpy(),
    })


def _asof_backward(ts: np.ndarray, vals: np.ndarray, target: np.ndarray,
                   tol_ms: float | None) -> np.ndarray:
    """For each (monotonic) target, take vals at the latest ts <= target.
    NaN where none exists or (tol_ms set and) the match is staler than tol_ms."""
    pos = np.searchsorted(ts, target, side="right") - 1
    ok = pos >= 0
    cl = np.clip(pos, 0, len(ts) - 1)
    out = np.where(ok, vals[cl], np.nan)
    if tol_ms is not None:
        stale = np.where(ok, target - ts[cl], np.inf)
        out = np.where(stale > tol_ms, np.nan, out)
    return out


def perp_feature_frame(symbol: str, perp_dir: Path) -> pd.DataFrame:
    """Hourly perp feature frame for one symbol.

    Reads {perp_dir}/funding/{symbol}.parquet and
    {perp_dir}/metrics_5m/{symbol}.parquet (either may be absent -> those
    features are NaN). Returns columns ['as_of','symbol', *FUNDING_COLS,
    *METRIC_COLS] with `as_of` an int64 ms epoch on the hour (bar close),
    sorted ascending. Integrate via merge_asof / join on (symbol, as_of)."""
    perp_dir = Path(perp_dir)
    fpath = perp_dir / "funding" / f"{symbol}.parquet"
    mpath = perp_dir / "metrics_5m" / f"{symbol}.parquet"
    fund = pd.read_parquet(fpath) if fpath.exists() else None
    met = pd.read_parquet(mpath) if mpath.exists() else None
    if met is not None:
        met = (met.dropna(subset=["ts_ms"]).drop_duplicates("ts_ms")
               .sort_values("ts_ms").reset_index(drop=True))

    # ---- hourly as_of grid (aligned to :00, spanning available data) ----
    starts, ends = [], []
    if fund is not None and len(fund):
        starts.append(int(fund["fundingTime"].min()))
        ends.append(int(fund["fundingTime"].max()))
    if met is not None and len(met):
        starts.append(int(met["ts_ms"].min()) + BAR_LAG_MS)
        ends.append(int(met["ts_ms"].max()) + BAR_LAG_MS)
    if not starts:
        return _empty_frame()
    g0 = (min(starts) // MS_H) * MS_H
    g1 = (max(ends) // MS_H) * MS_H
    grid = np.arange(g0, g1 + MS_H, MS_H, dtype="int64")
    out = pd.DataFrame({"as_of": grid, "symbol": symbol})

    # ---- funding: merge_asof backward, no tolerance (last settled rate) ----
    if fund is not None and len(fund):
        ff = _funding_features(fund)
        for c in FUNDING_COLS:
            out[c] = _asof_backward(ff["fundingTime"].to_numpy("int64"),
                                    ff[c].to_numpy("float64"), grid, None)
    else:
        for c in FUNDING_COLS:
            out[c] = np.nan

    # ---- metrics: reference = as_of - 1h (bar start), 2h staleness gate ----
    if met is not None and len(met):
        ref = grid - BAR_LAG_MS
        oi_ts = met["ts_ms"].to_numpy("int64")
        oi_val = met["sum_open_interest"].to_numpy("float64")
        tol = METRIC_MAX_STALE_MS
        oi_ref = _asof_backward(oi_ts, oi_val, ref, tol)
        out["oi_chg_1h"] = oi_ref / _asof_backward(oi_ts, oi_val, ref - MS_H, tol) - 1.0
        out["oi_chg_4h"] = oi_ref / _asof_backward(oi_ts, oi_val, ref - 4 * MS_H, tol) - 1.0
        out["oi_chg_24h"] = oi_ref / _asof_backward(oi_ts, oi_val, ref - 24 * MS_H, tol) - 1.0
        rolls = _metric_rolls(met)
        rt = rolls["ts_ms"].to_numpy("int64")
        for c in ("oi_z_7d", "toptrader_ls", "toptrader_ls_z_7d", "taker_ratio_24h"):
            out[c] = _asof_backward(rt, rolls[c].to_numpy("float64"), ref, tol)
    else:
        for c in METRIC_COLS:
            out[c] = np.nan

    return out[["as_of", "symbol"] + ALL_FEATURES].reset_index(drop=True)


if __name__ == "__main__":  # tiny manual smoke
    import sys
    root = Path(__file__).resolve().parents[2]
    pdir = root / "research" / "data" / "perp"
    sym = sys.argv[1] if len(sys.argv) > 1 else "SOLUSDT"
    fr = perp_feature_frame(sym, pdir)
    print(sym, fr.shape)
    print(fr.tail(3).to_string())
