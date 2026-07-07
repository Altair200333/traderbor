"""Outcome labeling on 5m paths (production gate-lab uses 1m; 5m chosen for
149x24m scale — ambiguity rate is measured and reported).

Per event (entry at trigger close, from as_of, horizon 24h — matrix.py parity):
- market entry: stop = entry -/+ d_final, tp = entry +/- d_final*tp_rr.
  First 5m bar touching either barrier resolves; same-bar both -> ambiguous.
- retest entry (production limit_retest policy p=0.4 TTL=120m): limit at
  entry -/+ p*d_final; fill on first bar with adverse excursion >= p*d;
  unfilled in TTL -> skip (r=0). R accounted in ORIGINAL d units with +p
  head-start (matrix.retest_sweep parity): tp -> +(tp_rr+p), sl -> -(1-p),
  none -> end_r+p.
- extras: MFE/MAE in R over horizon, time-to-exit, end-of-horizon r.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_MS = 24 * 3_600_000
RETEST_P = 0.4
RETEST_TTL_MS = 120 * 60_000


def label_symbol(events: pd.DataFrame, df5m: pd.DataFrame) -> pd.DataFrame:
    """events: one symbol's events (from candidates.scan_symbol).
    df5m: that symbol's 5m klines. Returns events + label columns."""
    t5 = df5m["open_time"].to_numpy(dtype=np.int64)
    hi5 = df5m["high"].to_numpy(dtype=np.float64)
    lo5 = df5m["low"].to_numpy(dtype=np.float64)
    cl5 = df5m["close"].to_numpy(dtype=np.float64)

    n = len(events)
    out = {
        "outcome": np.full(n, "missing_5m", dtype=object),
        "r_market": np.zeros(n),
        "t_exit_min": np.full(n, np.nan),
        "mfe_r": np.full(n, np.nan),
        "mae_r": np.full(n, np.nan),
        "end_r": np.full(n, np.nan),
        "retest_filled": np.zeros(n, dtype=bool),
        "retest_outcome": np.full(n, "no_fill", dtype=object),
        "r_retest": np.zeros(n),
        "retest_fill_min": np.full(n, np.nan),
    }
    as_of = events["as_of"].to_numpy(dtype=np.int64)
    entry = events["entry"].to_numpy(dtype=np.float64)
    d = events["d_final"].to_numpy(dtype=np.float64)
    rr = events["tp_rr"].to_numpy(dtype=np.float64)
    is_long = (events["side"] == "long").to_numpy()

    for i in range(n):
        j0 = np.searchsorted(t5, as_of[i], side="left")
        j1 = np.searchsorted(t5, as_of[i] + HORIZON_MS, side="left")
        if j1 - j0 < 3:  # need a real path
            continue
        h, l, c = hi5[j0:j1], lo5[j0:j1], cl5[j0:j1]
        e, dd = entry[i], d[i]
        if not np.isfinite(dd) or dd <= 0:
            out["outcome"][i] = "invalid_plan"
            continue
        if is_long[i]:
            stop_px, tp_px = e * (1 - dd), e * (1 + dd * rr[i])
            hit_sl, hit_tp = l <= stop_px, h >= tp_px
            fav, adv = (h - e), (e - l)
            end_r = (c[-1] - e) / (e * dd)
        else:
            stop_px, tp_px = e * (1 + dd), e * (1 - dd * rr[i])
            hit_sl, hit_tp = h >= stop_px, l <= tp_px
            fav, adv = (e - l), (h - e)
            end_r = (e - c[-1]) / (e * dd)

        out["mfe_r"][i] = fav.max() / (e * dd)
        out["mae_r"][i] = adv.max() / (e * dd)
        out["end_r"][i] = end_r

        j_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
        j_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
        if j_sl < 0 and j_tp < 0:
            oc, r_mkt, j_exit = "none", end_r, -1
        elif j_sl >= 0 and (j_tp < 0 or j_sl < j_tp):
            oc, r_mkt, j_exit = "sl", -1.0, j_sl
        elif j_tp >= 0 and (j_sl < 0 or j_tp < j_sl):
            oc, r_mkt, j_exit = "tp", float(rr[i]), j_tp
        else:  # same bar
            oc, r_mkt, j_exit = "ambiguous", 0.0, j_sl
        out["outcome"][i] = oc
        out["r_market"][i] = r_mkt
        if j_exit >= 0:
            out["t_exit_min"][i] = (t5[j0 + j_exit] - as_of[i]) / 60_000 + 5

        # retest limit entry
        p = RETEST_P
        fill_ok = adv >= p * e * dd
        ttl_bars = int(RETEST_TTL_MS // 300_000)
        fill_win = fill_ok[:ttl_bars]
        if not fill_win.any():
            continue  # no_fill, r_retest stays 0
        jf = int(np.argmax(fill_win))
        out["retest_filled"][i] = True
        out["retest_fill_min"][i] = (t5[j0 + jf] - as_of[i]) / 60_000 + 5
        if oc == "sl" and jf <= j_sl:
            out["retest_outcome"][i] = "sl"
            out["r_retest"][i] = -(1.0 - p)
        elif oc == "tp":
            if jf < j_tp:
                out["retest_outcome"][i] = "tp"
                out["r_retest"][i] = rr[i] + p
            else:  # filled at/after tp bar: treat as ambiguous, no credit
                out["retest_outcome"][i] = "ambiguous"
                out["r_retest"][i] = 0.0
        elif oc == "ambiguous":
            out["retest_outcome"][i] = "ambiguous"
            out["r_retest"][i] = 0.0
        else:  # none
            out["retest_outcome"][i] = "none"
            out["r_retest"][i] = end_r + p

    res = events.copy()
    for k, v in out.items():
        res[k] = v
    return res
