"""Scanner v2 runtime: ML-filtered momentum breakout scanner.

Policy (validated in walk-forward 2025-04..2026-07, see meta.json):
  LONG-only P1 (20-bar breakout) candidates, BTC>EMA50 regime,
  relaxed pool gates, EV = p*tp_rr - (1-p) - cost > 0,
  one signal per symbol per UTC day, time-priority (no p-ranking within scan).
Execution recommendation: market/next-open entry (retest REJECTED for these
picks), stop 1.0x d_final, TP ~3R, max hold 24h, 3-4 slots, 24h sym cooldown.

Usage (offline/replay):
  sc = ScannerV2()
  out = sc.scan(frames_1h, as_of_ms, emitted_today=set_of_pairs)
frames_1h: dict pair -> 1h klines df with bars whose open_time+1h <= as_of_ms
(trailing >= 1100 bars for warmup parity; BTCUSDT required).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import scan_symbol  # noqa: E402
from features import build_market_frames, extra_symbol_features, join_market  # noqa: E402
from indicators_vec import roc  # noqa: E402
from train_eval import FEATURES  # noqa: E402
from universe import REPO_ROOT, load_universe  # noqa: E402

ART = REPO_ROOT / "research" / "artifacts" / "scanner_v2"
WARMUP_BARS = 1100          # ewm drift < 1e-4 at this depth
COST_BPS_RT = 25.0
EXEC_TP_RR = 3.0


class ScannerV2:
    def __init__(self, artifact_dir: Path = ART):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(artifact_dir / "model.txt"))
        self.meta = json.loads((artifact_dir / "meta.json").read_text())
        cal = self.meta["calibration"]
        self._cal_a, self._cal_b = cal["coef"], cal["intercept"]
        self.features = self.meta["features"]
        coins = load_universe()
        self._meta_df = pd.DataFrame([{
            "symbol": c.pair,
            "tier": {"T1": 1, "T2": 2, "T3": 3}[c.tier],
            "uni_adr_pct": c.adr_pct,
            "uni_log_spot30": math.log10(max(c.spot30_musd, 0.05)),
        } for c in coins])

    def _calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        z = self._cal_a * p_raw + self._cal_b
        return 1.0 / (1.0 + np.exp(-z))

    def scan(self, frames_1h: dict[str, pd.DataFrame], as_of_ms: int,
             emitted_today: set[str] | None = None,
             cost_bps: float = COST_BPS_RT,
             min_ev: float = 0.0) -> dict:
        """Returns {'candidates': [...], 'heat24': int, 'regime_ok': bool}."""
        emitted_today = emitted_today or set()
        trigger_open = as_of_ms - 3_600_000

        cut = {}
        for pair, df in frames_1h.items():
            d = df[df["open_time"] <= trigger_open]
            if len(d) > WARMUP_BARS:
                d = d.iloc[-WARMUP_BARS:].reset_index(drop=True)
            if len(d) >= 400:
                cut[pair] = d
        if "BTCUSDT" not in cut:
            raise ValueError("BTCUSDT frame required")
        btc = cut["BTCUSDT"]
        btc_roc4h = pd.Series(roc(btc["close"], 4).to_numpy(),
                              index=btc["open_time"].to_numpy())
        # regime gate: BTC close > EMA50 on 1h
        btc_ema50 = btc["close"].ewm(span=50, adjust=False).mean()
        regime_ok = bool(btc["close"].iloc[-1] > btc_ema50.iloc[-1])

        events = []
        for pair, d in cut.items():
            ev = scan_symbol(d, pair, btc_roc4h)
            if ev.empty:
                continue
            ev = extra_symbol_features(d, ev)
            events.append(ev)
        if not events:
            return {"candidates": [], "heat24": 0, "regime_ok": regime_ok}
        ev = pd.concat(events, ignore_index=True)
        market, ranks = build_market_frames(cut)
        ev = join_market(ev, market, ranks, self._meta_df)

        # policy filters: long P1, feasible stop, regime
        ev = ev[(ev["side"] == "long") & (ev["pattern"] == "P1") & ev["stop_feasible"]]
        # last 24h of triggers for heat + the current bar for emission
        ev = ev[ev["as_of"] > as_of_ms - 24 * 3_600_000].copy()
        if len(ev) == 0 or not regime_ok:
            return {"candidates": [], "heat24": 0, "regime_ok": regime_ok}

        for p in ("P1", "P1H", "P2", "P3"):
            ev[f"pat_{p}"] = (ev["pattern"] == p).astype(int)
        ev["is_long"] = (ev["side"] == "long").astype(int)
        X = ev[self.features].astype(float)
        p_raw = self.booster.predict(X.to_numpy())
        ev["p_hat"] = self._calibrate(p_raw)
        ev["ev_r"] = (ev["p_hat"] * ev["tp_rr"] - (1 - ev["p_hat"])
                      - cost_bps / 1e4 / ev["d_final"])
        hot = ev[ev["ev_r"] > min_ev]
        heat24 = int(len(hot))

        now = hot[hot["as_of"] == as_of_ms]
        out = []
        for row in now.sort_values("open_time").itertuples():
            if row.symbol in emitted_today:
                continue
            entry = float(row.entry)
            d = float(row.d_final)
            out.append({
                "symbol": row.symbol, "side": "long", "pattern": "P1",
                "as_of": int(row.as_of),
                "p_hat": round(float(row.p_hat), 4),
                "ev_r": round(float(row.ev_r), 3),
                "entry_ref": entry,
                "stop": round(entry * (1 - d), 8),
                "d_final_pct": round(d * 100, 3),
                "tp": round(entry * (1 + d * EXEC_TP_RR), 8),
                "tp_rr_exec": EXEC_TP_RR,
                "max_hold_h": 24,
                "heat24": heat24,
                "entry_policy": "next_open",   # retest rejected for ML picks
                # plan primitives for runner integration (production parity)
                "boundary": float(row.boundary),
                "invalidation": entry * (1 - float(row.d_struct)),
                "d_atr": float(row.d_atr),
                "d_struct": float(row.d_struct),
                "d_noise": float(row.d_noise),
                "d_final": d,
                # display facts for the scan table
                "roc_4h": float(row.roc_4h),
                "roc_24h": float(row.roc_24h),
                "roc_1h": float(row.roc_1h),
                "vol_ratio": float(row.vol_ratio),
                "rsi14": float(row.rsi14),
                "atr_pct": float(row.atr_pct),
                "ema20_ext_atr": float(row.ema20_ext_atr),
            })
        return {"candidates": out, "heat24": heat24, "regime_ok": regime_ok}


if __name__ == "__main__":
    # demo: scan the last fully-closed hour in the research cache
    KL = REPO_ROOT / "research" / "data" / "klines" / "1h"
    frames = {p.stem: pd.read_parquet(p) for p in KL.glob("*.parquet")}
    last = max(int(df["open_time"].iloc[-1]) for df in frames.values())
    as_of = last + 3_600_000
    sc = ScannerV2()
    res = sc.scan(frames, as_of)
    print(f"as_of={pd.Timestamp(as_of, unit='ms')} regime_ok={res['regime_ok']} "
          f"heat24={res['heat24']}")
    for c in res["candidates"]:
        print(c)
