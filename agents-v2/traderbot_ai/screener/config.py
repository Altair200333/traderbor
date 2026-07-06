from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field


class ScreenerConfig(BaseModel):
    version: str = "screener-1.3.1"
    min_1h_bars: int = 168
    min_4h_bars: int = 60
    roc_4h_long: float = 0.025
    roc_24h_long: float = 0.020
    roc_4h_short: float = -0.025
    roc_24h_short: float = -0.020
    vol_ratio_min: float = 2.0
    vol_median_window: int = 24
    rsi_period: int = 14
    rsi_long: tuple[float, float] = (55.0, 78.0)
    rsi_short: tuple[float, float] = (22.0, 45.0)
    atr_period: int = 14
    atr_pct_min: float = 0.005
    atr_pct_max: float = 0.040
    ema_fast: int = 20
    ema_slow: int = 50
    btc_symbol: str = "BTCUSDT"
    btc_roc_4h_block_long: float = -0.010
    btc_roc_4h_block_short: float = 0.010
    funding_long_max: float = 0.0005
    funding_short_min: float = -0.0005
    last_hour_share_max: float = 0.45
    ext_atr_max: float = 1.5
    breakout_dist_atr_max: float = 0.8
    s10_lookback_bars: int = 72
    s10_range_expansion_max: float = 0.35
    s11_zscore_window: int = 20
    s11_zscore_max: float = 3.0
    max_entry_drift_pct: float = 0.02
    marginal_extension_enabled: bool = True
    marginal_extension_sides: tuple[str, ...] = ("long",)
    marginal_ext_atr_max: float = 2.5
    marginal_breakout_dist_atr_max: float = 1.6
    marginal_extension_max_per_scan: int = 2
    marginal_extension_patterns: tuple[str, ...] = ("P1", "P1H", "P3")
    # loose until gate-lab shows score-vs-outcome correlation; score is a tie-breaker, not a validated ranking
    max_candidates_per_scan: int = 4
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {"S1": 1.0, "S2": 1.0, "S3": 1.0, "S9a": 1.0, "S9b": 1.0, "S9c": 1.0, "S10": 1.0, "S11": 1.0}
    )
    p1_lookback: int = 20
    p1_hold_bars: int = 6
    p2_trend_window: int = 24
    p2_trend_share: float = 0.80
    p2_pullback_bars: int = 3
    p3_atr_lookback_bars: int = 72
    p3_atr_compression: float = 0.7
    p3_range_bars: int = 48
    stop_atr_mult_default: float = 2.0
    stop_pct_min: float = 0.010
    stop_pct_max: float = 0.040
    stop_noise_median_window: int = 20
    stop_noise_mult: float = 1.5
    stop_atr_floor_mult: float = 1.5
    cksp_p: int = 10
    cksp_x: float = 1.0
    cksp_q: int = 9
    struct_buffer_p1_p3_atr: float = 0.5
    struct_buffer_p2_atr: float = 0.5
    tp_rr_default: dict[str, float] = Field(default_factory=lambda: {"P1": 2.5, "P1H": 2.0, "P2": 2.0, "P3": 2.5})
    cooldown_candidate_ms: int = 4 * 3_600_000
    cooldown_stopout_ms: int = 24 * 3_600_000
    max_open_positions: int = 3
    max_same_direction: int = 2
    max_trades_per_day: int = 3
    stoploss_guard_lookback_ms: int = 24 * 3_600_000
    stoploss_guard_trade_limit: int = 3
    stoploss_guard_stop_duration_ms: int = 6 * 3_600_000
    stoploss_guard_only_per_side: bool = True
    daily_loss_limit_pct: float = -0.03
    weekly_loss_limit_pct: float = -0.06
    max_hold_hours: float = 24.0
    impulse_break_roc: float = 0.010


def config_hash(cfg: ScreenerConfig) -> str:
    payload = cfg.model_dump(mode="json")
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
