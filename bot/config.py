"""Bot configuration: pydantic models populated from environment variables.

All secrets (API keys, Telegram token) come ONLY from the environment / .env
file (gitignored). Nothing secret has a default here. See bot/.env.example.

Env var convention: BOT_<FIELD> flat names, mapped below. A sha256 of the
resolved NON-SECRET config is logged at startup (frozen-threshold audit trail).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

BOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BOT_DIR / "artifacts"

REST_BASE = {"live": "https://api.bybit.com", "demo": "https://api-demo.bybit.com"}
WS_PRIVATE = {"live": "wss://stream.bybit.com/v5/private",
              "demo": "wss://stream-demo.bybit.com/v5/private"}
# market data is identical on both contours -> public stream is always mainnet
WS_PUBLIC_LINEAR = "wss://stream.bybit.com/v5/public/linear"

Mode = Literal["off", "paper", "live"]


class BybitConfig(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    contour: Literal["demo", "live"] = "demo"
    recv_window_ms: int = 5000

    @property
    def rest_base(self) -> str:
        return REST_BASE[self.contour]

    @property
    def ws_private(self) -> str:
        return WS_PRIVATE[self.contour]

    @property
    def ws_public(self) -> str:
        return WS_PUBLIC_LINEAR


class TelegramConfig(BaseModel):
    bot_token: str = ""
    chat_id: str = ""


class RiskConfig(BaseModel):
    """Deploy-spec section 3 thresholds. DRAFT until frozen with Mike before
    the first live order; the config sha256 is logged at startup."""
    slots: int = 15
    # safety-first default per external audit 2026-07-10; the uncapped-overlay
    # variant (max_weight 2.0) requires explicit config (BOT_MAX_WEIGHT)
    max_weight: float = 1.0
    gross_cap_mult: float = 1.0         # abs open+pending notional cap, x equity
    gross_safety_buffer: float = 0.05   # cap = mult * (1 - buffer) * equity
    kill_dd: float = 0.10               # account kill-switch, cumulative DD
    fill_rate_min: float = 0.85         # fill-rate gate ...
    fill_rate_window: int = 30          # ... over the last N live signals
    review_trades: int = 50             # review gate: pause if mean net < 0
    max_trade_notional_usd: float = 1000.0
    min_notional_usd: float = 5.0       # Bybit linear minimum


class LiqrevConfig(BaseModel):
    """Frozen liqrev v2 detector/execution parameters (deploy-spec section 1).
    Parameters sit on a validated plateau — never retune."""
    mode: Mode = "paper"                # paper until go-live decision
    ret6_max: float = -0.08
    doi6_max: float = -0.10
    liq_gate_usd: float = 1_000_000.0   # 30d-median daily quote volume
    cooldown_h: int = 24
    ttl_s: int = 3600                   # maker limit self-cancel
    stop_pct: float = 0.20              # disaster stop, server-side
    hold_h: int = 24                    # exit at trigger-bar close + 24h (= close of bar i+24)


class UnlockConfig(BaseModel):
    """S2 post-cliff long (unlock-liqrev-glue.md). PAPER ONLY (n=23 LIVE)."""
    mode: Literal["off", "paper"] = "paper"
    min_frac_supply: float = 0.03
    hold_days: int = 7
    refresh_utc_hour: int = 6           # daily DefiLlama calendar refresh


class BotConfig(BaseModel):
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    paper_equity_usd: float = 5000.0    # equity basis when no live wallet
    clock_max_drift_s: float = 5.0
    bybit: BybitConfig = BybitConfig()
    telegram: TelegramConfig = TelegramConfig()
    risk: RiskConfig = RiskConfig()
    liqrev: LiqrevConfig = LiqrevConfig()
    unlock: UnlockConfig = UnlockConfig()

    def public_sha256(self) -> str:
        """Hash of the non-secret config (audit trail for frozen thresholds)."""
        d = self.model_dump(mode="json")
        d["bybit"].pop("api_key", None)
        d["bybit"].pop("api_secret", None)
        d.pop("telegram", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def load_config(env_file: str | Path | None = None) -> BotConfig:
    """Build config from environment; .env is loaded first if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file or Path(".env"), override=False)
    except ImportError:
        pass

    cfg = BotConfig(
        data_dir=Path(_env("BOT_DATA_DIR", "data")),
        log_level=_env("BOT_LOG_LEVEL", "INFO"),
        paper_equity_usd=float(_env("BOT_PAPER_EQUITY_USD", "5000")),
        bybit=BybitConfig(
            api_key=_env("BOT_BYBIT_API_KEY", "") or "",
            api_secret=_env("BOT_BYBIT_API_SECRET", "") or "",
            contour=_env("BOT_CONTOUR", "demo"),  # type: ignore[arg-type]
        ),
        telegram=TelegramConfig(
            bot_token=_env("BOT_TELEGRAM_TOKEN", "") or "",
            chat_id=_env("BOT_TELEGRAM_CHAT_ID", "") or "",
        ),
        risk=RiskConfig(
            max_weight=float(_env("BOT_MAX_WEIGHT", "1.0")),
            gross_cap_mult=float(_env("BOT_GROSS_CAP_MULT", "1.0")),
            gross_safety_buffer=float(_env("BOT_GROSS_SAFETY_BUFFER", "0.05")),
        ),
        liqrev=LiqrevConfig(mode=_env("BOT_LIQREV_MODE", "paper")),  # type: ignore[arg-type]
        unlock=UnlockConfig(mode=_env("BOT_UNLOCK_MODE", "paper")),  # type: ignore[arg-type]
    )
    return cfg
