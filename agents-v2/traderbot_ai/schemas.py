from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DecisionKind = Literal["long", "short", "hold"]


class TradeDecision(BaseModel):
    final_decision: DecisionKind = Field(description="One of long, short, hold.")
    symbol: str = Field(description="Trading symbol, for example BTCUSDT.")
    timeframe: str = Field(description="Main timeframe used for the decision.")
    thesis: str = Field(description="Short trading thesis.")
    price: float | None = Field(default=None, description="Entry/current price. Null only if unavailable.")
    stop_loss: float | None = Field(default=None, description="Stop-loss price. Null for hold if no trade.")
    take_profit: float | None = Field(default=None, description="Take-profit price. Null for hold if no trade.")
    amount: float = Field(default=0.0, ge=0.0, description="USDT amount allocated to the trade.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence from 0 to 1.")
    risk_summary: str = Field(description="Risk notes and validation result.")
    tool_summary: list[str] = Field(default_factory=list, description="Important tools used.")
    worklog_path: str | None = Field(default=None, description="Worklog file written by the agent.")


class RiskValidation(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimated_loss_usdt: float = 0.0
    max_loss_fraction: float = 0.0


class ProviderStatus(BaseModel):
    provider: str
    configured: bool
    live_ok: bool | None = None
    model: str | None = None
    detail: str | None = None
