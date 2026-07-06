from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from traderbot_ai.screener.market import normalize_symbol


Side = Literal["long", "short"]
CandidateQuality = Literal["hard", "marginal_extension"]


class OpenPosition(BaseModel):
    symbol: str
    side: Literal["Buy", "Sell", "long", "short"]
    opened_at_ms: int | None = None
    position_id: str | None = None
    orderLinkId: str | None = None

    @property
    def strategy_side(self) -> Side:
        return "long" if self.side in {"Buy", "long"} else "short"


class TradingState(BaseModel):
    open_positions: list[OpenPosition] = Field(default_factory=list)
    last_candidate_ts: dict[str, int] = Field(default_factory=dict)
    last_candidate_quality: dict[str, CandidateQuality] = Field(default_factory=dict)
    last_stopout_ts: dict[str, int] = Field(default_factory=dict)
    trades_opened_today: int = 0
    consecutive_stopouts: int = 0
    daily_realized_pnl_pct: float = 0.0
    weekly_realized_pnl_pct: float = 0.0
    halt: bool = False


class ScreenerStateStore(BaseModel):
    last_candidate_ts: dict[str, int] = Field(default_factory=dict)
    last_candidate_quality: dict[str, CandidateQuality] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ScreenerStateStore":
        target = Path(path)
        if not target.exists():
            return cls()
        return cls.model_validate(json.loads(target.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=True), encoding="utf-8")

    def update_from_scan_rows(self, rows: list[dict], as_of_ms: int) -> None:
        for row in rows:
            side = row.get("candidate")
            symbol = row.get("symbol")
            if symbol and side in {"long", "short"}:
                key = candidate_cooldown_key(str(symbol), side)
                quality = row.get("candidate_quality") or "hard"
                self.last_candidate_ts[key] = int(as_of_ms)
                if quality in {"hard", "marginal_extension"}:
                    self.last_candidate_quality[key] = quality


def candidate_cooldown_key(symbol: str, side: Side) -> str:
    return f"{normalize_symbol(symbol)}:{side}"


def candidate_cooldown_entry(symbol: str, side: Side, state: TradingState) -> tuple[int | None, CandidateQuality | None]:
    normalized = normalize_symbol(symbol)
    side_key = candidate_cooldown_key(normalized, side)
    if side_key in state.last_candidate_ts:
        return state.last_candidate_ts.get(side_key), state.last_candidate_quality.get(side_key)
    if normalized in state.last_candidate_ts:
        return state.last_candidate_ts.get(normalized), state.last_candidate_quality.get(normalized)
    return None, None


def state_from_wallet(wallet: dict, last_candidate_ts: dict[str, int] | None = None, last_candidate_quality: dict[str, CandidateQuality] | None = None) -> TradingState:
    positions = []
    for item in wallet.get("open_positions") or []:
        positions.append(
            OpenPosition(
                symbol=str(item.get("symbol") or ""),
                side=item.get("side") or "Buy",
                opened_at_ms=item.get("opened_at_ms"),
                position_id=item.get("position_id"),
                orderLinkId=item.get("orderLinkId"),
            )
        )
    return TradingState(open_positions=positions, last_candidate_ts=dict(last_candidate_ts or {}), last_candidate_quality=dict(last_candidate_quality or {}))


def state_from_wallet_and_events(
    wallet: dict[str, Any],
    events: list[dict[str, Any]],
    as_of_ms: int,
    last_candidate_ts: dict[str, int] | None = None,
    last_candidate_quality: dict[str, CandidateQuality] | None = None,
) -> TradingState:
    state = state_from_wallet(wallet, last_candidate_ts=last_candidate_ts, last_candidate_quality=last_candidate_quality)
    equity = _wallet_equity(wallet)
    day_start = _utc_day_start_ms(as_of_ms)
    week_start = int(as_of_ms) - 7 * 24 * 60 * 60_000
    daily_pnl = 0.0
    weekly_pnl = 0.0
    opened_today = 0
    closed_events: list[tuple[int, dict[str, Any]]] = []

    for record in events:
        raw_payload = record.get("payload")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        event_type = str(record.get("type") or "")
        event_ms = _event_replay_time_ms(record)
        if event_ms is None or event_ms > int(as_of_ms):
            continue
        if event_type == "place_order" and day_start <= event_ms <= int(as_of_ms) and _is_filled_linear_order(payload):
            opened_today += 1
        if event_type == "position_closed":
            closed_events.append((event_ms, payload))
            realized = _optional_float(payload.get("realized_pnl_usdt"))
            if realized is not None:
                if day_start <= event_ms <= int(as_of_ms):
                    daily_pnl += realized
                if week_start <= event_ms <= int(as_of_ms):
                    weekly_pnl += realized
            if _is_stopout(payload):
                symbol = payload.get("symbol")
                if symbol:
                    state.last_stopout_ts[normalize_symbol(str(symbol))] = event_ms

    state.trades_opened_today = opened_today
    if equity and equity > 0:
        state.daily_realized_pnl_pct = daily_pnl / equity
        state.weekly_realized_pnl_pct = weekly_pnl / equity
    state.consecutive_stopouts = _consecutive_stopouts(closed_events)
    return state


def _wallet_equity(wallet: dict[str, Any]) -> float | None:
    totals = wallet.get("totals") if isinstance(wallet.get("totals"), dict) else {}
    for key in ("equity_usdt", "total_equity_usdt"):
        value = totals.get(key) if totals else wallet.get(key)
        number = _optional_float(value)
        if number is not None:
            return number
    return None


def _utc_day_start_ms(as_of_ms: int) -> int:
    current = datetime.fromtimestamp(int(as_of_ms) / 1000.0, tz=timezone.utc)
    return int(current.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)


def _event_replay_time_ms(record: dict[str, Any]) -> int | None:
    raw_payload = record.get("payload")
    payload: dict[str, Any] | None = raw_payload if isinstance(raw_payload, dict) else None
    for container in (record, payload):
        if not isinstance(container, dict):
            continue
        for key in ("as_of_ms", "timestamp_ms", "ts_ms", "exit_time_ms", "settled_until_ms", "created_at_ms", "opened_at_ms"):
            value = container.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
    return None


def _is_filled_linear_order(payload: dict[str, Any]) -> bool:
    if str(payload.get("category", "")).lower() != "linear":
        return False
    return str(payload.get("status", "")).lower() == "filled"


def _is_stopout(payload: dict[str, Any]) -> bool:
    reason = str(payload.get("exit_reason") or "").lower()
    if "stop" in reason or "liquidation" in reason:
        return True
    if reason:
        return False
    realized = _optional_float(payload.get("realized_pnl_usdt"))
    return realized is not None and realized < 0


def _consecutive_stopouts(closed_events: list[tuple[int, dict[str, Any]]]) -> int:
    count = 0
    for _, payload in sorted(closed_events, key=lambda item: item[0], reverse=True):
        if not _is_stopout(payload):
            break
        count += 1
    return count


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
