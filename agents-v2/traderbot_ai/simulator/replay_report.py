from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_replay_report(replay_path: str | Path, exchange_events_path: str | Path | None = None) -> dict[str, Any]:
    replay_path = Path(replay_path)
    replay_read = _read_jsonl(replay_path)
    replay_events = _last_replay_run(replay_read["events"])
    exchange_read = _read_jsonl(exchange_events_path) if exchange_events_path is not None else {"events": [], "skipped_lines": 0}
    exchange_events = exchange_read["events"]
    started = _first_event(replay_events, "replay_started")
    completed = _last_event(replay_events, "replay_completed")
    failed = _last_event(replay_events, "replay_failed")
    steps = [_step_summary(event.get("payload") or {}) for event in replay_events if event.get("type") == "step_completed"]
    initial_wallet = steps[0]["wallet_before"] if steps else {}
    final_wallet = _final_wallet_payload(completed, failed, steps)
    initial_equity = _wallet_equity(initial_wallet)
    final_equity = _wallet_equity(final_wallet)
    return {
        "status": _status(started, completed, failed),
        "ok": _status(started, completed, failed) == "completed",
        "replay_path": str(replay_path),
        "exchange_events_path": None if exchange_events_path is None else str(exchange_events_path),
        "config": ((started or {}).get("payload") or {}).get("config", {}),
        "skipped_replay_lines": replay_read["skipped_lines"],
        "skipped_exchange_event_lines": exchange_read["skipped_lines"],
        "step_count": len(steps),
        "started_at": None if started is None else started.get("timestamp"),
        "completed_at": None if completed is None else completed.get("timestamp"),
        "failed_at": None if failed is None else failed.get("timestamp"),
        "failure": None if failed is None else failed.get("payload", {}),
        "initial_equity_usdt": initial_equity,
        "final_equity_usdt": final_equity,
        "equity_delta_usdt": None if initial_equity is None or final_equity is None else final_equity - initial_equity,
        "initial_wallet": initial_wallet,
        "final_wallet": final_wallet,
        "steps": steps,
        "replay_event_counts": dict(Counter(event.get("type", "unknown") for event in replay_events)),
        "exchange_event_counts": dict(Counter(event.get("type", "unknown") for event in exchange_events)),
        "exchange_events": [_exchange_event_summary(event) for event in exchange_events],
    }


def render_replay_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Exchange replay report",
        "",
        f"- Status: {report.get('status')}",
        f"- Steps: {report.get('step_count')}",
        f"- Initial equity USDT: {_money(report.get('initial_equity_usdt'))}",
        f"- Final equity USDT: {_money(report.get('final_equity_usdt'))}",
        f"- Equity delta USDT: {_money(report.get('equity_delta_usdt'))}",
    ]
    config = report.get("config") or {}
    if config:
        lines.extend(
            [
                f"- Run id: {config.get('run_id')}",
                f"- Symbols: {', '.join(config.get('symbols') or [])}",
                f"- Window: {_time_ms(config.get('start_ms'))} -> {_time_ms(config.get('end_ms'))}",
                f"- Decision interval: {config.get('decision_interval')}",
                f"- Execution interval: {config.get('execution_interval')}",
            ]
        )
    if report.get("failure"):
        failure = report["failure"]
        lines.extend(["", "## Failure", "", f"- Type: {failure.get('error_type')}", f"- Error: {failure.get('error')}"])
    elif report.get("status") == "incomplete":
        lines.extend(["", "## Incomplete", "", "Replay started but has no terminal completion or failure event."])
    lines.extend(["", "## Steps", ""])
    if not report.get("steps"):
        lines.append("No completed steps.")
    for step in report.get("steps", []):
        decision = step.get("decision") or {}
        lines.extend(
            [
                f"### {_time_ms(step.get('as_of_ms'))}",
                "",
                f"- Decision: {decision.get('final_decision')} {decision.get('symbol') or ''}",
                f"- Wallet equity: {_money(step.get('wallet_before_equity_usdt'))} -> {_money(step.get('wallet_after_equity_usdt'))}",
                f"- Wallet delta: {_money(step.get('wallet_delta_usdt'))}",
                f"- Settlement fills/closes: {step.get('settlement_filled_count', 0)} fills, {step.get('settlement_closed_count', 0)} closes",
                f"- Agent exchange events: {', '.join(step.get('agent_exchange_event_types') or []) or 'none'}",
            ]
        )
        if step.get("settlement_ambiguous_count"):
            details = []
            resolution_intervals = ", ".join(step.get("settlement_resolution_intervals") or [])
            source_intervals = ", ".join(step.get("settlement_source_intervals") or [])
            if resolution_intervals:
                details.append(f"resolution: {resolution_intervals}")
            if source_intervals:
                details.append(f"source: {source_intervals}")
            lines.append(f"- Ambiguous settlement closes: {step['settlement_ambiguous_count']} ({'; '.join(details) or 'unknown'})")
        thesis = decision.get("thesis")
        if thesis:
            lines.append(f"- Thesis: {thesis}")
        risk = decision.get("risk_summary")
        if risk:
            lines.append(f"- Risk: {risk}")
        lines.append("")
    counts = report.get("exchange_event_counts") or {}
    if counts:
        lines.extend(["## Exchange Events", ""])
        for key in sorted(counts):
            lines.append(f"- {key}: {counts[key]}")
        rendered = 0
        exchange_events = report.get("exchange_events") or []
        for event in exchange_events:
            if rendered >= 20:
                break
            details = [event.get("type") or "unknown"]
            for key in ("symbol", "side", "orderType", "status", "exit_reason", "realized_pnl_usdt", "ambiguous", "resolution_interval", "source_interval"):
                value = event.get(key)
                if value is not None:
                    rendered_value = _money(value) if key == "realized_pnl_usdt" else value
                    details.append(f"{key}={rendered_value}")
            lines.append(f"  - {'; '.join(details)}")
            rendered += 1
        if len(exchange_events) > rendered:
            lines.append(f"  - ... and {len(exchange_events) - rendered} more")
    return "\n".join(lines).rstrip() + "\n"


def write_replay_markdown_report(
    replay_path: str | Path,
    output_path: str | Path,
    exchange_events_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_replay_report(replay_path=replay_path, exchange_events_path=exchange_events_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_replay_markdown(report), encoding="utf-8")
    return {**report, "markdown_path": str(output)}


def _read_jsonl(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"events": [], "skipped_lines": 0}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    events = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(parsed, dict):
                skipped += 1
                continue
            events.append(parsed)
    return {"events": events, "skipped_lines": skipped}


def _last_replay_run(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_index = None
    for index, event in enumerate(events):
        if event.get("type") == "replay_started":
            start_index = index
    return events if start_index is None else events[start_index:]


def _first_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("type") == event_type), None)


def _last_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") == event_type:
            return event
    return None


def _status(started: dict[str, Any] | None, completed: dict[str, Any] | None, failed: dict[str, Any] | None) -> str:
    if failed is not None:
        return "failed"
    if completed is not None:
        return "completed"
    if started is not None:
        return "incomplete"
    return "unknown"


def _step_summary(payload: dict[str, Any]) -> dict[str, Any]:
    wallet_before = payload.get("wallet_before") or {}
    wallet_after = payload.get("wallet_after") or {}
    before_equity = _wallet_equity(wallet_before)
    after_equity = _wallet_equity(wallet_after)
    settlement = payload.get("settlement") or {}
    closed_positions = settlement.get("closed_positions") or []
    ambiguous_closes = [item for item in closed_positions if isinstance(item, dict) and item.get("ambiguous") is True]
    resolution_intervals = sorted(
        {
            str(item.get("resolution_interval"))
            for item in ambiguous_closes
            if isinstance(item, dict) and item.get("resolution_interval") is not None
        }
    )
    source_intervals = sorted(
        {
            str(item.get("source_interval"))
            for item in ambiguous_closes
            if isinstance(item, dict) and item.get("source_interval") is not None
        }
    )
    agent_events = payload.get("agent_exchange_events") or []
    return {
        "as_of_ms": payload.get("as_of_ms"),
        "as_of_iso": payload.get("as_of_iso"),
        "decision": payload.get("decision") or {},
        "wallet_before": wallet_before,
        "wallet_after": wallet_after,
        "wallet_before_equity_usdt": before_equity,
        "wallet_after_equity_usdt": after_equity,
        "wallet_delta_usdt": None if before_equity is None or after_equity is None else after_equity - before_equity,
        "settlement_filled_count": len(settlement.get("filled_orders") or []),
        "settlement_closed_count": len(closed_positions),
        "settlement_ambiguous_count": len(ambiguous_closes),
        "settlement_resolution_intervals": resolution_intervals,
        "settlement_source_intervals": source_intervals,
        "agent_exchange_event_types": [event.get("type", "unknown") for event in agent_events],
        "agent_exchange_events": [_exchange_event_summary(event) for event in agent_events],
    }


def _final_wallet_payload(
    completed: dict[str, Any] | None,
    failed: dict[str, Any] | None,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    if completed is not None:
        return (completed.get("payload") or {}).get("final_wallet") or {}
    if steps:
        return steps[-1].get("wallet_after") or {}
    if failed is not None:
        return {}
    return {}


def _wallet_equity(wallet: dict[str, Any]) -> float | None:
    totals = wallet.get("totals") or {}
    if totals.get("valuation_complete") is False:
        return None
    value = totals.get("equity_usdt")
    return None if value is None else float(value)


def _exchange_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    order = payload.get("order") if isinstance(payload, dict) else None
    cancelled = payload.get("cancelled_order") if isinstance(payload, dict) else None
    position = payload.get("closed_position") if isinstance(payload, dict) else None
    return {
        "type": event.get("type"),
        "timestamp": event.get("timestamp"),
        "symbol": _first_present(payload, order, cancelled, position, key="symbol"),
        "side": _first_present(payload, order, cancelled, position, key="side"),
        "orderType": _first_present(payload, order, cancelled, key="orderType"),
        "status": _first_present(payload, order, cancelled, position, key="status"),
        "exit_reason": _first_present(payload, position, key="exit_reason"),
        "realized_pnl_usdt": _first_present(payload, position, key="realized_pnl_usdt"),
        "ambiguous": _first_present(payload, position, key="ambiguous"),
        "resolution_interval": _first_present(payload, position, key="resolution_interval"),
        "source_interval": _first_present(payload, position, key="source_interval"),
    }


def _first_present(*items: Any, key: str) -> Any:
    for item in items:
        if isinstance(item, dict) and item.get(key) is not None:
            return item[key]
    return None


def _money(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _time_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return str(value)
