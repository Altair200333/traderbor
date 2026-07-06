from __future__ import annotations

import json
from typing import Any

from traderbot_ai.decision.serialization import jsonable
from traderbot_ai.runtime.run_context import current_worklog_root


def build_exchange_replay_prompt(context: dict[str, Any]) -> str:
    if context.get("screener_mode") == "deterministic":
        return _build_deterministic_exchange_replay_prompt(context)
    return f"""
Exchange replay step.

Run id: {context["run_id"]}
Symbols: {", ".join(context["symbols"])}
As-of: {context["as_of_iso"]} ({context["as_of_ms"]})
Next step: {context["next_as_of_ms"]}
Decision interval: {context["decision_interval"]}
Execution interval: {context["execution_interval"]}
Fee rate: {context["fee_rate"]}
Agent cadence in this runner: every {context["decision_interval"]}. Production target is candidate-driven; no valid candidate means hold.
Strategy horizon: 4h to 24h swing momentum.
Worklog memory root: {current_worklog_root()}.

Wallet:
{json.dumps(jsonable(context["wallet"]), ensure_ascii=False)}

Settlement just applied:
{json.dumps(jsonable(context["settlement"]), ensure_ascii=False)}

Use cached market data with this exact as_of.
Use exchange tools for wallet/order actions.
Follow the replay step procedure from the system prompt:
1. maintain existing positions first,
2. reconstruct risk state,
3. if available, coarse-scan symbols with scan_momentum_universe; otherwise use 4h candles only,
4. if available, deep-check at most 2 finalists with get_candidate_detail; otherwise use 1h candles,
5. build and validate a plan only for a surviving candidate.
For Codex MCP replay, prefer compact scanner/detail tools over raw candles. Use raw get_candles only when a compact tool is unavailable, failed, or omitted a required strategy fact.
For Codex MCP replay, use get_wallet_compact for the initial wallet read. Use full get_wallet only if compact output is missing a specific fact needed for an entry or maintenance close.
For simulator linear orders, TradeDecision.amount and calculate_position_size.amount are USDT notional, but place_order.qty is base-asset quantity: qty = USDT notional / current entry price. Do not pass USDT notional as linear qty.
Trade only valid momentum setups; otherwise return hold.
Never request 1m candles for signal analysis.
Never call exchange write tools on hold paths except mandatory position-maintenance close_position calls.
Do not call settle_exchange; settlement was already applied by the runner.
Read recent relevant worklogs/helpers only if needed before deciding.
Do not do unrelated code exploration during this replay step.
Return a structured TradeDecision.
"""


def _build_deterministic_exchange_replay_prompt(context: dict[str, Any]) -> str:
    return f"""
Exchange replay step.

Run id: {context["run_id"]}
Symbols: {", ".join(context["symbols"])}
As-of: {context["as_of_iso"]} ({context["as_of_ms"]})
Next step: {context["next_as_of_ms"]}
Decision interval: {context["decision_interval"]}
Execution interval: {context["execution_interval"]}
Fee rate: {context["fee_rate"]}
Screener mode: deterministic.
Agent cadence in this runner: candidate-driven by the runner-side deterministic screener. In this call, the runner has already scanned closed 1h data at the exact as_of.
Strategy horizon: 4h to 24h swing momentum.
Worklog memory root: {current_worklog_root()}.

Wallet:
{json.dumps(jsonable(context["wallet"]), ensure_ascii=False)}

Settlement just applied by runner:
{json.dumps(jsonable(context["settlement"]), ensure_ascii=False)}

Runner-owned maintenance actions already applied before this decision:
{json.dumps(jsonable(context.get("maintenance_actions", [])), ensure_ascii=False)}

Deterministic screener artifact:
- path: {context.get("scan_artifact_path")}
- sha256: {context.get("scan_hash")}

Deterministic screener candidates:
{json.dumps(jsonable(context.get("candidate_primitives", [])), ensure_ascii=False)}

Global blocks:
{json.dumps(jsonable(context.get("global_blocks", [])), ensure_ascii=False)}

Data warnings:
{json.dumps(jsonable(context.get("data_warnings", [])), ensure_ascii=False)}

Closed-candle scan table:
{context.get("scan_markdown", "")}

Use cached market data with this exact as_of.
Use exchange tools for wallet/order actions.
Follow the deterministic replay procedure:
1. Treat the provided scan table, scan hash, and candidate primitives as the only broad screener result for this step.
2. Do not call scan_momentum_universe. The runner already did the broad scan.
3. Do not call raw get_candles for broad symbol screening. If a finalist needs more structure, use get_setup_digest for that exact candidate and side.
4. Consider only symbols present in deterministic screener candidates for a new long/short entry.
5. Reconstruct risk state from wallet, settlement, maintenance actions, and recent event helpers if needed.
6. Build and validate a plan only for a surviving deterministic candidate.
For simulator linear orders, TradeDecision.amount and calculate_position_size.amount are USDT notional, but place_order.qty is base-asset quantity: qty = USDT notional / current entry price. Do not pass USDT notional as linear qty.
Trade only valid momentum setups; otherwise return hold.
Never request 1m candles for signal analysis.
Never call close_position, cancel_order, or settle_exchange in deterministic screener mode; runner-side settlement and maintenance already ran before this prompt.
Never call exchange write tools on hold paths.
Read recent relevant worklogs/helpers only if needed before deciding.
Do not do unrelated code exploration during this replay step.
Return a structured TradeDecision.
"""
