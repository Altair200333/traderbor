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
For Codex MCP replay, prefer compact scanner/detail tools and compute_indicators over raw candles. Use run_analysis_code only for custom bounded analysis over tool-loaded candles. Use raw get_candles only when compact/indicator tools are unavailable, failed, or omitted a required strategy fact.
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


def _entry_policy_prompt_section(context: dict[str, Any]) -> str:
    if context.get("entry_policy") != "limit_retest":
        return ""
    pullback = context.get("retest_pullback")
    ttl_min = context.get("retest_ttl_min")
    return (
        "\nRunner entry policy (limit_retest): an accepted entry order is placed as a pending limit "
        f"{pullback} x your stop distance below your entry for longs (above for shorts), valid for {ttl_min} minutes. "
        "The position opens only if price pulls back to that limit before expiry; otherwise the order expires and no position is opened. "
        "This is runner-owned: plan entry, stop, and take-profit exactly as usual at current prices, and do not try to pre-discount the pullback or cancel the pending order.\n"
    )


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
Agent cadence in this runner: candidate-driven by the runner-side deterministic screener. In this call, the runner has already scanned every closed 1h bar up to the exact as_of.
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

Deterministic screener candidates (high-recall triggers, facts only, no trade plan):
{json.dumps(jsonable(context.get("candidate_view", [])), ensure_ascii=False)}

Global blocks:
{json.dumps(jsonable(context.get("global_blocks", [])), ensure_ascii=False)}

Data warnings:
{json.dumps(jsonable(context.get("data_warnings", [])), ensure_ascii=False)}
{_entry_policy_prompt_section(context)}
Closed-candle scan table:
{context.get("scan_markdown", "")}

Use cached market data with this exact as_of.
Use exchange tools for wallet/order actions.
Follow the deterministic replay procedure:
1. Treat the provided scan table, scan hash, and candidate list as the only broad screener result for this step.
2. Do not call scan_momentum_universe. The runner already did the broad scan.
3. Do not call get_candles for broad symbol screening. If a finalist needs more structure, use get_setup_digest for that exact candidate and side, compute_indicators for standard indicator/chart facts, run_analysis_code for custom bounded calculations, and bounded get_candles only when raw rows are still needed. These deep-dive tools are limited to listed candidate symbols plus BTCUSDT, ETHUSDT, and SOLUSDT context anchors.
4. Consider only symbols present in deterministic screener candidates for a new long/short entry.
5. Reconstruct risk state from wallet, settlement, maintenance actions, and recent event helpers if needed. Runner-enforced locks are listed in global blocks; do not invent additional global pauses.
6. Treat deterministic candidates as high-recall screening triggers, not trade recommendations. The screener provides facts only: it does not suggest entry, stop, or take-profit. Build your own trade plan from structure and volatility evidence you gather with the tools.
7. Candidate quality is a gate fact: quality=hard means all S1-S11 gates passed; quality=marginal_extension means only bounded S9b/S9c extension gates failed on a P1/P1H/P3 setup. Marginal extension candidates require deeper confirmation, conservative sizing, and cleaner structure/context.
8. Choose stop and take-profit yourself. Place the stop beyond structural invalidation (broken boundary, nearest support/resistance from the digest) and never tighter than the measured noise floor (noise% in the scan table row; the runner rejects entry orders with a tighter stop). Judge extra stop room from ATR and recent 1h candles you fetch via compute_indicators or bounded get_candles. Target RR >= 3.0 at a realistic TP inside the 24h hold horizon; below RR 2.0 the setup does not pay for its risk - hold. Never tighten a stop to fit risk; if the honest stop breaks the risk bounds, hold. Runner-side risk validation is authoritative: a rejected order means hold or rework, not a tighter stop.
9. Deterministic candidates are pre-selected momentum breakouts: the screener already filtered pattern, regime, and per-symbol dedup on closed bars. Extension, a large last candle, last-hour impulse, or high RSI are expected properties of this flow, not veto reasons - two years of this candidate class show they do not separate winners from losers. A fresh breakout needs no retest confirmation to be tradable. Hold a candidate only for concrete operational reasons: data warnings on its row, stale or implausible price, entry drift beyond the runner clamp, documented illiquidity, or exhausted risk budget / position slots.
10. TP-side levels: for longs overhead resistance is context, not a veto - a fresh breakout is expected to attack overhead levels; prefer TP near 3R and hold only if the first heavily-tested resistance sits closer than 2R. For shorts the first support below is where bounces start - set TP in front of it, and if RR then falls below 2.0, hold.
11. retest_seen counts only closed bars after the breakout trigger bar. retest_seen=false with trigger_age_bars=0 means a retest is not possible yet, not that a retest failed. Retest facts inform stop placement (a held retest marks structure), never entry permission.
12. Shorts are not mirrored longs: crypto downside moves are impulsive and mean-revert fast. Do not short an exhausted move: prefer fresh breakdowns confirmed by retest_seen=true, avoid short entries with RSI near the oversold bound or price extended more than 1.5 ATR below EMA20(1h), and require the TP to be reachable before the first support below.
13. Several candidates on the same bar are usually one market-wide move, not independent bets: open at most the free position slots, keep total concurrent risk inside the budget, prefer the more liquid symbol on close calls, and do not pick solely by the largest recent ROC.
14. Late-entry fallback instead of chasing: if the market entry is rejected for drift beyond the clamp, do not resubmit tweaked numbers. Either hold, or place the entry as a pending limit at the scan reference boundary: place_order with orderType="Limit", price set to your limit, entryPolicy="limit_retest", entryRefPrice equal to that limit price, expiresAtMs at most 2h after as_of, qty = USDT notional / limit price, plus your stopLoss/takeProfit. An expired unfilled limit is a normal no-trade outcome.
Bounded deterministic market-analysis rule: exact as_of is required; intervals are only 1h with limit <= 170 or 4h with limit <= 60; 1m, start_time, and end_time style requests are rejected. run_analysis_code receives only df/candles loaded through this guarded boundary; do not read raw cache/exchange files from scratch code.
For simulator linear orders, TradeDecision.amount and calculate_position_size.amount are USDT notional, but place_order.qty is base-asset quantity: qty = USDT notional / current entry price. Do not pass USDT notional as linear qty.
Trade only valid momentum setups; otherwise return hold.
Never request 1m candles for signal analysis.
Never call close_position, cancel_order, or settle_exchange in deterministic screener mode; runner-side settlement and maintenance already ran before this prompt.
Never call exchange write tools on hold paths.
Read recent relevant worklogs/helpers only if needed before deciding.
Do not do unrelated code exploration during this replay step.
Scratch analysis through run_analysis_code is allowed for bounded decision analysis over tool-provided candles. Use MCP/tool outputs for data; do not read raw cache or exchange state files directly to bypass as_of guards.
Return a structured TradeDecision.
"""
