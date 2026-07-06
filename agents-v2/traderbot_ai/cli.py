from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from agents import Runner

from traderbot_ai.agents.trading import build_tools, build_trading_agent
from traderbot_ai.config import load_settings
from traderbot_ai.decision import CodexCliMcpDecisionProvider, CodexCliOptions, HoldDecisionProvider, OpenAIAgentsDecisionProvider
from traderbot_ai.decision.serialization import jsonable
from traderbot_ai.providers import provider_status
from traderbot_ai.runtime.run_store import write_run_record
from traderbot_ai.runtime.run_context import agent_run_context, current_worklog_root
from traderbot_ai.runtime.sessions import get_session
from traderbot_ai.simulator.backtest import build_config, run_backtest
from traderbot_ai.simulator.exchange_replay import build_exchange_replay_config, run_exchange_replay
from traderbot_ai.simulator.replay_report import build_replay_report, write_replay_markdown_report
from traderbot_ai.tools.charts import store_chart_file
from traderbot_ai.tools.simulator import (
    get_market_cache_status_impl,
    preload_market_cache_impl,
    simulate_order_exit_impl,
)
from traderbot_ai.tools.market import parse_time_ms
from traderbot_ai.tools.workspace import append_worklog_record


def _dump(value: Any) -> None:
    value = _jsonable(value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _jsonable(value: Any) -> Any:
    return jsonable(value)


def _close_session(session: Any) -> None:
    close = getattr(session, "close", None)
    if callable(close):
        close()


def cmd_providers(args: argparse.Namespace) -> None:
    _dump([status.model_dump(mode="json") for status in provider_status(live=args.live)])


def cmd_tools(args: argparse.Namespace) -> None:
    settings = load_settings()
    tools = build_tools(settings)
    _dump(
        [
            {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", None),
            }
            for tool in tools
        ]
    )


def cmd_chart(args: argparse.Namespace) -> None:
    _dump(store_chart_file(symbol=args.symbol, interval=args.interval, limit=args.limit))


def cmd_market_cache(args: argparse.Namespace) -> None:
    if args.status:
        _dump(get_market_cache_status_impl())
        return
    _dump(
        preload_market_cache_impl(
            symbols=args.symbols,
            intervals=args.intervals,
            start_time=args.start_time,
            end_time=args.end_time,
            max_candles_per_pair=args.max_candles_per_pair,
            include_agg_trades=args.agg_trades,
            max_agg_trades_per_symbol=args.max_agg_trades_per_symbol,
        )
    )


def cmd_simulate_order(args: argparse.Namespace) -> None:
    _dump(
        simulate_order_exit_impl(
            final_decision=args.final_decision,
            symbol=args.symbol,
            price=args.price,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            amount=args.amount,
            opened_at=args.opened_at,
            scan_until=args.scan_until,
            interval=args.interval,
            fee_rate=args.fee_rate,
        )
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    if not args.no_preload or args.preload:
        preload = preload_market_cache_impl(
            symbols=args.symbol,
            intervals=args.cache_intervals,
            start_time=args.start_time,
            end_time=args.end_time,
            max_candles_per_pair=args.max_candles_per_pair,
            include_agg_trades=args.agg_trades,
            max_agg_trades_per_symbol=args.max_agg_trades_per_symbol,
        )
        if not preload.get("ok"):
            _dump(preload)
            return

    config = build_config(
        symbol=args.symbol,
        start_time=args.start_time,
        end_time=args.end_time,
        decision_interval=args.decision_interval,
        execution_interval=args.execution_interval,
        balance_usdt=args.balance_usdt,
        fee_rate=args.fee_rate,
    )
    settings = replace(load_settings(), market_data_mode="cache")
    agent = build_trading_agent(settings, backtest=True)
    session_name = args.session or f"backtest-{config.symbol.lower()}-{config.start_ms}-{config.end_ms}-{uuid.uuid4().hex[:8]}"
    session = get_session(session_name, settings=settings)

    def decide(context: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""
Offline backtest step.

Symbol: {context["symbol"]}
As-of: {context["as_of_iso"]} ({context["as_of_ms"]})
Decision interval: {context["decision_interval"]}
Execution interval: {context["execution_interval"]}
Agent cadence in this runner: every {context["decision_interval"]}. Production target is candidate-driven; no valid candidate means hold.
Strategy horizon: 4h to 24h swing momentum.
Worklog memory root: {current_worklog_root()}.
Simulated portfolio summary:
{json.dumps(_jsonable(context["portfolio"]), ensure_ascii=False)}

Use cached market tools only. Use as_of exactly.
Trade only valid momentum setups; otherwise return hold.
Read existing notes/helpers if needed, but do not write worklogs during backtests.
If a reusable screener/helper exists and its outputs are available, prefer it over manual one-off calculations.
Return a structured TradeDecision.
"""
        result = Runner.run_sync(agent, prompt, session=session, max_turns=args.max_turns)
        return _jsonable(result.final_output)

    try:
        _dump(run_backtest(config=config, decide=decide))
    finally:
        _close_session(session)


def cmd_exchange_replay(args: argparse.Namespace) -> None:
    if not args.no_preload or args.preload:
        preload_start_time: str | int | float | None = args.start_time
        if args.cache_lookback_hours and args.cache_lookback_hours > 0:
            start_ms = parse_time_ms(args.start_time)
            if start_ms is not None:
                preload_start_time = start_ms - int(args.cache_lookback_hours * 60 * 60_000)
        preload = preload_market_cache_impl(
            symbols=args.symbols,
            intervals=args.cache_intervals,
            start_time=preload_start_time,
            end_time=args.end_time,
            max_candles_per_pair=args.max_candles_per_pair,
            include_agg_trades=args.agg_trades,
            max_agg_trades_per_symbol=args.max_agg_trades_per_symbol,
        )
        if not preload.get("ok"):
            _dump(preload)
            return

    balances = json.loads(args.balances_json) if args.balances_json else None
    if balances is not None and not isinstance(balances, dict):
        _dump({"ok": False, "error": "balances_json must be an object"})
        return
    config = build_exchange_replay_config(
        symbols=args.symbols,
        start_time=args.start_time,
        end_time=args.end_time,
        decision_interval=args.decision_interval,
        execution_interval=args.execution_interval,
        balance_usdt=args.balance_usdt,
        balances=None if balances is None else {str(key): float(value) for key, value in balances.items()},
        fee_rate=args.fee_rate,
        linear_leverage=args.linear_leverage,
        run_id=args.run_id,
        state_path=args.state_path,
        events_path=args.events_path,
        replay_path=args.replay_path,
        screener_mode=args.screener_mode,
    )
    provider_name = args.decision_provider or ("hold" if args.decision_mode == "hold" else "openai-agents")
    if args.decision_mode == "hold" and provider_name != "hold":
        _dump({"ok": False, "error": "--decision-mode hold conflicts with --decision-provider"})
        return
    if provider_name == "hold":
        provider = HoldDecisionProvider()
    elif provider_name == "openai-agents":
        settings = replace(load_settings(), market_data_mode="cache", enable_codex_tool=False)
        session_name = args.session or f"{config.run_id}-agent"
        provider = OpenAIAgentsDecisionProvider(settings=settings, session_name=session_name, max_turns=args.max_turns, screener_mode=config.screener_mode)
    elif provider_name == "codex-cli-mcp":
        if args.codex_use_existing_mcp_config and not args.codex_use_user_config:
            _dump({"ok": False, "error": "--codex-use-existing-mcp-config requires --codex-use-user-config"})
            return
        settings = replace(load_settings(), market_data_mode="cache", enable_codex_tool=False)
        session_name = args.session or f"{config.run_id}-codex"
        provider = CodexCliMcpDecisionProvider(
            settings=settings,
            session_name=session_name,
            screener_mode=config.screener_mode,
            options=CodexCliOptions(
                model=args.codex_model,
                reasoning_effort=args.codex_reasoning_effort,
                profile=args.codex_profile,
                sandbox=args.codex_sandbox,
                timeout_sec=args.codex_timeout_sec,
                output_dir=args.codex_output_dir,
                fail_open=args.codex_fail_open,
                codex_executable=args.codex_executable,
                python_executable=args.codex_mcp_python,
                ignore_user_config=not args.codex_use_user_config,
                full_auto=not args.codex_no_full_auto,
                include_mcp_config=not args.codex_use_existing_mcp_config,
            ),
        )
    else:
        _dump({"ok": False, "error": f"unsupported decision provider: {provider_name}"})
        return

    try:
        _dump(run_exchange_replay(config=config, decide=provider.decide))
    finally:
        provider.close()


def cmd_replay_report(args: argparse.Namespace) -> None:
    try:
        if args.markdown_out:
            _dump(
                write_replay_markdown_report(
                    replay_path=args.replay_path,
                    output_path=args.markdown_out,
                    exchange_events_path=args.exchange_events_path,
                )
            )
            return
        _dump(build_replay_report(replay_path=args.replay_path, exchange_events_path=args.exchange_events_path))
    except Exception as error:
        _dump({"ok": False, "error": str(error), "error_type": type(error).__name__})


def cmd_worklog(args: argparse.Namespace) -> None:
    _dump(append_worklog_record(args.markdown, run_id=args.run_id))


def cmd_run(args: argparse.Namespace) -> None:
    settings = load_settings()
    run_id = args.run_id or args.session
    with agent_run_context(run_id):
        worklog_root = current_worklog_root()
        agent = build_trading_agent(settings)
        session = get_session(args.session, settings=settings)
        try:
            result = Runner.run_sync(
                agent,
                args.prompt,
                session=session,
                max_turns=args.max_turns,
            )
        finally:
            _close_session(session)
    run_log = write_run_record(args.session, args.prompt, result)
    final_output = result.final_output
    _dump(
        {
            "session": args.session,
            "run_id": run_id,
            "worklog_root": worklog_root,
            "run_log": run_log,
            "final_output": final_output,
        }
    )


def cmd_history(args: argparse.Namespace) -> None:
    session = get_session(args.session)
    try:
        items = asyncio.run(session.get_items(limit=args.limit))
    finally:
        _close_session(session)
    _dump({"session": args.session, "items": items})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traderbot Agents V2 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    providers = sub.add_parser("providers", help="Check configured model providers.")
    providers.add_argument("--live", action="store_true", help="Make a small live OpenAI API probe.")
    providers.set_defaults(func=cmd_providers)

    tools = sub.add_parser("tools", help="List tools exposed to the agent.")
    tools.set_defaults(func=cmd_tools)

    chart = sub.add_parser("chart", help="Render and store a chart without running the agent.")
    chart.add_argument("--symbol", default="BTCUSDT")
    chart.add_argument("--interval", default="15m")
    chart.add_argument("--limit", type=int, default=120)
    chart.set_defaults(func=cmd_chart)

    market_cache = sub.add_parser("market-cache", help="Preload or inspect the local market cache.")
    market_cache.add_argument("--symbols", default="BTCUSDT", help="CSV symbols, e.g. BTCUSDT,ETHUSDT.")
    market_cache.add_argument("--intervals", default="1m,1h,4h", help="CSV intervals, e.g. 1m,1h,4h.")
    market_cache.add_argument("--start-time")
    market_cache.add_argument("--end-time")
    market_cache.add_argument("--max-candles-per-pair", type=int)
    market_cache.add_argument("--agg-trades", action="store_true", help="Also cache Binance aggregate trades for the requested time range.")
    market_cache.add_argument("--max-agg-trades-per-symbol", type=int)
    market_cache.add_argument("--status", action="store_true")
    market_cache.set_defaults(func=cmd_market_cache)

    simulate_order = sub.add_parser("simulate-order", help="Simulate TP/SL exit against cached candles.")
    simulate_order.add_argument("--final-decision", choices=["long", "short"], required=True)
    simulate_order.add_argument("--symbol", required=True)
    simulate_order.add_argument("--price", type=float, required=True)
    simulate_order.add_argument("--stop-loss", type=float, required=True)
    simulate_order.add_argument("--take-profit", type=float, required=True)
    simulate_order.add_argument("--amount", type=float, required=True)
    simulate_order.add_argument("--opened-at", required=True)
    simulate_order.add_argument("--scan-until", required=True)
    simulate_order.add_argument("--interval", default="1m")
    simulate_order.add_argument("--fee-rate", type=float, default=0.0)
    simulate_order.set_defaults(func=cmd_simulate_order)

    backtest = sub.add_parser("backtest", help="Run the agent through cached historical data.")
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--start-time", required=True)
    backtest.add_argument("--end-time", required=True)
    backtest.add_argument("--decision-interval", default="4h")
    backtest.add_argument("--execution-interval", default="1m")
    backtest.add_argument("--cache-intervals", default="1m,1h,4h")
    backtest.add_argument("--balance-usdt", type=float, default=1000.0)
    backtest.add_argument("--fee-rate", type=float, default=0.0)
    backtest.add_argument("--session")
    backtest.add_argument("--max-turns", type=int, default=12)
    backtest.add_argument("--preload", action="store_true", help="Preload cache before running. This is the default unless --no-preload is set.")
    backtest.add_argument("--no-preload", action="store_true", help="Use the existing local cache without fetching first.")
    backtest.add_argument("--max-candles-per-pair", type=int)
    backtest.add_argument("--agg-trades", action="store_true", help="Also preload aggregate trades for trade-level ambiguous-candle resolution.")
    backtest.add_argument("--max-agg-trades-per-symbol", type=int)
    backtest.set_defaults(func=cmd_backtest)

    exchange_replay = sub.add_parser("exchange-replay", help="Run a multi-symbol exchange-wallet replay with cached data.")
    exchange_replay.add_argument("--symbols", default="BTCUSDT,ETHUSDT,XRPUSDT")
    exchange_replay.add_argument("--start-time", required=True)
    exchange_replay.add_argument("--end-time", required=True)
    exchange_replay.add_argument("--decision-interval", default="4h")
    exchange_replay.add_argument("--execution-interval", default="1m")
    exchange_replay.add_argument("--cache-intervals", default="1m,1h,4h")
    exchange_replay.add_argument("--cache-lookback-hours", type=float, default=240.0, help="Extra market-cache history to preload before start-time for agent indicators.")
    exchange_replay.add_argument("--balance-usdt", type=float, default=1000.0)
    exchange_replay.add_argument("--balances-json", help='Optional wallet balances, e.g. {"USDT": 1000, "BTC": 0.01}.')
    exchange_replay.add_argument("--fee-rate", type=float, default=0.0)
    exchange_replay.add_argument("--linear-leverage", type=float)
    exchange_replay.add_argument("--run-id")
    exchange_replay.add_argument("--state-path")
    exchange_replay.add_argument("--events-path")
    exchange_replay.add_argument("--replay-path")
    exchange_replay.add_argument("--session")
    exchange_replay.add_argument("--max-turns", type=int, default=12)
    exchange_replay.add_argument("--decision-mode", choices=["agent", "hold"], default="agent", help="Compatibility alias. Use --decision-provider for new runs.")
    exchange_replay.add_argument("--decision-provider", choices=["openai-agents", "codex-cli-mcp", "hold"], help="Decision provider for replay.")
    exchange_replay.add_argument("--screener-mode", choices=["off", "deterministic", "legacy-self-screen"], default="off", help="Deterministic runner-owned screener mode. Default preserves legacy provider-side screening.")
    exchange_replay.add_argument("--codex-model", help="Model for --decision-provider codex-cli-mcp. If omitted, Codex CLI chooses its configured/default model.")
    exchange_replay.add_argument("--codex-reasoning-effort", choices=["low", "medium", "high", "xhigh"], help="Codex model_reasoning_effort config override.")
    exchange_replay.add_argument("--codex-profile", help="Codex config profile for --decision-provider codex-cli-mcp.")
    exchange_replay.add_argument("--codex-sandbox", default="read-only", choices=["read-only", "workspace-write", "danger-full-access"])
    exchange_replay.add_argument("--codex-timeout-sec", type=int, default=1800)
    exchange_replay.add_argument("--codex-output-dir", type=Path, default=None)
    exchange_replay.add_argument("--codex-fail-open", choices=["error", "hold"], default="error")
    exchange_replay.add_argument("--codex-executable", default="codex")
    exchange_replay.add_argument("--codex-mcp-python", default=sys.executable)
    exchange_replay.add_argument("--codex-no-full-auto", action="store_true", help="Disable Codex --full-auto. Not recommended for noninteractive MCP replay.")
    exchange_replay.add_argument("--codex-use-user-config", action="store_true", help="Load ~/.codex/config.toml instead of --ignore-user-config.")
    exchange_replay.add_argument("--codex-use-existing-mcp-config", action="store_true", help="Do not inject the local traderbot MCP server config.")
    exchange_replay.add_argument("--preload", action="store_true", help="Preload cache before running. This is the default unless --no-preload is set.")
    exchange_replay.add_argument("--no-preload", action="store_true", help="Use existing local cache without fetching first.")
    exchange_replay.add_argument("--max-candles-per-pair", type=int)
    exchange_replay.add_argument("--agg-trades", action="store_true", help="Also preload aggregate trades for the requested time range.")
    exchange_replay.add_argument("--max-agg-trades-per-symbol", type=int)
    exchange_replay.set_defaults(func=cmd_exchange_replay)

    replay_report = sub.add_parser("replay-report", help="Inspect an exchange replay JSONL log.")
    replay_report.add_argument("--replay-path", required=True)
    replay_report.add_argument("--exchange-events-path")
    replay_report.add_argument("--markdown-out", help="Write a human-readable markdown report.")
    replay_report.set_defaults(func=cmd_replay_report)

    worklog = sub.add_parser("worklog", help="Append to worklog/YYYY-MM-DD-record.md.")
    worklog.add_argument("markdown")
    worklog.add_argument("--run-id", help="Write under worklog/runs/<run-id>.")
    worklog.set_defaults(func=cmd_worklog)

    run = sub.add_parser("run", help="Run the trading agent.")
    run.add_argument("--session", default="default")
    run.add_argument("--run-id", help="Worklog run id. Defaults to --session.")
    run.add_argument("--prompt", required=True)
    run.add_argument("--max-turns", type=int, default=12)
    run.set_defaults(func=cmd_run)

    history = sub.add_parser("history", help="Show stored local session items.")
    history.add_argument("--session", default="default")
    history.add_argument("--limit", type=int, default=20)
    history.set_defaults(func=cmd_history)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
