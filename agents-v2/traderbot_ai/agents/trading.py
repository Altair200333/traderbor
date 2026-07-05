from __future__ import annotations

from agents import Agent, ModelSettings, function_tool, tool_namespace
from openai.types.shared import Reasoning

from traderbot_ai.config import Settings, load_settings
from traderbot_ai.paths import PROJECT_ROOT
from traderbot_ai.schemas import TradeDecision
from traderbot_ai.tools.charts import list_stored_charts, load_chart_metadata, store_chart
from traderbot_ai.tools import market as live_market
from traderbot_ai.tools import simulator as simulator_tools
from traderbot_ai.tools.exchange import cancel_order, close_position, get_wallet, place_order, reset_exchange, set_leverage, settle_exchange
from traderbot_ai.tools.portfolio import get_current_balance, paper_place_order, reset_paper_portfolio
from traderbot_ai.tools.risk import calculate_position_size, validate_order
from traderbot_ai.tools.vision import inspect_image
from traderbot_ai.tools.workspace import (
    append_worklog,
    list_local_files,
    read_local_file,
    run_python_code,
    search_local_files,
    write_local_file,
)


def _maybe_codex_tool(settings: Settings):
    if not settings.enable_codex_tool:
        return []
    try:
        from agents.extensions.experimental.codex.codex_tool import codex_tool
    except Exception:
        return []
    try:
        return [
            codex_tool(
                name="codex_code_worker",
                description=(
                    "Ask Codex to inspect or modify code in the traderbor workspace. "
                    "Use for coding tasks that are too large for simple file tools."
                ),
                working_directory=str(PROJECT_ROOT),
                skip_git_repo_check=True,
                persist_session=True,
            )
        ]
    except Exception:
        return []


def build_tools(settings: Settings | None = None, *, backtest: bool = False, exchange_replay: bool = False):
    settings = settings or load_settings()
    tools = []
    market_tools = (
        [simulator_tools.get_candles, simulator_tools.get_current_price]
        if backtest or exchange_replay or settings.market_data_mode == "cache"
        else [
            live_market.get_candles,
            live_market.get_current_price,
            live_market.get_order_book,
            live_market.save_market_artifact,
        ]
    )
    simulator_namespace_tools = [
        simulator_tools.get_market_cache_status,
        simulator_tools.simulate_order_exit,
    ]
    if not backtest and not exchange_replay:
        simulator_namespace_tools = [
            simulator_tools.preload_market_cache,
            *simulator_namespace_tools,
            simulator_tools.get_simulation_portfolio,
            simulator_tools.reset_simulation,
            simulator_tools.place_simulated_order,
            simulator_tools.settle_simulation,
        ]
    if settings.market_data_mode != "cache" and not exchange_replay:
        simulator_namespace_tools.append(simulator_tools.get_cached_candles)
    workspace_tools = (
        [list_local_files, read_local_file, search_local_files]
        if backtest or exchange_replay
        else [list_local_files, read_local_file, search_local_files, write_local_file, append_worklog, run_python_code]
    )
    tools.extend(
        tool_namespace(
            name="workspace",
            description="Local project file tools.",
            tools=workspace_tools,
        )
    )
    tools.extend(
        tool_namespace(
            name="market",
            description="Read-only market data tools.",
            tools=market_tools,
        )
    )
    tools.extend(
        tool_namespace(
            name="simulator",
            description="Local cached market data and deterministic TP/SL simulation tools.",
            tools=simulator_namespace_tools,
        )
    )
    if not backtest and not exchange_replay:
        tools.extend(
            tool_namespace(
                name="charts",
                description="Chart rendering and chart metadata tools.",
                tools=[store_chart, load_chart_metadata, list_stored_charts],
            )
        )
        tools.extend(
            tool_namespace(
                name="vision",
                description="Image inspection tools.",
                tools=[inspect_image],
            )
        )
    if not backtest:
        exchange_tools = [get_wallet, set_leverage, place_order, cancel_order, settle_exchange, close_position]
        if not exchange_replay:
            exchange_tools.append(reset_exchange)
        tools.extend(
            tool_namespace(
                name="exchange",
                description="Generic exchange wallet tools. Simulator now; Bybit adapter later.",
                tools=exchange_tools,
            )
        )
    if not backtest and not exchange_replay:
        tools.extend(
            tool_namespace(
                name="portfolio",
                description="Local paper portfolio and paper order tools.",
                tools=[get_current_balance, reset_paper_portfolio, paper_place_order],
            )
        )
    tools.extend(
        tool_namespace(
            name="risk",
            description="Deterministic risk and sizing validation tools.",
            tools=[validate_order, calculate_position_size],
        )
    )
    if not backtest and not exchange_replay:
        tools.extend(_maybe_codex_tool(settings))
    return tools


def build_trading_agent(settings: Settings | None = None, *, backtest: bool = False, exchange_replay: bool = False) -> Agent:
    settings = settings or load_settings()
    model_settings = ModelSettings(
        reasoning=Reasoning(effort=settings.reasoning_effort),
        verbosity="low",
        parallel_tool_calls=True,
        prompt_cache_retention="24h",
        include_usage=True,
        max_tokens=2000,
    )
    mode_instructions = (
        """
Offline backtest mode:
- The runner owns portfolio writes and order settlement.
- Use read-only cached market tools, the portfolio summary in the prompt, simulate_order_exit, and risk tools.
- Do not try to place, reset, settle, or inspect orders with portfolio-write tools.
- Do not write worklogs during backtests.
- Return the structured TradeDecision; the runner will fill entries at the next execution-candle open.
"""
        if backtest
        else """
Exchange replay mode:
- Use cached market tools only. Never use live market assumptions.
- Use exchange tools as the only wallet/order interface: get_wallet, set_leverage, place_order, cancel_order, settle_exchange, close_position.
- The runner owns reset and clock movement. Do not ask to reset the exchange.
- Pass the exact as_of from the prompt to every exchange write tool.
- Use the fee_rate from the prompt on place_order, settle_exchange, and close_position when modeling fees.
- Before placing a new order, inspect get_wallet and cached market data for the relevant symbol.
- Return the structured TradeDecision even if you also used exchange tools.
"""
        if exchange_replay
        else """
Live research / paper mode:
- Use exchange tools as the primary wallet/order interface: get_wallet, set_leverage, place_order, cancel_order, settle_exchange, close_position.
- Before proposing long or short, call get_wallet, calculate_position_size, and validate_order.
- For replay/simulator workflows, pass as_of to every exchange write tool and use the same fee_rate on place_order, settle_exchange, and close_position when modeling fees.
- Call settle_exchange before get_wallet or close_position at a later as_of when you need an explicit settlement event.
- Use legacy paper portfolio tools only when the user explicitly asks for the old paper workflow.
- Never claim a live exchange order was placed; exchange tools currently run against the local simulator.
- Write a concise markdown worklog with append_worklog before final output.
- Store charts with store_chart when chart context is useful.
- Use codex_code_worker only for larger code changes.
"""
    )
    instructions = f"""
You are Traderbot V2, a local research and paper-trading agent.

You run in the user's local traderbor workspace.
Default model: {settings.model}.
Default reasoning effort: {settings.reasoning_effort}.

Operating rules:
- Use tools for live facts. Do not invent prices, balances, files, or chart paths.
- Use simulator tools for offline as-of market reads, cached candles, and TP/SL execution checks.
- In offline simulations, use exchange wallet/order tools when order state, balances, leverage, or replayable fills matter.
- For long: stop_loss < price < take_profit.
- For short: take_profit < price < stop_loss.
- Use at most 30% of paper balance unless the user explicitly changes risk settings.
- Use workspace tools for local notes and code snippets.
- Final output must be a structured TradeDecision.
{mode_instructions}

If market data tools fail, return hold and explain the failure in risk_summary.
"""
    return Agent(
        name="Traderbot V2",
        instructions=instructions,
        model=settings.model,
        model_settings=model_settings,
        tools=build_tools(settings, backtest=backtest, exchange_replay=exchange_replay),
        output_type=TradeDecision,
    )
