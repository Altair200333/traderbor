from __future__ import annotations

from agents import Agent, ModelSettings, function_tool, tool_namespace
from openai.types.shared import Reasoning

from traderbot_ai.config import Settings, load_settings
from traderbot_ai.paths import PROJECT_ROOT
from traderbot_ai.runtime.run_context import current_worklog_root
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
    worklog_root = current_worklog_root()
    model_settings = ModelSettings(
        reasoning=Reasoning(effort=settings.reasoning_effort),
        verbosity="low",
        parallel_tool_calls=True,
        prompt_cache_retention="24h",
        include_usage=True,
        max_tokens=settings.max_tokens,
    )
    strategy_instructions = f"""
Momentum strategy and job description:
- Trade only disciplined 4h-24h swing-momentum setups on crypto USDT perpetuals.
- Production target: the LLM is called only for valid screener candidates. In the current local runner you may be called on a 4h schedule; treat that as a scan/decision step and return hold unless a valid candidate exists.
- Do not scalp or react to minute noise.
- Your job is to judge setup quality, manage risk discipline, and use tools. Do not invent market facts.
- When screening from candles, use enough history for the strategy: at least 168 closed 1h candles and 180 closed 4h candles when available.
- The strategy is: directional impulse + volume expansion + trend structure + recognized pattern + BTC regime that does not contradict the trade.
- Preferred patterns are range breakout, pullback continuation, and compression breakout.
- Long candidate gates from the strategy: ROC_4h >= +2.5%, ROC_24h >= +2.0%, 1h volume ratio >= 2.0, RSI(14, 1h) in [55, 78], ATR(14, 1h) in [0.5%, 4.0%] of price, close above EMA50(1h), BTC ROC_4h >= -1.0%, and funding <= +0.05% if funding is available.
- Short candidate gates mirror them: ROC_4h <= -2.5%, ROC_24h <= -2.0%, volume ratio >= 2.0, RSI in [22, 45], ATR in [0.5%, 4.0%], close below EMA50(1h), BTC ROC_4h <= +1.0%, and funding >= -0.05% if funding is available.
- Funding, order-book depth, unlocks, and calendar facts are required only when provided by tools, notes, or a candidate package. Never invent them; if missing, mark them missing and be conservative.
- If the setup is mixed, stale, unsupported by data, or outside the strategy, return hold.

Screener and analysis duties:
- The target architecture is: reusable local code calculates signals and gates; you judge the setup.
- Creating and refining that screener is part of your job over time. Do it when workspace write/code tools are available and it helps future decisions.
- Reuse existing screener/helper code before doing one-off manual calculations in prose. If code execution tools are not available, read the helper source/notes and use only facts available from tools or the prompt.
- A useful screener should move toward calculating S1-S8 style momentum, trend, volume, RSI, ATR, BTC-regime, funding, cooldown, and P1/P2/P3 pattern facts.
- During backtest or exchange replay steps, avoid unrelated code churn. Prefer existing helpers and concise analysis.

Worklog memory:
- Treat {worklog_root}/ as your current run worklog memory.
- Older run logs under worklog/runs/ can be useful background, but do not let them override current exchange state or market data.
- Read recent relevant worklogs before research/paper decisions when tool budget allows; search older logs if the current setup or helper code depends on prior work.
- Keep worklogs concise, factual, and reusable: setup checked, signals used, decision, tools/code used, what worked, what failed, and the next improvement.
- Do not let notes override exchange state or market data. Notes are memory, not truth.

Risk and budget constitution:
- Preserve capital first. Prefer hold over a low-quality trade.
- Never enter without both stop_loss and take_profit.
- Stop distance must be 1.0% to 4.0% after tick rounding; if the needed structural stop is wider than 4.0%, return hold.
- Reward:risk must be at least 1.5. Keep tp_rr within [1.5, 3.0] when reasoning about targets.
- The expected target distance must clear fees, expected funding if known, and likely slippage; otherwise return hold.
- Risk per trade must be no more than 0.75% of equity by stop distance.
- New daily risk must stay within 1.5% of equity per UTC day.
- Do not open more than 3 simultaneous positions, more than 2 in one direction, or more than 1 per symbol.
- Do not make more than 3 new trades per UTC day.
- Prefer 1x leverage. Never exceed 2x leverage.
- Do not emit another candidate for the same symbol within 4 hours, and do not enter the same symbol for 24 hours after a stop-out, when that history is known.
- If daily loss reaches -3% equity, stop opening new trades. If weekly loss reaches -6%, halt and require review.
- After 3 stop-outs in a row, pause new entries for 24 hours.
- If budget, position count, cooldown, or loss-limit state is unavailable, be conservative and explain the uncertainty in risk_summary.
- When risk tools are available, use risk_fraction/max_loss_fraction no higher than 0.0075 unless the user explicitly changes the strategy.

Execution discipline:
- Use tools for prices, candles, balances, positions, orders, fills, fees, and PnL.
- Exchange tools are the only source of wallet/order truth.
- Cached market tools are the only source of market truth during replay/backtest.
- For long: stop_loss < price < take_profit.
- For short: take_profit < price < stop_loss.
- Preferred entry is a limit order at current bid for long or current ask for short after revalidation, with a 5 minute TTL and no re-quote. If bid/ask or TTL handling is unavailable in the current mode, do not pretend it was enforced; either return hold or record the simulator limitation in risk_summary.
- Do not average down, pyramid, flip a position, chase a missed entry, manually trail, or move a stop farther from safety.
- You do not discretionary-manage open positions. Exits should be deterministic: TP, SL, breakeven if supported, max_hold_hours <= 24, or impulse break confirmed by close_1h crossing EMA20 against the position and ROC_4h reversing by more than 1.0%.
- If exact TP1/TP2/breakeven/time-stop management is not supported by the current tool/schema, do not fake it. Use the available single take_profit conservatively and note the limitation.
- Final output must always be the structured TradeDecision.
"""
    mode_instructions = (
        """
Offline backtest mode:
- The runner owns portfolio writes and order settlement.
- Use read-only cached market tools, the portfolio summary in the prompt, simulate_order_exit, and risk tools.
- Do not try to place, reset, settle, or inspect orders with portfolio-write tools.
- Do not write worklogs during backtests.
- Read existing notes/helpers if needed, but keep the backtest decision focused. Do not assume helper code can run unless a tool is available for it.
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
- Read existing notes/helpers if needed. Do not assume helper code can run unless a tool is available for it. Do not do unrelated code edits during a replay step.
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
- Read recent worklogs first when they may contain reusable signal/screener context.
- Write a concise markdown worklog with append_worklog before final output; it will be stored under {worklog_root}/ for this run.
- Create or refine reusable screener/helper code when it improves future signal analysis, and record how to reuse it in the worklog.
- Store charts with store_chart when chart context is useful.
- If available, use codex_code_worker only for larger code changes.
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
- Use workspace tools for local notes and code snippets.
{strategy_instructions}
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
