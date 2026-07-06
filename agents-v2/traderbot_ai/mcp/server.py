from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from traderbot_ai.mcp import tools


mcp = FastMCP("traderbot")

mcp.tool()(tools.scan_momentum_universe)
mcp.tool()(tools.get_candidate_detail)
mcp.tool()(tools.get_wallet_compact)
mcp.tool()(tools.get_open_positions)
mcp.tool()(tools.get_recent_trade_events)
mcp.tool()(tools.get_cache_status_compact)
mcp.tool()(tools.get_candles)
mcp.tool()(tools.get_current_price)
mcp.tool()(tools.get_market_cache_status)
mcp.tool()(tools.simulate_order_exit)
mcp.tool()(tools.validate_order)
mcp.tool()(tools.calculate_position_size)
mcp.tool()(tools.get_wallet)
mcp.tool()(tools.set_leverage)
mcp.tool()(tools.place_order)
mcp.tool()(tools.cancel_order)
mcp.tool()(tools.close_position)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
