"""Take-every-candidate baseline decision provider (no judgment).

The strategy-ceiling robot for agent-vs-robot A/Bs: it enters every long
candidate the deterministic scan emits, up to the same caps the live agent
prompt imposes (max new positions per bar, notional <= 20% equity, fixed
fractional risk), using the scan plan's geometry (stop at invalidation_price,
TP at tp_rr * d_final above the reference entry). Differences from a codex run
are then attributable to judgment, not to limits.
"""
from __future__ import annotations

from typing import Any

from traderbot_ai.tools.exchange import place_order_impl

RISK_FRACTION = 0.0075
NOTIONAL_CAP_FRACTION = 0.20
MAX_NEW_POSITIONS_PER_BAR = 3


class TakeAllDecisionProvider:
    name = "take-all"

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        as_of_ms = context["as_of_ms"]
        equity = float(((context.get("wallet") or {}).get("totals") or {}).get("equity_usdt") or 0.0)
        primitives = context.get("candidate_primitives") or []
        placed: list[dict[str, Any]] = []
        skipped: list[str] = []
        for item in primitives:
            if len(placed) >= MAX_NEW_POSITIONS_PER_BAR:
                skipped.append("bar-cap-reached")
                break
            if not isinstance(item, dict) or item.get("side") != "long":
                continue
            symbol = str(item.get("symbol") or "")
            plan = item.get("plan") if isinstance(item.get("plan"), dict) else {}
            ref_entry = plan.get("ref_entry")
            stop = plan.get("invalidation_price")
            d_final = plan.get("d_final")
            tp_rr = float(plan.get("tp_rr_default") or 3.0)
            if not symbol or not ref_entry or not stop or not d_final or float(d_final) <= 0:
                skipped.append(f"{symbol}:no-plan")
                continue
            ref_entry = float(ref_entry)
            stop_price = float(stop)
            stop_fraction = float(d_final)
            take_profit = ref_entry * (1.0 + tp_rr * stop_fraction)
            notional = min(RISK_FRACTION * equity / stop_fraction, NOTIONAL_CAP_FRACTION * equity)
            if notional <= 0:
                skipped.append(f"{symbol}:no-funds")
                continue
            qty = notional / ref_entry
            result = place_order_impl(
                category="linear",
                symbol=symbol,
                side="Buy",
                orderType="Market",
                qty=qty,
                takeProfit=take_profit,
                stopLoss=stop_price,
                orderLinkId=f"takeall-{as_of_ms}-{symbol}",
                fee_rate=float(context.get("fee_rate") or 0.0),
                as_of=as_of_ms,
                mark_interval=str(context.get("execution_interval") or "1m"),
            )
            if result.get("ok"):
                placed.append({"result": result, "symbol": symbol, "qty": qty,
                               "stop_loss": stop_price, "take_profit": take_profit,
                               "ref_entry": ref_entry})
            else:
                skipped.append(f"{symbol}:{str(result.get('error'))[:80]}")
        if not placed:
            return {
                "final_decision": "hold",
                "symbol": context["symbols"][0],
                "amount": 0.0,
                "risk_summary": "take-all: no entries (" + ("; ".join(skipped[:5]) or "no long candidates") + ")",
            }
        first = placed[0]
        result = first["result"]
        amount = result.get("notional_usdt")
        if amount is None:
            amount = first["qty"] * first["ref_entry"]
        return {
            "final_decision": "long",
            "symbol": first["symbol"],
            "price": result.get("avg_price") or result.get("price"),
            "stop_loss": first["stop_loss"],
            "take_profit": first["take_profit"],
            "amount": float(amount),
            "confidence": 1.0,
            "thesis": "take-all baseline: enter every long candidate up to caps",
            "risk_summary": f"take-all: entered {len(placed)} of {len(primitives)} candidates; skipped {len(skipped)}",
        }

    def close(self) -> None:
        return None
