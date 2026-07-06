from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from traderbot_ai.exchange import SimulatedExchange
from traderbot_ai.exchange.simulated import CancelOrderError
from traderbot_ai.simulator.market_cache import Candle, LocalMarketCache


SYMBOL = "BTCUSDT"
BASE_MS = 1_704_067_200_000


def candle(symbol: str, open_time: int, open_price: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        symbol=symbol,
        interval="1m",
        open_time=open_time,
        close_time=open_time + 59_999,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class SimulatedExchangeTests(unittest.TestCase):
    def test_wallet_summary_converts_balances_to_usdt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)

            exchange.reset({"USDT": 1000.0, "BTC": 0.1}, as_of=BASE_MS)
            summary = exchange.wallet_summary(symbols=SYMBOL, as_of=BASE_MS + 59_999)

            btc = next(item for item in summary["balances"] if item["asset"] == "BTC")
            self.assertEqual(btc["usdt_rate"], 50_000.0)
            self.assertEqual(btc["total_usdt"], 5000.0)
            self.assertEqual(summary["totals"]["free_usdt"], 6000.0)
            self.assertEqual(summary["totals"]["wallet_usdt"], 6000.0)

    def test_reset_accepts_explicit_empty_wallet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl")

            state = exchange.reset({}, as_of=BASE_MS)

            self.assertEqual(state["balances"], {})

    def test_load_missing_state_does_not_truncate_existing_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            events_path.write_text('{"type": "old"}\n', encoding="utf-8")
            exchange = SimulatedExchange(Path(tmp) / "missing.json", events_path)

            state = exchange.load()

            self.assertEqual(state["balances"]["USDT"]["free"], 1000.0)
            self.assertEqual(events_path.read_text(encoding="utf-8"), '{"type": "old"}\n')

    def test_spot_limit_buy_locks_and_cancel_releases_usdt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            placed = exchange.place_order(
                category="spot",
                symbol=SYMBOL,
                side="Buy",
                orderType="Limit",
                qty=0.01,
                price=50_000.0,
                orderLinkId="spot-buy-1",
                as_of=BASE_MS,
            )
            state = placed["state"]
            self.assertEqual(state["balances"]["USDT"]["free"], 500.0)
            self.assertEqual(state["balances"]["USDT"]["locked"], 500.0)

            cancelled = exchange.cancel_order(orderLinkId="spot-buy-1", as_of=BASE_MS)
            self.assertEqual(cancelled["state"]["balances"]["USDT"]["free"], 1000.0)
            self.assertEqual(cancelled["state"]["balances"]["USDT"]["locked"], 0.0)
            self.assertEqual(cancelled["cancelled_order"]["status"], "Cancelled")

    def test_cancel_order_can_match_by_bybit_category_and_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl")
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, orderLinkId="bybit-cancel", as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.cancel_order(orderLinkId="bybit-cancel", category="spot", symbol="ETHUSDT", as_of=BASE_MS)
            cancelled = exchange.cancel_order(orderLinkId="bybit-cancel", category="spot", symbol=SYMBOL, as_of=BASE_MS)

            self.assertEqual(cancelled["cancelled_order"]["orderLinkId"], "bybit-cancel")

    def test_cancel_order_prioritizes_order_id_over_mismatched_link_id_like_bybit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl")
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            first = exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, orderLinkId="first", as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=40_000.0, orderLinkId="second", as_of=BASE_MS)

            cancelled = exchange.cancel_order(order_id=first["order"]["order_id"], orderLinkId="second", as_of=BASE_MS)

            self.assertEqual(cancelled["cancelled_order"]["orderLinkId"], "first")
            self.assertEqual(exchange.load()["orders"][0]["orderLinkId"], "second")

    def test_rejects_spending_more_than_free_balance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 100.0}, as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=1.0, price=50_000.0, as_of=BASE_MS)

    def test_spot_market_buy_ignores_supplied_price_and_uses_quote_qty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            placed = exchange.place_order(
                category="spot",
                symbol=SYMBOL,
                side="Buy",
                orderType="Market",
                qty=500.0,
                price=1.0,
                as_of=BASE_MS + 59_999,
            )

            self.assertEqual(placed["order"]["price"], 50_000.0)
            self.assertEqual(placed["order"]["marketUnit"], "quoteCoin")
            self.assertAlmostEqual(placed["order"]["base_qty"], 0.01)
            self.assertEqual(placed["state"]["balances"]["USDT"]["free"], 500.0)
            self.assertAlmostEqual(placed["state"]["balances"]["BTC"]["free"], 0.01)

    def test_market_order_accepts_bybit_ioc_time_in_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            placed = exchange.place_order("spot", SYMBOL, "Buy", "Market", qty=500.0, timeInForce="IOC", as_of=BASE_MS + 59_999)

            self.assertEqual(placed["order"]["status"], "Filled")

    def test_spot_market_buy_charges_fee_in_base_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            placed = exchange.place_order("spot", SYMBOL, "Buy", "Market", qty=500.0, fee_rate=0.001, as_of=BASE_MS + 59_999)

            self.assertEqual(placed["order"]["fee_asset"], "BTC")
            self.assertAlmostEqual(placed["order"]["fee_amount"], 0.00001)
            self.assertAlmostEqual(placed["order"]["net_base_qty"], 0.00999)
            self.assertEqual(placed["state"]["balances"]["USDT"]["free"], 500.0)
            self.assertAlmostEqual(placed["state"]["balances"]["BTC"]["free"], 0.00999)

    def test_spot_market_sell_charges_fee_in_quote_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"BTC": 0.02}, as_of=BASE_MS)

            placed = exchange.place_order("spot", SYMBOL, "Sell", "Market", qty=500.0, marketUnit="quoteCoin", fee_rate=0.001, as_of=BASE_MS + 59_999)

            self.assertEqual(placed["order"]["fee_asset"], "USDT")
            self.assertEqual(placed["order"]["fee_amount"], 0.5)
            self.assertEqual(placed["order"]["net_quote_qty"], 499.5)
            self.assertAlmostEqual(placed["state"]["balances"]["BTC"]["free"], 0.01)
            self.assertEqual(placed["state"]["balances"]["USDT"]["free"], 499.5)

    def test_spot_limit_order_fills_on_settle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 101.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=2.0, price=100.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(len(settled["filled_orders"]), 1)
            self.assertEqual(settled["filled_orders"][0]["status"], "Filled")
            self.assertEqual(settled["state"]["orders"], [])
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 800.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["locked"], 0.0)
            self.assertEqual(settled["state"]["balances"]["BTC"]["free"], 2.0)

    def test_spot_limit_buy_fill_charges_fee_in_base_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 101.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=2.0, price=100.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m", fee_rate=0.001)

            self.assertEqual(settled["filled_orders"][0]["fee_asset"], "BTC")
            self.assertEqual(settled["filled_orders"][0]["fee_amount"], 0.002)
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 800.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["locked"], 0.0)
            self.assertEqual(settled["state"]["balances"]["BTC"]["free"], 1.998)

    def test_spot_limit_buy_fill_uses_order_fee_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 101.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=2.0, price=100.0, fee_rate=0.001, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(settled["filled_orders"][0]["fee_asset"], "BTC")
            self.assertEqual(settled["filled_orders"][0]["fee_amount"], 0.002)
            self.assertEqual(settled["state"]["balances"]["BTC"]["free"], 1.998)

    def test_spot_limit_sell_fill_charges_fee_in_quote_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"BTC": 2.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Sell", "Limit", qty=2.0, price=100.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m", fee_rate=0.001)

            self.assertEqual(settled["filled_orders"][0]["fee_asset"], "USDT")
            self.assertEqual(settled["filled_orders"][0]["fee_amount"], 0.2)
            self.assertEqual(settled["state"]["balances"]["BTC"]["free"], 0.0)
            self.assertEqual(settled["state"]["balances"]["BTC"]["locked"], 0.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 199.8)

    def test_spot_limit_buy_gap_fill_uses_open_and_refunds_usdt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 90.0, 95.0, 89.0, 92.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=2.0, price=100.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(settled["filled_orders"][0]["limit_price"], 100.0)
            self.assertEqual(settled["filled_orders"][0]["fill_price"], 90.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 820.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["locked"], 0.0)
            self.assertEqual(settled["state"]["balances"]["BTC"]["free"], 2.0)

    def test_cancel_after_limit_fill_keeps_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 101.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=2.0, price=100.0, orderLinkId="filled-before-cancel", as_of=BASE_MS + 59_999)

            with self.assertRaises(CancelOrderError) as context:
                exchange.cancel_order(orderLinkId="filled-before-cancel", as_of=BASE_MS + 119_999)

            self.assertEqual(context.exception.filled_orders[0]["orderLinkId"], "filled-before-cancel")
            state = exchange.load()
            self.assertEqual(state["orders"], [])
            self.assertEqual(state["balances"]["USDT"]["free"], 800.0)
            self.assertEqual(state["balances"]["USDT"]["locked"], 0.0)
            self.assertEqual(state["balances"]["BTC"]["free"], 2.0)

    def test_cancel_after_limit_fill_uses_order_fee_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 101.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order(
                "spot",
                SYMBOL,
                "Buy",
                "Limit",
                qty=2.0,
                price=100.0,
                orderLinkId="fee-filled-before-cancel",
                fee_rate=0.001,
                as_of=BASE_MS + 59_999,
            )

            with self.assertRaises(CancelOrderError) as context:
                exchange.cancel_order(orderLinkId="fee-filled-before-cancel", as_of=BASE_MS + 119_999)

            self.assertEqual(context.exception.filled_orders[0]["fee_amount"], 0.002)
            state = exchange.load()
            self.assertEqual(state["balances"]["BTC"]["free"], 1.998)

    def test_cancel_order_advances_position_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=95.0, as_of=BASE_MS + 59_999)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=1.0, price=50.0, orderLinkId="cancel-me", as_of=BASE_MS + 59_999)

            cancelled = exchange.cancel_order(orderLinkId="cancel-me", as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(cancelled["closed_positions_before_cancel"][0]["exit_reason"], "TP")
            self.assertEqual(cancelled["state"]["positions"], [])
            self.assertEqual(cancelled["state"]["balances"]["USDT"]["locked"], 0.0)

    def test_cancel_missing_order_does_not_claim_unrelated_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 101.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=2.0, price=100.0, orderLinkId="real-order", as_of=BASE_MS + 59_999)

            with self.assertRaises(ValueError) as context:
                exchange.cancel_order(orderLinkId="missing-order", as_of=BASE_MS + 119_999)

            self.assertNotIsInstance(context.exception, CancelOrderError)
            state = exchange.load()
            self.assertEqual(state["orders"], [])
            self.assertEqual(state["balances"]["BTC"]["free"], 2.0)

    def test_failed_place_order_does_not_log_phantom_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            events_path = Path(tmp) / "events.jsonl"
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 101.0, 102.0, 99.0, 101.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", events_path, cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=2.0, price=100.0, as_of=BASE_MS + 59_999)

            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=1000.0, price=100.0, as_of=BASE_MS + 119_999)

            state = exchange.load()
            self.assertEqual(len(state["orders"]), 1)
            self.assertNotIn("BTC", state["balances"])
            events = [json.loads(line)["type"] for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("order_filled", events)

    def test_place_order_advances_existing_fills_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 101.0, 99.0, 100.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"BTC": 1.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Sell", "Limit", qty=1.0, price=100.0, as_of=BASE_MS + 59_999)

            placed = exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=1.0, price=50.0, as_of=BASE_MS + 119_999)

            self.assertEqual(len(placed["advanced_before_order"]["filled_orders"]), 1)
            self.assertEqual(placed["state"]["balances"]["USDT"]["free"], 50.0)
            self.assertEqual(placed["state"]["balances"]["USDT"]["locked"], 50.0)
            self.assertEqual(placed["state"]["balances"]["BTC"]["free"], 0.0)

    def test_place_order_advance_applies_fee_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=95.0, as_of=BASE_MS + 59_999)

            placed = exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=1.0, price=100.0, fee_rate=0.001, as_of=BASE_MS + 119_999)

            self.assertAlmostEqual(placed["advanced_before_order"]["closed_positions"][0]["fees_usdt"], 0.21)
            self.assertAlmostEqual(placed["advanced_before_order"]["closed_positions"][0]["realized_pnl_usdt"], 9.79)

    def test_linear_market_order_uses_leverage_and_settles_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")

            placed = exchange.place_order(
                category="linear",
                symbol=SYMBOL,
                side="Buy",
                orderType="Market",
                qty=1.0,
                price=1.0,
                takeProfit=110.0,
                stopLoss=95.0,
                as_of=BASE_MS + 59_999,
            )

            self.assertEqual(placed["order"]["leverage"], 5.0)
            self.assertEqual(placed["order"]["price"], 100.0)
            self.assertEqual(placed["state"]["balances"]["USDT"]["free"], 980.0)
            self.assertEqual(placed["state"]["balances"]["USDT"]["locked"], 20.0)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")
            self.assertEqual(len(settled["closed_positions"]), 1)
            self.assertEqual(settled["closed_positions"][0]["exit_reason"], "TP")
            self.assertEqual(settled["closed_positions"][0]["realized_pnl_usdt"], 10.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 1010.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["locked"], 0.0)

    def test_settle_applies_fee_rate_to_linear_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order(
                category="linear",
                symbol=SYMBOL,
                side="Buy",
                orderType="Market",
                qty=1.0,
                takeProfit=110.0,
                stopLoss=95.0,
                as_of=BASE_MS + 59_999,
            )

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m", fee_rate=0.001)

            self.assertAlmostEqual(settled["closed_positions"][0]["fees_usdt"], 0.21)
            self.assertAlmostEqual(settled["closed_positions"][0]["realized_pnl_usdt"], 9.79)
            self.assertAlmostEqual(settled["state"]["balances"]["USDT"]["free"], 1009.79)

    def test_linear_position_settle_uses_order_fee_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order(
                "linear",
                SYMBOL,
                "Buy",
                "Market",
                qty=1.0,
                takeProfit=110.0,
                stopLoss=95.0,
                fee_rate=0.001,
                as_of=BASE_MS + 59_999,
            )

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertAlmostEqual(settled["closed_positions"][0]["fees_usdt"], 0.21)
            self.assertAlmostEqual(settled["closed_positions"][0]["realized_pnl_usdt"], 9.79)

    def test_linear_limit_fill_uses_settle_fee_rate_for_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 120_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order(
                "linear",
                SYMBOL,
                "Buy",
                "Limit",
                qty=1.0,
                price=100.0,
                takeProfit=110.0,
                stopLoss=95.0,
                as_of=BASE_MS + 59_999,
            )
            filled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m", fee_rate=0.001)

            settled = exchange.settle(as_of=BASE_MS + 179_999, interval="1m")

            self.assertEqual(filled["filled_orders"][0]["fee_rate"], 0.001)
            self.assertEqual(filled["filled_orders"][0]["position"]["fee_rate"], 0.001)
            self.assertAlmostEqual(settled["closed_positions"][0]["fees_usdt"], 0.21)

    def test_linear_position_without_tpsl_can_be_closed_with_fee(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 110.0, 111.0, 109.0, 110.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            placed = exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 59_999, interval="1m")
            self.assertEqual(settled["closed_positions"], [])

            position_id = placed["order"]["position"]["position_id"]
            closed = exchange.close_position(position_id, as_of=BASE_MS + 119_999, fee_rate=0.001)

            self.assertEqual(closed["closed_position"]["exit_reason"], "Manual")
            self.assertAlmostEqual(closed["closed_position"]["fees_usdt"], 0.21)
            self.assertAlmostEqual(closed["closed_position"]["realized_pnl_usdt"], 9.79)
            self.assertAlmostEqual(closed["state"]["balances"]["USDT"]["free"], 1009.79)
            self.assertEqual(closed["state"]["balances"]["USDT"]["locked"], 0.0)

    def test_linear_position_can_be_closed_by_symbol_and_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 110.0, 111.0, 109.0, 110.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, as_of=BASE_MS + 59_999)

            closed = exchange.close_position(category="linear", symbol=SYMBOL, side="Buy", as_of=BASE_MS + 119_999)

            self.assertEqual(closed["closed_position"]["symbol"], SYMBOL)
            self.assertEqual(closed["closed_position"]["side"], "Buy")

    def test_close_position_advances_pending_limit_fill_before_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 120_000, 110.0, 111.0, 109.0, 110.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Limit", qty=1.0, price=100.0, as_of=BASE_MS + 59_999)

            closed = exchange.close_position(category="linear", symbol=SYMBOL, side="Buy", as_of=BASE_MS + 179_999)

            self.assertEqual(closed["closed_position"]["exit_reason"], "Manual")
            self.assertEqual(closed["state"]["orders"], [])
            self.assertEqual(closed["state"]["positions"], [])

    def test_close_position_missing_target_persists_prior_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=95.0, as_of=BASE_MS + 59_999)

            with self.assertRaises(ValueError):
                exchange.close_position("missing-position", as_of=BASE_MS + 119_999)

            state = exchange.load()
            self.assertEqual(state["positions"], [])
            self.assertEqual(state["closed_positions"][0]["exit_reason"], "TP")

    def test_linear_manual_close_uses_order_fee_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 110.0, 111.0, 109.0, 110.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            placed = exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, fee_rate=0.001, as_of=BASE_MS + 59_999)

            position_id = placed["order"]["position"]["position_id"]
            closed = exchange.close_position(position_id, as_of=BASE_MS + 119_999)

            self.assertAlmostEqual(closed["closed_position"]["fees_usdt"], 0.21)
            self.assertAlmostEqual(closed["closed_position"]["realized_pnl_usdt"], 9.79)

    def test_close_position_rejects_spoofed_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 110.0, 111.0, 109.0, 110.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            placed = exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, as_of=BASE_MS + 59_999)

            with self.assertRaises(ValueError):
                exchange.close_position(placed["order"]["position"]["position_id"], price=999.0, as_of=BASE_MS + 119_999)

    def test_close_position_spoofed_price_persists_prior_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            other_symbol = "ETHUSDT"
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 110.0, 111.0, 109.0, 110.0),
                    candle(other_symbol, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(other_symbol, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.set_leverage("linear", other_symbol, "5", "5")
            target = exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, as_of=BASE_MS + 59_999)
            exchange.place_order("linear", other_symbol, "Buy", "Market", qty=1.0, takeProfit=110.0, stopLoss=95.0, as_of=BASE_MS + 59_999)

            with self.assertRaises(ValueError):
                exchange.close_position(target["order"]["position"]["position_id"], price=999.0, as_of=BASE_MS + 119_999)

            state = exchange.load()
            self.assertEqual(len(state["closed_positions"]), 1)
            self.assertEqual(state["closed_positions"][0]["symbol"], other_symbol)

    def test_close_position_respects_prior_stop_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 111.0, 94.0, 110.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            placed = exchange.place_order(
                "linear",
                SYMBOL,
                "Buy",
                "Market",
                qty=1.0,
                takeProfit=120.0,
                stopLoss=95.0,
                as_of=BASE_MS + 59_999,
            )

            closed = exchange.close_position(placed["order"]["position"]["position_id"], as_of=BASE_MS + 119_999)

            self.assertTrue(closed["triggered_before_manual_close"])
            self.assertEqual(closed["closed_position"]["exit_reason"], "SL")
            self.assertEqual(closed["closed_position"]["exit_price"], 95.0)
            self.assertEqual(closed["state"]["balances"]["USDT"]["free"], 995.0)

    def test_linear_one_sided_take_profit_settles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 112.0, 99.0, 111.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, takeProfit=110.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(len(settled["closed_positions"]), 1)
            self.assertEqual(settled["closed_positions"][0]["exit_reason"], "TP")

    def test_rejects_duplicate_order_link_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, orderLinkId="dup-1", as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=49_000.0, orderLinkId="dup-1", as_of=BASE_MS)

    def test_order_link_id_can_be_reused_after_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, orderLinkId="retry-1", as_of=BASE_MS)
            exchange.cancel_order(orderLinkId="retry-1", as_of=BASE_MS)

            placed = exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=49_000.0, orderLinkId="retry-1", as_of=BASE_MS)

            self.assertEqual(placed["order"]["orderLinkId"], "retry-1")

    def test_unpriced_asset_marks_valuation_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 100.0, "DOGE": 10.0}, as_of=BASE_MS)

            summary = exchange.wallet_summary(as_of=BASE_MS)

            self.assertFalse(summary["totals"]["valuation_complete"])
            self.assertEqual(summary["totals"]["unpriced_assets"], ["DOGE"])
            self.assertEqual(summary["totals"]["wallet_usdt"], 100.0)

    def test_wallet_marks_unpriced_open_position_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            placed = exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, as_of=BASE_MS + 59_999)

            summary = exchange.wallet_summary(symbols=SYMBOL, as_of=BASE_MS + 119_999)

            self.assertFalse(summary["totals"]["valuation_complete"])
            self.assertEqual(summary["totals"]["unpriced_positions"], [placed["order"]["position"]["position_id"]])

    def test_market_order_rejects_stale_mark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Market", qty=100.0, as_of=BASE_MS + 119_999)

    def test_linear_limit_fill_does_not_settle_on_same_candle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 105.0, 106.0, 100.0, 105.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order(
                "linear",
                SYMBOL,
                "Buy",
                "Limit",
                qty=1.0,
                price=100.0,
                takeProfit=104.0,
                stopLoss=95.0,
                as_of=BASE_MS + 59_999,
            )

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(len(settled["filled_orders"]), 1)
            self.assertEqual(settled["closed_positions"], [])
            self.assertEqual(len(settled["state"]["positions"]), 1)
            self.assertEqual(settled["state"]["positions"][0]["opened_at_ms"], BASE_MS + 120_000)

    def test_linear_limit_gap_fill_uses_open_and_releases_margin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 90.0, 95.0, 89.0, 92.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Limit", qty=1.0, price=100.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(settled["filled_orders"][0]["fill_price"], 90.0)
            self.assertEqual(settled["state"]["positions"][0]["entry_price"], 90.0)
            self.assertEqual(settled["state"]["positions"][0]["margin_usdt"], 18.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 982.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["locked"], 18.0)

    def test_linear_limit_gap_fill_falls_back_when_tpsl_would_break(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 90.0, 95.0, 89.0, 92.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Buy", "Limit", qty=1.0, price=100.0, takeProfit=110.0, stopLoss=95.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(settled["filled_orders"][0]["fill_price"], 100.0)
            self.assertEqual(settled["state"]["positions"][0]["entry_price"], 100.0)

    def test_linear_sell_gap_fill_falls_back_when_margin_top_up_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 120.0, 121.0, 119.0, 120.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 20.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Sell", "Limit", qty=1.0, price=100.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(settled["filled_orders"][0]["fill_price"], 100.0)
            self.assertEqual(settled["state"]["positions"][0]["entry_price"], 100.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["locked"], 20.0)

    def test_linear_limit_fill_settles_later_candle_in_same_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 105.0, 106.0, 100.0, 105.0),
                    candle(SYMBOL, BASE_MS + 120_000, 105.0, 106.0, 104.0, 106.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order(
                "linear",
                SYMBOL,
                "Buy",
                "Limit",
                qty=1.0,
                price=100.0,
                takeProfit=104.0,
                stopLoss=95.0,
                as_of=BASE_MS + 59_999,
            )

            settled = exchange.settle(as_of=BASE_MS + 179_999, interval="1m")

            self.assertEqual(len(settled["filled_orders"]), 1)
            self.assertEqual(len(settled["closed_positions"]), 1)
            self.assertEqual(settled["closed_positions"][0]["exit_reason"], "TP")

    def test_linear_position_liquidates_when_margin_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 101.0, 98.0, 98.5),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "100", "100")
            exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(settled["closed_positions"][0]["exit_reason"], "Liquidation")
            self.assertEqual(settled["closed_positions"][0]["cash_returned_usdt"], 0.0)
            self.assertEqual(settled["closed_positions"][0]["realized_pnl_usdt"], -1.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 999.0)

    def test_linear_short_take_profit_settles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles(
                [
                    candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0),
                    candle(SYMBOL, BASE_MS + 60_000, 100.0, 101.0, 88.0, 90.0),
                ]
            )
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.set_leverage("linear", SYMBOL, "5", "5")
            exchange.place_order("linear", SYMBOL, "Sell", "Market", qty=1.0, takeProfit=90.0, stopLoss=105.0, as_of=BASE_MS + 59_999)

            settled = exchange.settle(as_of=BASE_MS + 119_999, interval="1m")

            self.assertEqual(settled["closed_positions"][0]["exit_reason"], "TP")
            self.assertEqual(settled["closed_positions"][0]["realized_pnl_usdt"], 10.0)
            self.assertEqual(settled["state"]["balances"]["USDT"]["free"], 1010.0)

    def test_spot_market_sell_quote_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"BTC": 0.02}, as_of=BASE_MS)

            placed = exchange.place_order("spot", SYMBOL, "Sell", "Market", qty=500.0, marketUnit="quoteCoin", as_of=BASE_MS + 59_999)

            self.assertAlmostEqual(placed["order"]["base_qty"], 0.01)
            self.assertEqual(placed["state"]["balances"]["USDT"]["free"], 500.0)
            self.assertAlmostEqual(placed["state"]["balances"]["BTC"]["free"], 0.01)

    def test_as_of_cannot_move_backwards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, as_of=BASE_MS + 60_000)

            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=49_000.0, as_of=BASE_MS + 59_999)

    def test_wallet_without_as_of_does_not_trust_stale_historical_mark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 50_000.0, 50_100.0, 49_900.0, 50_000.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 1000.0, "BTC": 0.1}, as_of=BASE_MS)

            summary = exchange.wallet_summary(symbols=SYMBOL)

            self.assertFalse(summary["totals"]["valuation_complete"])
            self.assertEqual(summary["totals"]["unpriced_assets"], ["BTC"])

    def test_wallet_rejects_past_as_of_after_state_advanced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, as_of=BASE_MS + 60_000)

            with self.assertRaises(ValueError):
                exchange.wallet_summary(as_of=BASE_MS)

    def test_rejects_unknown_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl")
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.wallet_summary(symbols=SYMBOL, as_of=BASE_MS, mark_interval="bad")

    def test_rejects_spot_tpsl_and_invalid_time_in_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, takeProfit=60_000.0, as_of=BASE_MS)
            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, timeInForce="DAY", as_of=BASE_MS)
            with self.assertRaises(ValueError):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, timeInForce="IOC", as_of=BASE_MS)
            with self.assertRaises(ValueError):
                exchange.place_order("linear", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, tpslMode="Partial", as_of=BASE_MS)

    def test_rejects_linear_margin_overspend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
            cache.upsert_candles([candle(SYMBOL, BASE_MS, 100.0, 101.0, 99.0, 100.0)])
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
            exchange.reset({"USDT": 10.0}, as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, leverage=1.0, as_of=BASE_MS + 59_999)

    def test_rejects_bybit_one_way_leverage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)

            with self.assertRaises(ValueError):
                exchange.set_leverage("linear", SYMBOL, "3", "5")

    def test_writes_replay_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            exchange = SimulatedExchange(Path(tmp) / "exchange.json", events_path, cache=LocalMarketCache(Path(tmp) / "market.sqlite3"))
            exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
            exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=0.01, price=50_000.0, orderLinkId="log-1", as_of=BASE_MS)
            exchange.cancel_order(orderLinkId="log-1", as_of=BASE_MS)

            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["type"] for event in events], ["exchange_reset", "place_order", "cancel_order"])


class LinearLimitExpiryTests(unittest.TestCase):
    """TTL expiry for pending linear limit orders (limit_retest entry policy plumbing)."""

    def _exchange(self, tmp: str, candles: list[Candle]) -> SimulatedExchange:
        cache = LocalMarketCache(Path(tmp) / "market.sqlite3")
        cache.upsert_candles(candles)
        exchange = SimulatedExchange(Path(tmp) / "exchange.json", Path(tmp) / "events.jsonl", cache=cache)
        exchange.reset({"USDT": 1000.0}, as_of=BASE_MS)
        return exchange

    def _place_retest_limit(self, exchange: SimulatedExchange, expires_min: int = 10) -> dict:
        return exchange.place_order(
            "linear",
            SYMBOL,
            "Buy",
            "Limit",
            qty=1.0,
            price=99.5,
            takeProfit=104.0,
            stopLoss=98.0,
            as_of=BASE_MS,
            expiresAtMs=BASE_MS + expires_min * 60_000,
            entryPolicy="limit_retest",
            entryRefPrice=100.0,
        )

    def test_limit_fills_before_expiry_and_opens_position(self) -> None:
        candles = [candle(SYMBOL, BASE_MS, 100.0, 100.2, 99.9, 100.0), candle(SYMBOL, BASE_MS + 60_000, 100.0, 100.1, 99.4, 100.0)]
        candles += [candle(SYMBOL, BASE_MS + k * 60_000, 100.0, 100.2, 99.9, 100.0) for k in range(2, 8)]
        with tempfile.TemporaryDirectory() as tmp:
            exchange = self._exchange(tmp, candles)
            placed = self._place_retest_limit(exchange)
            self.assertEqual(placed["order"]["status"], "New")
            self.assertEqual(placed["order"]["entry_policy"], "limit_retest")
            self.assertEqual(placed["order"]["entry_ref_price"], 100.0)

            settled = exchange.settle(as_of=BASE_MS + 7 * 60_000)

            self.assertEqual(len(settled["filled_orders"]), 1)
            self.assertEqual(settled["expired_orders"], [])
            fill = settled["filled_orders"][0]
            self.assertEqual(fill["fill_price"], 99.5)
            position = settled["state"]["positions"][0]
            self.assertEqual(position["entry_price"], 99.5)
            self.assertEqual(position["takeProfit"], 104.0)
            self.assertEqual(position["stopLoss"], 98.0)

    def test_limit_expires_when_no_pullback_and_releases_margin(self) -> None:
        candles = [candle(SYMBOL, BASE_MS + k * 60_000, 100.0, 100.2, 99.9, 100.0) for k in range(0, 16)]
        with tempfile.TemporaryDirectory() as tmp:
            exchange = self._exchange(tmp, candles)
            self._place_retest_limit(exchange, expires_min=10)
            locked_after_place = exchange.load()["balances"]["USDT"]["locked"]
            self.assertAlmostEqual(locked_after_place, 99.5)

            settled = exchange.settle(as_of=BASE_MS + 15 * 60_000)

            self.assertEqual(settled["filled_orders"], [])
            self.assertEqual(len(settled["expired_orders"]), 1)
            expired = settled["expired_orders"][0]
            self.assertEqual(expired["status"], "Expired")
            self.assertEqual(expired["expired_at_ms"], BASE_MS + 10 * 60_000)
            state = settled["state"]
            self.assertEqual(state["orders"], [])
            self.assertEqual(state["positions"], [])
            self.assertAlmostEqual(state["balances"]["USDT"]["locked"], 0.0)
            self.assertAlmostEqual(state["balances"]["USDT"]["free"], 1000.0)
            events = [json.loads(line)["type"] for line in (Path(tmp) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("order_expired", events)

    def test_touch_after_expiry_does_not_fill(self) -> None:
        candles = [candle(SYMBOL, BASE_MS + k * 60_000, 100.0, 100.2, 99.9, 100.0) for k in range(0, 12)]
        candles.append(candle(SYMBOL, BASE_MS + 12 * 60_000, 100.0, 100.1, 99.0, 99.2))
        candles += [candle(SYMBOL, BASE_MS + k * 60_000, 99.2, 99.4, 99.0, 99.2) for k in range(13, 21)]
        with tempfile.TemporaryDirectory() as tmp:
            exchange = self._exchange(tmp, candles)
            self._place_retest_limit(exchange, expires_min=10)

            settled = exchange.settle(as_of=BASE_MS + 20 * 60_000)

            self.assertEqual(settled["filled_orders"], [])
            self.assertEqual(len(settled["expired_orders"]), 1)
            self.assertEqual(settled["state"]["positions"], [])

    def test_expiry_param_is_rejected_for_market_and_spot_orders(self) -> None:
        candles = [candle(SYMBOL, BASE_MS, 100.0, 100.2, 99.9, 100.0)]
        with tempfile.TemporaryDirectory() as tmp:
            exchange = self._exchange(tmp, candles)
            with self.assertRaisesRegex(ValueError, "linear Limit"):
                exchange.place_order("linear", SYMBOL, "Buy", "Market", qty=1.0, takeProfit=104.0, stopLoss=98.0, as_of=BASE_MS, expiresAtMs=BASE_MS + 600_000)
            with self.assertRaisesRegex(ValueError, "linear Limit"):
                exchange.place_order("spot", SYMBOL, "Buy", "Limit", qty=1.0, price=99.5, as_of=BASE_MS, expiresAtMs=BASE_MS + 600_000)
            with self.assertRaisesRegex(ValueError, "after as_of"):
                exchange.place_order("linear", SYMBOL, "Buy", "Limit", qty=1.0, price=99.5, takeProfit=104.0, stopLoss=98.0, as_of=BASE_MS, expiresAtMs=BASE_MS)


if __name__ == "__main__":
    unittest.main()
