from src.signal_providers.signal_manager import *
from src.utils import *


def match_candle(order, candle):
    tp = order["take_profit"]
    sl = order["stop_loss"]

    order_time = order["t"]
    candle_time = candle["t"]

    if in_range(tp, candle["l"], candle["h"]) and candle_time > order_time:
        return True, "TP"
    elif in_range(sl, candle["l"], candle["h"]) and candle_time > order_time:
        return True, "SL"

    return False, "-"


def match_candle_list(order, candles):
    for candle in candles:
        matched, kind = match_candle(order, candle)
        if matched:
            return matched, kind, candle

    return False, "-", None


def calculate_adjustment(order):
    tp = order["take_profit"]
    sl = order["stop_loss"]
    amount = order["amount"]
    kind = order["kind"]
    entry = order["price"]

    number_of_units = amount / entry

    if kind == "short":
        tp_diff = (entry - tp) * number_of_units
        sl_diff = (entry - sl) * number_of_units
    elif kind == "long":
        tp_diff = (tp - entry) * number_of_units
        sl_diff = (sl - entry) * number_of_units

    return tp_diff, sl_diff


INTERVAL_H = 6
STEPS = 5


class TestingEngine:
    def __init__(
        self,
        coin,
        agent,
        balance,
        interval_h=INTERVAL_H,
        days_back=30,
    ) -> None:
        self.orders = []
        self.balance = balance
        self.logs = []

        self.interval_h = interval_h
        self.days_back = days_back
        self.coin = coin
        self.agent = agent

        start_date = subtract_time(get_current_datetime(), days=self.days_back)

        self.current_date = start_date

    def process_step(self):
        if self.balance["usdt"] < 100:
            return get_day_history(self.coin, self.current_date)

        predict, _, history = self.agent.decide(
            coin=self.coin, balance=self.balance, cutoff=self.current_date
        )

        self.logs.append(f"Predict {predict}")

        final_decision = predict["final_decision"]
        if final_decision != "hold":
            tp = predict["take_profit"]
            sl = predict["stop_loss"]
            amount = predict["amount"]
            price = predict["price"]

            new_order = {
                "kind": final_decision,
                "take_profit": tp,
                "stop_loss": sl,
                "amount": amount,
                "price": price,
                "t": get_timestamp(self.current_date),
            }

            self.orders.append(new_order)

            self.balance["usdt"] -= amount

        self.logs.append(f"balance {self.balance}")

        return history

    def test(self, steps=STEPS):
        for i in range(steps):
            self.logs.append(f"Start step {i}")

            history = self.process_step()

            for idx, order in enumerate(self.orders):
                matched, kind, candle_data = match_candle_list(order, history)
                if matched:
                    amount = order["amount"]
                    self.logs.append(f"MATCHED {str(order)} {kind}")

                    tp_diff, sl_diff = calculate_adjustment(order)

                    if kind == "TP":
                        self.balance["usdt"] += amount + tp_diff
                    if kind == "SL":
                        self.balance["usdt"] += amount + sl_diff

                    self.logs.append(f"new balance {self.balance}")
                    del self.orders[idx]

            self.current_date = add_time(self.current_date, hours=self.interval_h)
