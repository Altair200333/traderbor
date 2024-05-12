from src.agents.marginal_agent import MarginalAgent
from src.agents.news_filtering_agent import NewsFilteringAgent
from src.agents.supervisor_margin_trader import SupervisorMarginTrader
from src.api import *
from src.const import *
from src.news_providers.tradingview_provider import TradingViewProvider
from src.signal_providers.signal_manager import *
import random


class RandomPredictor:
    def __init__(self, trade_rate=0.03, amount_rate=0.3) -> None:
        self.trade_rate = trade_rate
        self.amount_rate = amount_rate

    def decide(
        self,
        coin,
        balance,
        cutoff=None,
    ):
        date_cutoff = cutoff if cutoff else get_current_datetime()

        day_history, week_history = map_async(
            lambda: get_day_history(coin, date_cutoff),
            lambda: get_week_history(coin, date_cutoff),
        )

        current_price = day_history[-1]["c"]

        action = "hold"
        option = random.randint(1, 3)
        tp = 0.0
        sl = 0.0

        if option == 1:
            action = "hold"
        elif option == 2:
            action = "long"
            tp = current_price * (1.0 + self.trade_rate)
            sl = current_price * (1.0 - self.trade_rate * 0.5)
        elif option == 3:
            action = "short"
            tp = current_price * (1.0 - self.trade_rate)
            sl = current_price * (1.0 + self.trade_rate * 0.5)

        return (
            {
                "final_decision": action,
                "price": current_price,
                "stop_loss": sl,
                "take_profit": tp,
                "leverage": "1x",
                "amount": balance["usdt"] * self.amount_rate,
            },
            [],
            day_history,
        )
