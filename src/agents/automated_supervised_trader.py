from src.agents.marginal_agent import MarginalAgent
from src.agents.news_filtering_agent import NewsFilteringAgent
from src.agents.supervisor_margin_trader import SupervisorMarginTrader
from src.api import *
from src.const import *
from src.utils import *
from src.news_providers.tradingview_provider import TradingViewProvider
from src.signal_providers.signal_manager import *
import json

NEWS_ITEMS_LIMIT = 30


class AutomatedSupervisedTrader:
    def __init__(self) -> None:
        self.napi = TradingViewProvider()
        self.trading_agent = SupervisorMarginTrader()
        self.filtering_agent = NewsFilteringAgent()

    def decide(self, coin, balance, use_news=False, cutoff=None, verbose=False):
        date_cutoff = cutoff if cutoff else get_current_datetime()
        filtered_news = None
        if use_news:
            news = self.napi.get_news(coin)
            filtered_news = self.filtering_agent.filter_news(
                coin, news[:NEWS_ITEMS_LIMIT]
            )

        day_history, week_history = map_async(
            lambda: get_day_history(coin, date_cutoff),
            lambda: get_week_history(coin, date_cutoff),
        )

        response, messages = self.trading_agent.decide(
            coin=coin,
            news=filtered_news,
            # day_12h_data=day_12h_history,
            day_data=day_history,
            # week_data=week_history,
            # year_data=year_history,
            # operations_history=operations_history,
            current_balance=balance,
            leverage="1x",
            verbose=verbose,
        )

        response = json.loads(response)
        response["amount"] = try_float(response["amount"])
        response["price"] = try_float(response["price"])
        response["stop_loss"] = try_float(response["stop_loss"])
        response["take_profit"] = try_float(response["take_profit"])

        return response, messages, day_history
