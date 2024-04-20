from src.agents.news_filtering_agent import *
from src.plots import make_candlebars
from src.api import *


class TradingAgent:
    """Basic trading agent making decion based on several inputs and history"""

    def __init__(self):
        pass

    def decide(
        self,
        coin,
        news=None,
        day_data=None,
        week_data=None,
        month_data=None,
        year_data=None,
        operations_history=None,
        current_balance=None,
        verbose=False,
    ):
        latest_data = day_data or week_data or month_data or year_data
        if not latest_data:
            raise Exception("Provide at least some historical data")

        current_price = latest_data[-1]["c"]

        day_buf = make_candlebars(day_data, useBuf=True) if day_data else None
        week_buf = make_candlebars(week_data, useBuf=True) if week_data else None
        month_buf = make_candlebars(month_data, useBuf=True) if month_data else None
        year_buf = make_candlebars(year_data, useBuf=True) if year_data else None

        messages = [
            ai_client.make_msg(
                """You are professional trader.
            You have deep knowledge of cryptocurrencies and trading markets, and you can maximize profits the most efficiently.
            You trade in USDT.
            
            You an only buy or sell at the current price, you can not set stop loss.
            Avoid spending all money in one deal, play smart.
            
            Respond with json of this format: 
            {
                'description': what you see on charts, general short and long term trends, 
                'trend_analysys': rough prediction of near future development of coin,
                'techical_analysys': visual technical analysis of plot and indicators,
                'optimal_strategy": optimal trading strategy to maximize profits in this situation,
                'final_decision': what to do, possible variants: 'sell', 'buy', 'hold' (do nothing),
                'amount': amount of coin to buy or sell, 0 if action is 'hold'
                'usdt_amount': usdt equivalent of amount field based on current price,
                'price': price of coin at this deal
            }
            """,
                role=ROLE_SYSTEM,
            ),
        ]

        if day_buf:
            messages.append(
                ai_client.make_msg(
                    text="This is price history of this coin in the last day",
                    img=day_buf,
                ),
            )

        if week_buf:
            messages.append(
                ai_client.make_msg(
                    text="This is price history of this coin in the last week",
                    img=week_buf,
                ),
            )

        if month_buf:
            messages.append(
                ai_client.make_msg(
                    text="This is price history of this coin in the last month",
                    img=month_buf,
                ),
            )

        if year_buf:
            messages.append(
                ai_client.make_msg(
                    text="This is price history of this coin in the last year",
                    img=year_buf,
                ),
            )

        if news:
            messages.append(
                ai_client.make_msg(
                    text=f"""This is news related to this coin ({coin}) from the last week: {news}""",
                ),
            )

        if operations_history:
            messages.append(
                ai_client.make_msg(
                    text=f"""Use data about previos operations to create better strategy.
                    History of operations with this coin: {str(operations_history)}""",
                ),
            )

        if current_balance:
            messages.append(
                ai_client.make_msg(
                    text=f"""your current funds: {str(current_balance)}""",
                ),
            )

        messages.append(
            ai_client.make_msg(
                text=f"""This is price charts of {coin} coin, you need to come up with optimal strategy at this moment. 
                Permorm price, signal and trend analisys of this graph of this crypto coin. Explain your decisons.
                Avoid spending all money in one deal and develop a smart strategy.

                Latest price: {current_price}
                """,
            ),
        )

        if verbose:
            print(messages)

        response = ai_client.create(messages, format=JSON_MODE)
        return response
