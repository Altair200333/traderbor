from src.agents.news_filtering_agent import *
from src.plots import make_candlebars
from src.api import *
from src.const import *


class MarginalAgent:
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
                f"""You are professional futures trader with an extensive understanding of cryptocurrency markets. 
                
                Respond with JSON of described format.
                Guidelines:
                - Do not buy on everything you have, distribute spendings!
                - Feel free to sell all if selling improves total net_worth of acccount
                - Consider historical market data and recent news.
                - Use only money from your 'balance'
                
                Your ultimate goal is to make make as much profit as possible
                """,
                role=ROLE_SYSTEM,
            ),
        ]

        if day_buf:
            messages.append(
                ai_client.make_msg(
                    text=f"This is price history of this coin in the last day. Current price: {current_price}",
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
                    text=f"""This is news relevant news and sentiment avout {coin}: '''{news}'''""",
                ),
            )

        if operations_history:
            messages.append(
                ai_client.make_msg(
                    text=f"""This is your trading history with this coin: {str(operations_history)}.
                    Use it to understand profitable deals""",
                ),
            )

        if current_balance:
            messages.append(
                ai_client.make_msg(
                    text=f"""Your current balance: {str(current_balance)}. You can only use money that you have""",
                ),
            )

        messages.append(
            ai_client.make_msg(
                text=f"""Decise best actions in the market. Take your tading history into account. 

                Decide what to do in futures trading: set futures with stop-loss and take-profit or just wait.
                
                Analyze current market conditions and respond with a structured JSON output that includes:
                {{
                    'trend_analysis': "Detailed prediction of short and long-term market movements based on price history.",
                    'technical_analysis': "Insights from visual technical analysis including relevant trading indicators.",
                    'prediction': "Where do you think price can go".
                    
                    'profits_on_sell': 'Calculate profits or loss in usdt if "buying" futures hits.',
                    'profits_on_buy': 'Calculate profits or loss in usdt if "selling" futures hits.',
                    'profits_on_hold': 'Explain why waiting at this point in market can be preferable',
                    
                    'decision_process': "Compare profits or setting each type of limit or waiting, and pick the best at the moment",
                    
                    'final_decision': "should be one of: 'buy', 'sell', 'hold' kind of futures",
                    'price': "price to open futures",
                    'stop_loss': "stop loss price",
                    'take_profit': "take profit price"
                }}
                
                example output:
                {{
                    'trend_analysis': <some analysis including news and trends from charts>,
                    'technical_analysis': <investigation of indicators and volumes>,
                    'prediction': <rough prediction of price in short term>,
                    
  
                    'profits_on_sell': <Opening "selling" futures at price X, take-profit Y and stop-loss Z would result in N% income or will close at M%>, 
                    'profits_on_buy': <Opening "buying" futures at price X, take-profit Y and stop-loss Z would result in N% income or will close at M%>,
                    'profits_on_hold': <Holding right now is/is not preferable because market is ...>,
                    
                    'decision_process': <...comparing options...> and the best now is to <action_name>,
                    
                    'final_decision': <kind of futures, one of 'sell' 'buy' or 'hold'>,
                    
                    'price': "price to open futures",
                    'stop_loss': "stop loss price",
                    'take_profit': "take profit price"
                }}
                field names are case sensitive!
                """,
            ),
        )

        if verbose:
            print(messages)

        response = ai_client.create(messages, format=JSON_MODE)

        messages.append(
            ai_client.make_msg(text=response, role=ROLE_ASSISTANT),
        )

        return response, messages
