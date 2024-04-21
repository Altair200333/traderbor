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
        goal="3k$",
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
                f"""You are professional trader with an extensive understanding of cryptocurrency markets. 
                
                Respond with JSON of descrived format
                Guidelines:
                - Do not buy on everything you have, distribute spendings!
                - Feel free to sell all if selling improves total net_worth of acccount
                - Consider historical market data and recent news.
                - Use only money from your 'balance'
                - Never close deals resulting in negative balance
                - Never sell if you will loose money after it
                
                Your ultimate goal is to make {goal}
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
                text=f"""Decise best actions in the market. 
                
                Do not buy on more then 50% of what you have, distribute spendings.
                
                Analyze current market conditions and respond with a structured JSON output that includes:
                {{
                    'trend_analysis': "Prediction of short and long-term market movements based on price history.",
                    'technical_analysis': "Insights from visual technical analysis including relevant trading indicators.",
                    
                    'profits_on_sell': 'Calculate profits or loss in usdt if sold X amount of {coin} now. Show calculation process', 
                    'profits_on_buy': 'Calculate profits or loss if buying amount X of coin right now. do not buy "on high"', 
                    'profits_on_hold': 'Explain why waiting at this point in market can be preferable',
                    
                    'decision_process': "Compare profits or selling, buying or holding right now, and pick the best at the moment",
                    
                    'final_decision': "should be one of: 'buy', 'sell', 'hold'",

                    'price': "same as {current_price}",
                    'amount': "Amount of coin to trade, 0 if holding.",
                    'usdt_amount': "USDT amount in transaction, based on the current price.",
                }}
                
                example output:
                {{
                    'trend_analysis': <some analysis includin news and trends from charts>,
                    'technical_analysis': <investigation of indicators and volumes>,
                  
                    'price': "{current_price}",
  
                    'profits_on_sell': <Selling amount X would result in profilt Y which will increase net worth by N usdt, Balance after operation: {coin}: M, USDT: M>, 
                    'profits_on_buy': <Buying at this level amount X would allow us to sell it at Z or higher because..., Balance after operation: {coin}: M, USDT: M>, 
                    'profits_on_hold': <Holding right now is/is not becase market is ...>,
                    
                    'decision_process': <...comparing options...> and the best now is to <action_name>,
                    
                    'final_decision': <action_name>,

                    'amount': "X",
                    'usdt_amount': "Y",
                }}
                field names are case sensitive!
                """,
            ),
        )

        if verbose:
            print(messages)

        response = ai_client.create(messages, format=JSON_MODE)
        return response
