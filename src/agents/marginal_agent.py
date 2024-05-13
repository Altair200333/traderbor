from src.agents.news_filtering_agent import *
from src.plots import make_candlebars
from src.api import *
from src.const import *


class MarginalAgent:
    """Basic trading agent making decion based on several inputs and history"""

    def __init__(self):
        pass

    def _make_buf(self, data):
        return make_candlebars(data, useBuf=True) if data else None

    def _append_messages(self, messages, images):
        for item in images:
            description = item["text"]
            img = item["img"] if "img" in item else None
            data = item["data"] if "data" in item else None
            if img:
                buf = self._make_buf(img)
                messages.append(
                    ai_client.make_msg(
                        text=description,
                        img=buf,
                    ),
                )
            elif data:
                messages.append(
                    ai_client.make_msg(
                        text=f"{description}. {str(data)}",
                    ),
                )

    def decide(
        self,
        coin,
        news=None,
        day_12h_data=None,
        day_data=None,
        week_data=None,
        month_data=None,
        year_data=None,
        current_balance=None,
        leverage="1x",
        verbose=False,
    ):
        latest_data = day_12h_data or day_data or week_data or month_data or year_data
        if not latest_data:
            raise Exception("Provide at least some historical data")

        current_price = latest_data[-1]["c"]

        messages = [
            ai_client.make_msg(
                f"""You are professional momentum trader with an extensive understanding of cryptocurrency markets. 
                
                Respond with JSON of described format.
                Guidelines:
                - Do not buy on everything you have, distribute spendings!
                - Consider historical market data and recent news
                - Use only money from your 'balance'
                
                Your goal is to make make as much profit as possible
                """,
                role=ROLE_SYSTEM,
            ),
        ]

        self._append_messages(
            messages,
            [
                {
                    "text": f"This is price history of this coin in the 12 hours.",
                    "img": day_12h_data,
                },
                {
                    "text": f"This is price history of this coin in the last day.",
                    "img": day_data,
                },
                {
                    "text": f"This is price history of this coin in the last week.",
                    "img": week_data,
                },
                {
                    "text": f"This is price history of this coin in the last month.",
                    "img": month_data,
                },
                {
                    "text": f"This is price history of this coin in the last year.",
                    "img": year_data,
                },
                {
                    "text": f"This is news relevant news and sentiment about {coin}",
                    "data": news,
                },
                {
                    "text": f"Your current balance, you can only use money that you have:",
                    "data": current_balance,
                },
            ],
        )

        messages.append(
            ai_client.make_msg(
                text=f"""Decise best actions in the market following momentum trading strategy. Current price: {current_price}.

                Decide what to do in futures trading: set futures with stop-loss and take-profit or just wait.
                You can utilize up to {leverage} leverage.
                
                Analyze current market conditions and respond with a structured JSON output that includes:
                {{
                    'technical_analysis': <visual technical analysis including relevant trading indicators>,
                    'trend_analysis': <step by step analysis process of current price movement>,
                    'analysis': <EVALUATE current entry point based on momentum, volume, indicators. perform STEP-BY-STEP analysys>,
                    
  
                    'profits_on_long': <Opening "long" futures at price X, take-profit Y and stop-loss Z would result in N% income or will close at M%>, 
                    'profits_on_short': <Opening "short" futures at price X, take-profit Y and stop-loss Z would result in N% income or will close at M%>,
                    'profits_on_hold': <Holding right now is/is not preferable because market is ...>,
                    
                    'decision_process': <...comparing options...> and the best now is to <action_name>,
                    
                    'final_decision': <kind of futures, one of 'long' 'short' or 'hold'>,
                    
                    'price': "price to open futures",
                    'stop_loss': "stop loss price",
                    'take_profit': "take profit price",
                    'leverage': "leverage to take",
                    'amount': 'amount of usd in the deal',
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
