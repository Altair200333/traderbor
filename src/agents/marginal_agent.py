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
                
                Guidelines:
                - Distribute spendings, use at most 30% of your budget
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
                    "text": f"This is price history of {coin} coin in the 12 hours.",
                    "img": day_12h_data,
                },
                {
                    "text": f"This is price history of {coin} coin in the last day.",
                    "img": day_data,
                },
                {
                    "text": f"This is price history of {coin} coin in the last week.",
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
                text=f"""Evaluate the price movement and possibility to enter momentum trading at this spot. Explain your reasoning process step-by step""",
            ),
        )

        response = ai_client.create(messages)

        messages.append(
            ai_client.make_msg(
                text=f"""Evaluate the price movement and possibility to enter momentum trading at this spot. Explain your reasoning process step-by step""",
            ),
        )

        # append first response here
        messages.append(
            ai_client.make_msg(text=response, role=ROLE_ASSISTANT),
        )

        messages.append(
            ai_client.make_msg(
                text=f"""Now based on this respond with this JSON:
                
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
                    
                    'price': "price to open futures, must be float number",
                    'stop_loss': "stop loss price, must be float number",
                    'take_profit': "take profit price, must be float number",
                    'amount': 'amount of usdt in the deal, must be float number',
                }}
                
                make sure that you take_profit is always higher then stop_loss to secure profits and avoid losses.
                field names are case sensitive! Respond with a valid JSON!
                """,
            ),
        )

        response = ai_client.create(messages, format=JSON_MODE)

        messages.append(
            ai_client.make_msg(text=response, role=ROLE_ASSISTANT),
        )

        if verbose:
            print(messages)

        return response, messages
