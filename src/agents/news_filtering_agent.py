from src.api import ai_client
from src.const import *


class NewsFilteringAgent:
    def __init__(self) -> None:
        pass

    def filter_news(self, coin, news):
        filtered_news = ai_client.create(
            [
                ai_client.make_msg(
                    f"""You are news manager, you validate and summarize reliable news relevant to the topic of {coin}.
            
                    Respond with json of this format: 
                    {{
                        'description': detailed summary of relevant news,
                        'sentiment': general sentiment on the market about the topic,
                        'sentiment_decription': 'explain the reason for such sentiment',
                    }}
                    """,
                    role=ROLE_SYSTEM,
                ),
                ai_client.make_msg(
                    text=f"""These are the news about the {coin} coin for the last week:
                    {news}
                    
                    Analyze and extract data relevant to cryptocurrencies and {coin} coin. Pay attention to anything fresh that could affect the market
                    """,
                ),
            ],
            format=JSON_MODE,
        )

        return filtered_news
