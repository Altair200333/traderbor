from src.api import ai_client
from src.const import *


class NewsFilteringAgent:
    def __init__(self) -> None:
        pass

    def filter_news(self, news):
        filtered_news = ai_client.create(
            [
                ai_client.make_msg(
                    """You are news manager, you validate and summarize reliable news relevant to the topic.
            
                    Respond with json of this format: 
                    {
                        'headlines': most trending headlines, 
                        'sentiment': general sentiment on the market about the topic,
                        'description': detailed summary of relevant news,
                    }
                    """,
                    role=ROLE_SYSTEM,
                ),
                ai_client.make_msg(
                    text=f"""These are the news about the TON coin for the last week:
                    {news}
                    
                    Analyze and extract data relevant to cryptocurrencies and TON coin. Pay attention to anything fresh that could affect the market
                    """,
                ),
            ],
            format=JSON_MODE,
        )

        return filtered_news
