import pydash
from src.ai_chat import AiChat
from src.api import BasicApiClient
from src.news_providers.coindesk_provider import CoinDeskProvider
from src.news_providers.tradingview_provider import TradingViewProvider
from src.news_providers.cryptopanic_provider import CryptoPanicProvider

from src.openai_api import ai_client
from src.const import *
from src.utils import *

NEWS_CHUNK_SIZE = 100


class NewsFilteringAgent:
    def __init__(self) -> None:
        pass

    def _get_provider(self, provider):
        """
        Basically factory, let it live here because it makes life easier
        """
        if provider == "cryptopanic":
            return CryptoPanicProvider()
        elif provider == "tradingview":
            return TradingViewProvider()
        elif provider == "coindesk":
            return CoinDeskProvider()

    def _get_news_by_provider(self, provider, coin, **args):
        provider = self._get_provider(provider)
        try:
            return provider.get_news(coin=coin, **args)
        except Exception as e:
            print(f"Error fetching news from {provider}: {e}")
            return []

    def _extract_news_points(self, news, coin):
        print(f"Extracting news points for {coin} {len(news)}")
        chat = AiChat(client=BasicApiClient(model="gpt-4o"))

        chat.message(
            """You are news manager, you validate and summarize reliable news relevant to the topic of {coin}.
            First remove very obvious clickbaits, or news not relevant to the topic of {coin}.
            Then summarize the news into bullet points that represet the most important trends and events for the coin.
            Include anything that can affect price/market or user's opinion.
            Responds with JSON of following format:
            Array of objects with the following fields:
            
            {{
                "title": "Title of the bullet point",
                "summary": "What the group of news means for the coin",
                "importance": "How important is this news for the coin",
                "headlines": ["List of headlines that are relevant to the point"]
            }}""", role=ROLE_SYSTEM)

        response = chat.message(
            f"""
            Extract 5 points at most following the structure.
            Respond with JSON: [
                reasoning: "thing about the news, trends and their correlations. Which news are important and which are not?",
                points: ["array of poitns with described structure"]
                ]
            This is list of relevant news for the coin {coin}:
            News:
            {json.dumps(news, indent=4)}
            """,
            format=JSON_MODE
        )

        try:
            return json.loads(response)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return []

    def get_news(self, coin):
        # run requests in parallel

        news_by_provider = map_async(
            lambda: self._get_news_by_provider(
                "cryptopanic", coin, kind="news"),
            lambda: self._get_news_by_provider(
                "cryptopanic", coin, kind="media"),
            lambda: self._get_news_by_provider("tradingview", coin),
            lambda: self._get_news_by_provider("coindesk", coin),
        )

        news = []
        for provider_news in news_by_provider:
            news.extend(provider_news)

        # extract fields relevant for processing the news using AI
        news_items = [{"title": pydash.get(n, "title", ""),
                       "description": pydash.get(n, "description", "")} for n in news]

        # divide into chunks to avoid context overflow
        chunks = pydash.chunk(news_items, NEWS_CHUNK_SIZE)

        # run requests in parallel to make it faster
        responses = map_async(
            *[lambda chunk=chunk: self._extract_news_points(news=chunk, coin=coin) for chunk in chunks])

        news_points = []
        for response in responses:
            # there is also reasoning in the response, ignore it as it is not needed
            news_points.extend(pydash.get(response, "points", []))

        return news_points
