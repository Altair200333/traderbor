import requests
import os
from src.utils import *
import pydash


class TradingViewProvider:
    def __init__(self, source="BINANCE") -> None:
        self.source = source
        self.api_url = "https://news-headlines.tradingview.com/headlines/"

    def _extract_item(self, item):
        queue = [x for x in item["astDescription"]["children"]]
        result = ""
        while len(queue) > 0:
            element = queue.pop(0)
            if isinstance(element, str):
                result = result + element
            else:
                children = element["children"] if "children" in element else []
                for child in children:
                    queue.insert(0, child)

        date = pydash.get(item, "published")
        return {
            "source": pydash.get(item, "provider"),
            "title": pydash.get(item, "title") or "unknown",
            "description": result,
            "date": str(convert_to_datetime(date)) if date else "unknown",
        }

    def get_news(self, coin):
        params = {
            "lang": "en",
            "symbol": f"{self.source}:{coin}USDT",
        }

        response = requests.get(self.api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            news = [self._extract_item(x) for x in data]
            return news
        else:
            raise Exception(
                f"Failed to fetch news: {response.status_code} - {response.text}"
            )
