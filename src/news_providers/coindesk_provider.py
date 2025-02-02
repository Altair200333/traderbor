import requests
from src.rsc_object_extractor import *
import pydash
from src.utils import *


class CoinDeskProvider:
    def __init__(self):
        self.api_url = "https://www.coindesk.com/latest-crypto-news"

    def _extract_item(self, item):
        title = item.get("title")
        description = item.get("description")
        date = pydash.get(item, "date.publishedAt") or None
        url = pydash.get(item, "pathname") or None

        return {
            "title": title,
            "description": description,
            "date": str(convert_to_datetime(date)) if date else "unknown",
            "url": "https://www.coindesk.com" + url if url is not None else None,
        }

    def _is_news_item(self, item):
        if not item or not isinstance(item, dict):
            return False
        title = item.get("title")
        description = item.get("description")
        return bool(title) and bool(description)

    def _extract_news(self, rows):
        objects = RscObjectExtractor.extract_objects(rows)

        queue = []
        news_items = []

        for item in objects:
            queue.append(item)

        while queue:
            item = queue.pop()
            if isinstance(item, list):
                queue.extend(item)
            elif self._is_news_item(item):
                news_items.append(item)
            elif item and isinstance(item, dict):
                queue.extend(item.values())

        return [self._extract_item(x) for x in news_items]

    def get_news(self, **args):
        """
        Fetch news articles from coindesk.com
        """

        headers = {
            "rsc": "1"  # request react server component respons
        }
        response = requests.get(self.api_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            items = response.text.split("\n")
            return self._extract_news(items)
        else:
            print(f"[CoinDeskProvider] Error: {
                  response.status_code} - {response.text}")
            return []
