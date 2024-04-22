import requests
import os
from src.utils import *
import pydash


class NewsApi:
    def __init__(self):
        self.api_url = "https://newsapi.org/v2/everything"
        self.api_key = os.environ["NEWS_API_KEY"]

    def _extract_item(self, item):
        source = pydash.get(item, "source.name") or "unknown"
        title = pydash.get(item, "title") or "unknown"
        description = pydash.get(item, "description") or "unknown"
        date = pydash.get(item, "publishedAt") or "unknown"
        return {
            "source": source,
            "title": title,
            "description": description,
            "date": date,
        }

    def _format_date(self, date):
        f_date = convert_to_datetime(date)
        return f_date.strftime("%Y-%m-%d")

    def get_news(self, coin, start_date=None, end_date=None, sort_by="relevancy"):
        """
        Fetch news articles based on a query from a specific date, sorted by a criterion.

        query: Search term, e.g., "Toncoin"
        date_from: Start date for news, format "YYYY-MM-DD"
        sort_by: Sorting criterion ('relevancy', 'publishedAt', 'popularity')
        """

        params = {
            "q": coin,
            "sortBy": sort_by,
            "apiKey": self.api_key,
        }
        if start_date:
            params["from"] = self._format_date(start_date)
        if end_date:
            params["to"] = self._format_date(end_date)

        response = requests.get(self.api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data["articles"]
            news = [self._extract_item(x) for x in articles]
            return news
        else:
            raise Exception(
                f"Failed to fetch news: {response.status_code} - {response.text}"
            )
