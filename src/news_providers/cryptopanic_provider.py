import os
import requests
import pydash


class CryptoPanicProvider:
    def __init__(self):
        self.api_key = os.environ["CRYPTOPANIC_API_KEY"]
        self.api_url = "https://cryptopanic.com/api/v1/posts/"

    def _extract_item(self, item):
        return {
            "title": item.get("title"),
        }

    def get_news(self, coin, kind="news", **args):
        """
        Fetch news articles based on a query.
        query: Search term, e.g., "ETH / XRP"
        kind: "news" or "media"
        """

        params = {
            "auth_token": self.api_key,
            "currencies": coin,
            "kind": kind,
        }

        # Make the GET request
        response = requests.get(self.api_url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            news = [self._extract_item(x) for x in data.get("results", [])]
            return news
        else:
            print(f"[CryptoPanicProvider] Error: {
                  response.status_code} - {response.text}")
            return []
