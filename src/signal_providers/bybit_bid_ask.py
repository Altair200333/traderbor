import requests
from src.utils import *


class BybitBidAskProvider:
    def __init__(self):
        self.url = "http://api-testnet.bybit.com"

    def _transform_response(self, data):
        bid = data["result"]["b"]
        ask = data["result"]["a"]

        b = [{"price": x[0], "amount": x[1]} for x in bid]
        a = [{"price": x[0], "amount": x[1]} for x in ask]

        return {"bid": b, "ask": a}

    def get_data(
        self,
        coin,
        category="linear",
        price="USDT",
        limit=500,
    ):
        url = self.url + "/v5/market/orderbook"
        params = {
            "category": category,
            "symbol": coin + price,
            "limit": limit,
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return self._transform_response(data)
        else:
            return response.status_code, response.text
