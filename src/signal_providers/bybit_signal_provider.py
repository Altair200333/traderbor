import requests
from src.utils import *


class BybitProvider:
    def __init__(self):
        self.url = "http://api-testnet.bybit.com"

    def _transform_response(self, data):
        candle_list = data["result"]["list"]

        candles = [
            {
                "t": float(x[0]),
                "o": float(x[1]),
                "h": float(x[2]),
                "l": float(x[3]),
                "c": float(x[4]),
                "v": float(x[5]),  # use turnover
            }
            for x in candle_list
        ]

        candles.sort(key=lambda x: x["t"])

        return candles

    def get_history(
        self,
        coin,
        start_date,
        end_date,
        price="USDT",
        resolution="15",
        category="linear",
        limit=1000,
    ):
        """
        Get candlebars signal of coin
        resolution: discreteness of candles, possible values: 1, 5, 15, 30, 45, 60, 120, 180, 240, D, W, M
        coin: coin of interest (BTC, ETH, TON etc)
        price: second coin of pair (usually usdt)

        response: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        """
        start_date = get_timestamp(start_date)
        end_date = get_timestamp(end_date)

        url = self.url + "/v5/market/kline"
        params = {
            "category": category,
            "symbol": coin + price,
            "interval": resolution,
            "start": start_date * 1000,
            "end": end_date * 1000,
            "limit": limit,
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return self._transform_response(data)
        else:
            return response.status_code, response.text
