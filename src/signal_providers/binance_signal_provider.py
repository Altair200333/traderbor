import requests
from src.utils import *


class BinanceProvider:
    def __init__(self):
        self.url = "https://api.binance.com"
        self.resolution_map = {
            "1": "1m",
            "3": "3m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "120": "2h",
            "240": "4h",
            "360": "6h",
            "480": "8h",
            "D": "1d",
            "W": "1w",
            "M": "1m",
        }

    def _transform_response(self, data):
        candle_list = data

        candles = [
            {
                "t": float(x[0]),
                "o": float(x[1]),
                "h": float(x[2]),
                "l": float(x[3]),
                "c": float(x[4]),
                "v": float(x[5]),
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
        limit=1000,
    ):
        """
        Fetch candlestick data from Binance API.

        :param symbol: Symbol of the cryptocurrency pair (e.g., 'BTCUSDT').
        :param interval: Time frame interval (e.g., '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M').
        :param limit: Number of candlesticks to fetch.
        :return: List of candlestick data.
        """

        url = self.url + "/api/v3/klines"

        start_date = get_timestamp(start_date)
        end_date = get_timestamp(end_date)

        params = {
            "symbol": coin + price,
            "interval": self.resolution_map[resolution],
            "limit": limit,
            "startTime": start_date * 1000,
            "endTime": end_date * 1000,
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return self._transform_response(data)
        else:
            return response.status_code, response.text
