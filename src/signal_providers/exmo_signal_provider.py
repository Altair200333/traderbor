import requests


class ExmoProvider:
    def __init__(self):
        self.url = "https://api.exmo.com/v1"

    def get_history(
        self,
        coin,
        start_date,
        end_date,
        price="USDT",
        resolution="15",
    ):
        """
        Get candlebars signal of coin
        resolution: discreteness of candles, possible values: 1, 5, 15, 30, 45, 60, 120, 180, 240, D, W, M
        coin: coin of interest (BTC, ETH, TON etc)
        price: second coin of pair (usually usdt)
        """

        url = self.url + "/candles_history"
        params = {
            "symbol": coin + "_" + price,
            "resolution": resolution,
            "from": start_date,
            "to": end_date,
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["candles"]
        else:
            return response.status_code, response.text
