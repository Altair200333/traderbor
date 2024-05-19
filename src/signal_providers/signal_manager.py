from src.signal_providers.binance_signal_provider import BinanceProvider
from src.signal_providers.exmo_signal_provider import ExmoProvider
from src.signal_providers.bybit_signal_provider import BybitProvider
from src.utils import *


global_signal_provider = BinanceProvider()


def get_12h_history(coin, end_date=None):
    end_date = end_date or get_current_datetime()
    start_date = subtract_time(end_date, hours=12)

    return global_signal_provider.get_history(
        coin, start_date=start_date, end_date=end_date, resolution="5"
    )


def get_day_history(coin, end_date=None):
    end_date = end_date or get_current_datetime()
    start_date = subtract_time(end_date, hours=24)

    return global_signal_provider.get_history(
        coin, start_date=start_date, end_date=end_date, resolution="15"
    )


def get_week_history(coin, end_date=None):
    end_date = end_date or get_current_datetime()
    start_date = subtract_time(end_date, days=7)

    return global_signal_provider.get_history(
        coin, start_date=start_date, end_date=end_date, resolution="120"
    )


def get_month_history(coin, end_date=None):
    end_date = end_date or get_current_datetime()
    start_date = subtract_time(end_date, days=30)

    return global_signal_provider.get_history(
        coin, start_date=start_date, end_date=end_date, resolution="240"
    )


def get_year_history(coin, end_date=None):
    end_date = end_date or get_current_datetime()
    start_date = subtract_time(end_date, days=365)

    return global_signal_provider.get_history(
        coin, start_date=start_date, end_date=end_date, resolution="W"
    )
