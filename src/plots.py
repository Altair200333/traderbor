import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from datetime import datetime
import io

# Your data


def make_candlebars(candles, useBuf=False):
    """
    Pass candlebars data here, seems more or less standard format is: { t, o, c, h, l, v} - values;
    t - timestamp, o, c, h, l, v = open, close, high, low, volume
    returns BytesIO with figure if `useBuf` us True, None otherwise
    """

    data = {"candles": candles}

    # Transforming data into a DataFrame
    df = pd.DataFrame(data["candles"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("Date", inplace=True)
    df.drop(columns=["t"], inplace=True)

    df = df[["o", "h", "l", "c", "v"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]  # Renaming for clarity

    # TODO customize colors (not important really, it's just something vivid for gpt)
    mc = mpf.make_marketcolors(
        up="#1a75ff",
        down="#ff3333",
        wick="black",
        volume="in",
        ohlc="i",
    )
    s = mpf.make_mpf_style(base_mpl_style="seaborn-v0_8-pastel", marketcolors=mc)

    args = {
        "type": "candle",
        "style": s,
        "ylabel": "Price",
        "volume": True,
        "ylabel_lower": "Volume",
        "figratio": (10, 8),
        "figscale": 1.2,
    }

    buf = io.BytesIO() if useBuf is True else None

    if buf is not None:
        args["savefig"] = buf

    # Plotting candlestick chart with the custom style
    mpf.plot(
        df,
        **args,
    )

    return buf
