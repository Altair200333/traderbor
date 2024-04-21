import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from datetime import datetime
import io

# Your data

DEFAULT_FIG_SHAPE = (20, 8)


def make_candlebars(candles, useBuf=False, size=DEFAULT_FIG_SHAPE):
    """
    Pass candlebars data here, seems more or less standard format is: { t, o, c, h, l, v} - values;
    t - timestamp, o, c, h, l, v = open, close, high, low, volume
    returns BytesIO with figure if `useBuf` us True, None otherwise
    """

    df = pd.DataFrame(candles)
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("Date", inplace=True)
    df.rename(
        columns={
            "t": "Time",
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        },
        inplace=True,
    )

    # Calculate MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # Plot Configuration
    mc = mpf.make_marketcolors(
        up="#1a75ff", down="#ff3333", wick="black", volume="in", ohlc="i"
    )
    s = mpf.make_mpf_style(base_mpl_style="seaborn-darkgrid", marketcolors=mc)

    # Create MACD and Signal plots
    apds = [
        mpf.make_addplot(macd, panel=2, color="fuchsia", ylabel="MACD"),
        mpf.make_addplot(signal, panel=2, color="b"),
    ]

    # Generate Plot
    # mpf.figure(figsize=size)

    args = {
        "type": "candle",
        "style": s,
        "volume": True,
        # "ylabel": "Price",
        # "ylabel_lower": "Volume",
        "addplot": apds,
        "panel_ratios": (6, 3, 2),
        "figratio": size,
        "title": "Stock Data with MACD",
    }

    buf = io.BytesIO() if useBuf is True else None

    if buf is not None:
        args["savefig"] = buf

    mpf.plot(df, **args)

    return buf
