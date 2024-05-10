import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from datetime import datetime
import io
from src.utils import *
import numpy as np

# Your data

DEFAULT_FIG_SHAPE = (20, 8)


def make_candles_df(candles):
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

    return df


def make_candlebars(candles, useBuf=False, size=DEFAULT_FIG_SHAPE):
    """
    Pass candlebars data here, seems more or less standard format is: { t, o, c, h, l, v} - values;
    t - timestamp, o, c, h, l, v = open, close, high, low, volume
    returns BytesIO with figure if `useBuf` us True, None otherwise
    """

    df = make_candles_df(candles)

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


# normalize values by amplitued
def normalized_grid(grid):
    threshold = 0.00001
    min_value, max_value = grid.min(), grid.max()
    values_range = max(abs(max_value), abs(min_value)) * 2.0
    if values_range < threshold:
        return grid
    return grid / values_range


def normalize_heights(heights):
    if len(heights) == 0:
        return heights
    heights = np.array(heights)
    values = heights[:, 1]

    values = normalized_grid(values)

    heights[:, 1] = values
    return heights


def create_heatmap_grid(heights, height_range, display_range, heatmap_size):
    min_h, max_h = height_range
    w, h = heatmap_size
    heatmap_grid = np.zeros((h, w))

    # clamp and normalize data displayed
    heights = np.array(heights).astype(float)
    if display_range is not None and len(heights) > 0:
        range_low, range_high = display_range
        heights = [x for x in heights if x[1] >= range_low and x[1] <= range_high]
    heights = normalize_heights(heights)

    for height_data in heights:
        height = height_data[0]
        value = height_data[1]

        t = get_t(height, min_h, max_h)
        pixel_h = round(t * (h - 1))
        if pixel_h < 0 or pixel_h > h - 1:
            continue

        for i in range(0, w):
            heatmap_grid[pixel_h, i] += value

    return heatmap_grid


def plot_candles_overlay(
    candles,
    heights,
    display_range=None,
    figsize=(12, 8),
    heatmap_size=(2, 100),
    cmap="bwr",
):
    df = make_candles_df(candles)

    # Plot Configuration
    mc = mpf.make_marketcolors(
        up="#1a75ff", down="#ff3333", wick="black", volume="in", ohlc="i"
    )
    s = mpf.make_mpf_style(base_mpl_style="seaborn-darkgrid", marketcolors=mc)

    args = {
        "type": "candle",
        "style": s,
        "volume": False,
        # "ylabel": "Price",
        # "ylabel_lower": "Volume",
        "figratio": figsize,
        "title": "Stock Data with MACD",
        "returnfig": True,
    }

    fig, ax = mpf.plot(df, **args)
    ax = ax[0]
    [min_h, max_h] = ax.get_ylim()
    [min_x, max_x] = ax.get_xlim()

    heatmap_grid = create_heatmap_grid(
        heights,
        height_range=(min_h, max_h),
        display_range=display_range,
        heatmap_size=heatmap_size,
    )

    ## Plotting and Overlaying the heatmap
    cmap_heatmap = plt.get_cmap(cmap)
    ax.imshow(
        heatmap_grid,
        extent=[min_x, max_x, min_h, max_h],
        cmap=cmap_heatmap,
        aspect="auto",
        interpolation="bicubic",
        alpha=0.8,
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
    )
    mpf.show()


def plot_support_heatmap(
    data_points,
    heights,
    display_range=None,
    figsize=(12, 8),
    heatmap_size=(2, 100),
    cmap="bwr",
):
    """
    data_points - array of datapoints [[x, y] ... ]
    heights - array of values at y [[y, value] ... ]
    range - where to clamp values to
    """

    x_scatter = [x[0] for x in data_points]
    y_scatter = [x[1] for x in data_points]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_scatter, y_scatter, label="Data Points")
    [min_h, max_h] = ax.get_ylim()
    [min_x, max_x] = ax.get_xlim()

    heatmap_grid = create_heatmap_grid(
        heights,
        height_range=(min_h, max_h),
        display_range=display_range,
        heatmap_size=heatmap_size,
    )

    ## Plotting and Overlaying the heatmap
    cmap_heatmap = plt.get_cmap(cmap)
    ax.imshow(
        heatmap_grid,
        extent=[min_x, max_x, min_h, max_h],
        cmap=cmap_heatmap,
        aspect="auto",
        interpolation="bicubic",
        alpha=0.8,
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
    )
    plt.show()
