
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class kijun_cross_strong_s(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Proposed timeframe for the strategy. Can be altered to your own preferred timeframe.
    timeframe = "1d"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # Set to 10000% since the exit signal determines the trade exit.
    # Some crypto even got ROI triggered at 100% so had to set it to this value.
    minimal_roi = {"0": 100.0}

    # Optimal stoploss designed for the strategy.
    # Set to 100% since the exit signal dermines the trade exit.
    stoploss = -1.0

    # Trailing stoploss
    trailing_stop = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    # Set to the default of 30.
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "kijun": {"color": "blue"},
                'senkou_a': {
                    'color': 'green',
                    'fill_to': 'senkou_b',
                    'fill_label': 'Ichimoku Cloud',
                    'fill_color': 'rgba(255,76,46,0.2)', 
                },
                'senkou_b': {}
            }
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # CREATE ICHIMOKU INDICATOR
        # Specify the lenghts for each indicator (20, 60, 120, 60 is for crypto trading)

        TS = 9
        KS = 26
        SS = 52
        CS = 26
        OS = 0

        # Each column represents the output of the First Ichimoku Variable tuple and its specific column.
        dataframe["tenkan"] = pta.ichimoku(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            tenkan=TS,
            kijun=KS,
            senkou=SS,
            offset=OS,
        )[0][f"ITS_{TS}"]
        dataframe["kijun"] = pta.ichimoku(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            tenkan=TS,
            kijun=KS,
            senkou=SS,
            offset=OS,
        )[0][f"IKS_{KS}"]
        dataframe["senkou_a"] = pta.ichimoku(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            tenkan=TS,
            kijun=KS,
            senkou=SS,
            offset=OS,
        )[0][f"ISA_{TS}"]
        dataframe["senkou_b"] = pta.ichimoku(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            tenkan=TS,
            kijun=KS,
            senkou=SS,
            offset=OS,
        )[0][f"ISB_{KS}"]
        dataframe["chikou"] = pta.ichimoku(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            tenkan=TS,
            kijun=KS,
            senkou=SS,
            offset=OS,
        )[0][f"ICS_{KS}"]

        # Buy long singals are enhanced by explanations
        # Kijun should be above senkou_a AND kijun should be above senkou_b to make the signal TRUE
        dataframe["kijun_above_cloud"] = (dataframe["kijun"] > dataframe["senkou_a"]) & (dataframe["kijun"] > dataframe["senkou_b"])
        # Close price should also be above senkou_a AND senkou_b to let the signal be TRUE
        dataframe["close_above_cloud"] = (dataframe["close"] > dataframe["senkou_a"]) & (dataframe["close"] > dataframe["senkou_b"])
        # The third part of the total buy signal is when the close price is above the kijun
        dataframe["close_above_kijun"] = dataframe["close"] > dataframe["kijun"]
        # The buy long signal can only be True if the kijun_above_kumo, close_above_kumo and close_above_kijun are all True
        dataframe["buy_long"] = (
            (dataframe["kijun_above_cloud"] == True)
            & (dataframe["close_above_cloud"] == True)
            & (dataframe["close_above_kijun"] == True)
        )

        # The sell short signals are similar to buy long, but reversed
        dataframe["kijun_below_cloud"] = (dataframe["kijun"] < dataframe["senkou_a"]) & (dataframe["kijun"] < dataframe["senkou_b"])
        dataframe["close_below_cloud"] = (dataframe["close"] < dataframe["senkou_a"]) & (dataframe["close"] < dataframe["senkou_b"])
        dataframe["close_below_kijun"] = dataframe["close"] < dataframe["kijun"]
        dataframe["sell_short"] = (
            (dataframe["kijun_below_cloud"] == True)
            & (dataframe["close_below_cloud"] == True)
            & (dataframe["close_below_kijun"] == True)
        )

        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ("live", "dry_run"):
                ob = self.dp.orderbook(metadata["pair"], 1)
                dataframe["best_bid"] = ob["bids"][0][0]
                dataframe["best_ask"] = ob["asks"][0][0]

        # print(self)
        print(metadata)
        # print(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # If buy long signal is True, then enter a long trade
                (dataframe["buy_long"] == True) & (dataframe["volume"] > 0)  # Guard
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Strong_long_signal")

        # For short trades, use the section below
        dataframe.loc[
            (
                # If sell short signal is True, then enter a short trade
                (dataframe["sell_short"] == True) & (dataframe["volume"] > 0)  # Guard
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Strong_short_signal")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # The exit signal for long trades is pretty straightforward.
                # Sell when the close price is below the kijun sen
                (dataframe["close"] < dataframe["kijun"]) & (dataframe["volume"] > 0)  # Guard
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "Close_below_kijun")

        # For short trades, use the section below
        dataframe.loc[
            (
                # The exit signal for shorts trades is pretty straightforward.
                # Sell when the close price is above the kijun sen
                (dataframe["close"] > dataframe["kijun"]) & (dataframe["volume"] > 0)  # Guard
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "Close_above_kijun")

        return dataframe