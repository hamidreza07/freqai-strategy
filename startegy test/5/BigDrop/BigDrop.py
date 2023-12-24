
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

import Config



class BigDrop(IStrategy):
    """
    Simple strategy that looks for a big drop and then buys
    This version doesn't issue a sell signal, just holds until ROI or stoploss kicks in

    How to use it?
    > python3 ./freqtrade/main.py -s BigDrop

    """

    # Hyperparameters
    # Buy hyperspace params:
    buy_params = Config.strategyParameters["BigDrop"]


    buy_num_candles = IntParameter(2, 9, default=9, space="buy")
    buy_drop = DecimalParameter(0.01, 0.06, decimals=3, default=0.06, space="buy")

    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.23, space="buy")
    buy_mfi = DecimalParameter(10, 40, decimals=0, default=31.0, space="buy")

    # Categorical parameters that control whether a trend/check is used or not
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_bb_enabled = CategoricalParameter([True, False], default=False, space="buy")

    # set the startup candles count to the longest average used (EMA, EMA etc)
    startup_candle_count = 20

    # set common parameters
    minimal_roi = Config.minimal_roi
    trailing_stop = Config.trailing_stop
    trailing_stop_positive = Config.trailing_stop_positive
    trailing_stop_positive_offset = Config.trailing_stop_positive_offset
    trailing_only_offset_is_reached = Config.trailing_only_offset_is_reached
    stoploss = Config.stoploss
    timeframe = Config.timeframe
    process_only_new_candles = Config.process_only_new_candles
    use_exit_signal = Config.use_sell_signal
    exit_profit_only = Config.sell_profit_only
    ignore_roi_if_entry_signal = Config.ignore_roi_if_buy_signal
    order_types = Config.order_types

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger bands
        #bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # A little different than normal - adjust band values based on buy_bb_ratio
        #dataframe['bb_upperband'] = bollinger['mid'] + (bollinger['upper']-bollinger['mid'])*self.buy_bb_uratio.value
        #dataframe['bb_middleband'] = bollinger['mid']
        #dataframe['bb_lowerband'] = bollinger['mid'] - (bollinger['mid']-bollinger['lower'])*self.buy_bb_lratio.value
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS
        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_fisher.value)

        if self.buy_bb_enabled.value:
            conditions.append(dataframe['close'] <= dataframe['bb_lowerband'])

        # TRIGGERS

        # big enough drop?
        conditions.append(
            (((dataframe['open'].shift(self.buy_num_candles.value-1) - dataframe['close']) /
            dataframe['open'].shift(self.buy_num_candles.value-1)) >= self.buy_drop.value)
        )
        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """
        # Don't sell (have to set something in 'sell' column)
        dataframe.loc[(dataframe['close'] >= 0), 'sell'] = 0
        return dataframe