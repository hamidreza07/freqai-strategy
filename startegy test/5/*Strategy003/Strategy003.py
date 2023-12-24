
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

class Strategy003(IStrategy):
    """
    Strategy 003
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy003
    """

    # Buy hyperspace params:
    buy_params = {
        "buy_ema_enabled": False,
        "buy_fastd_enabled": True,
        "buy_fisher": 0.01,
        "buy_fisher_enabled": True,
        "buy_mfi": 30.0,
        "buy_mfi_enabled": True,
        "buy_rsi": 11.0,
        "buy_rsi_enabled": True,
        "buy_sma_enabled": False,
    }

    buy_rsi = DecimalParameter(0, 50, decimals=0, default=11, space="buy")
    buy_mfi = DecimalParameter(0, 50, decimals=0, default=30, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=0.01, space="buy")

    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_fastd_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_fisher = DecimalParameter(-1, 1, decimals=2, default=0.3, space="sell")

    sell_fisher_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_sar_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_hold_enabled = CategoricalParameter([True, False], default=True, space="sell")


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
    use_exit_signal = Config.use_exit_signal
    exit_profit_only = Config.exit_profit_only
    ignore_roi_if_entry_signal = Config.ignore_roi_if_entry_signal
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
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        conditions = []

        # GUARDS AND TRENDS
        if self.buy_rsi_enabled.value:
            conditions.append(
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['rsi'] > 0)
            )

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['close'] < dataframe['sma'])

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_fisher.value)

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        if self.buy_ema_enabled.value:
            conditions.append(
                (dataframe['ema50'] > dataframe['ema100']) |
                (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10']))
            )

        if self.buy_fastd_enabled.value:
            conditions.append(
                (dataframe['fastd'] > dataframe['fastk']) &
                (dataframe['fastd'] > 0)
            )

            # build the dataframe using the conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        conditions = []

        # if hold, then don't set a sell signal
        if self.sell_hold_enabled.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        else:
            if self.sell_sar_enabled.value:
                conditions.append(dataframe['sar'] > dataframe['close'])

            if self.sell_fisher_enabled.value:
                conditions.append(dataframe['fisher_rsi'] > self.sell_fisher.value)

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

# +--------+---------+----------+--------------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------+
# |   Best |   Epoch |   Trades |    Win  Draw  Loss  Win% |   Avg profit |                        Profit |    Avg duration |   Objective |           Max Drawdown (Acct) |
# |--------+---------+----------+--------------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------|
# | * Best |    1/50 |       10 |      8     0     2  80.0 |        0.84% |       858.136 USDT    (8.58%) | 1 days 05:20:00 |    -858.136 |        10.677 USDT    (0.10%) |
# | * Best |    3/50 |       33 |     16     1    16  48.5 |        0.60% |      2135.155 USDT   (21.35%) | 0 days 22:00:00 | -2,135.15474 |       188.418 USDT    (1.53%) |
# | * Best |   15/50 |       43 |     22     2    19  51.2 |        0.49% |      2310.068 USDT   (23.10%) | 0 days 14:49:00 | -2,310.06849 |       103.756 USDT    (0.84%) |
# | * Best |   24/50 |       29 |     16     1    12  55.2 |        0.81% |      2568.430 USDT   (25.68%) | 1 days 01:28:00 | -2,568.43027 |       209.025 USDT    (1.64%) |
# Epochs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━