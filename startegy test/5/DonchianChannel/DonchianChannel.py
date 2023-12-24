
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


class DonchianChannel(IStrategy):
    """
    Simple strategy based on Donchian Channel Breakouts

    How to use it?
    > python3 ./freqtrade/main.py -s DonchianChannel
    """

    # Hyperparameters
    buy_params = {
        "buy_adx": 2.0,
        "buy_adx_enabled": False,
        "buy_dc_period": 13,
        "buy_dm_enabled": False,
        "buy_ema_enabled": False,
        "buy_fisher": 0.06,
        "buy_fisher_enabled": True,
        "buy_macd_enabled": True,
        "buy_mfi": 5.0,
        "buy_mfi_enabled": True,
        "buy_sar_enabled": False,
        "buy_sma_enabled": True,
    }

    buy_dc_period = IntParameter(1, 50, default=27, space="buy")
    buy_adx = DecimalParameter(1, 99, decimals=0, default=30, space="buy")
    buy_mfi = DecimalParameter(1, 99, decimals=0, default=50, space="buy")
    buy_fisher = DecimalParameter(-1.0, 1.0, decimals=2, default=0.81, space="buy")

    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_dm_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    sell_adx_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_dm_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_sma_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_ema_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_sar_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_macd_enabled = CategoricalParameter([True, False], default=False, space="sell")

    # set the startup candles count to the longest average used (SMA, EMA etc)
    startup_candle_count = max(buy_dc_period.value, 20)

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

        # Bollinger Bands
        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_diff"] = (dataframe["bb_upperband"] - dataframe["close"])

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_mid'] = ((dataframe['dc_upper'] + dataframe['dc_lower']) / 2)
        dataframe['dc_diff'] = (dataframe['dc_upper'] - dataframe['close'])

        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper']  - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib

        #print("\nupper: ", dataframe['dc_upper'])
        #print("\nlower: ", dataframe['dc_lower'])
        #print("\nmid: ", dataframe['dc_mid'])

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabolic
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        #print("\nSMA: ", dataframe['sma'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS

        # check that volume is not 0 (can happen in testing, or if there are issues with exchange data)
        # conditions.append(dataframe['volume'] > 0)

        # during back testing, data can be undefined, so check
        # conditions.append(dataframe['dc_hf'].notnull())

        if self.buy_sar_enabled.value:
            conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['sma200'].notnull())
            conditions.append(dataframe['close'] > dataframe['sma200'])

        if self.buy_ema_enabled.value:
            conditions.append(dataframe['ema50'].notnull())
            conditions.append(dataframe['close'] > dataframe['ema50'])

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'].notnull())
            conditions.append(dataframe['mfi'] >= self.buy_mfi.value)

        # ADX with DM+ > DM- indicates uptrend
        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] >= self.buy_adx.value)

        if self.buy_dm_enabled.value:
            conditions.append(dataframe['dm_delta'] > 0)

        if self.buy_macd_enabled.value:
            conditions.append(dataframe['macd'] > dataframe['macdsignal'])

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi']  < self.buy_fisher.value)

        # TRIGGERS

        # 2 green candles, one crosses or jumps above high band
        #level = 'dc_hf'
        level = 'dc_upper'

        conditions.append(
            (dataframe[level].notnull()) &
            (dataframe['close'] >= dataframe[level])
            # (
            #         (dataframe['close'] >= dataframe['open']) &
            #         (dataframe['close'].shift(1) >= dataframe['open'].shift(1)) &
            #         (
            #                 ( # current candle crosses HF band
            #                          (qtpylib.crossed_above(dataframe['close'], dataframe[level]))
            #                 ) |
            #                 ( # previous candle crosses HF band
            #                         (qtpylib.crossed_above(dataframe['close'].shift(1), dataframe[level].shift(1)))
            #                 ) |
            #                 ( # current candle jumped higher than HF band (but may noy have crossed)
            #                         (dataframe['close'] >= dataframe[level]) &
            #                         (dataframe['close'].shift(1) < dataframe[level].shift(1))
            #                 ) |
            #                 ( # 2 candles close above HF band
            #                         (dataframe['close'] >= dataframe[level]) &
            #                         (dataframe['close'].shift(1) >= dataframe[level].shift(1))
            #                 )
            #         )
            # )
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

        # if hold, then don't set a sell signal
        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0
            return dataframe

        conditions = []
        #level = 'dc_lf'
        level = 'dc_lower'

        # red candle crosses or any candles jump below low band
        conditions.append(
            (dataframe['dc_lf'].notnull()) &
            (
                    (
                            (dataframe['close'] < dataframe['open']) &
                            (qtpylib.crossed_below(dataframe['close'], dataframe[level]))
                    ) |
                    (
                            (dataframe['close'] <= dataframe[level]) &
                            (dataframe['close'].shift(1) > dataframe[level].shift(1))
                    )
            )
        )

        # The following conditions ar ORd, i.e. any one of them will trigger a sell
        # These should be strong sell signals
        orconditions = []

        if self.sell_sar_enabled.value:
            #orconditions.append(dataframe['sar'].notnull())
            orconditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['sar']))

        if self.sell_sma_enabled.value:
            #orconditions.append(dataframe['sma200'].notnull())
            orconditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['sma200']))

        if self.sell_ema_enabled.value:
            #orconditions.append(dataframe['ema50'].notnull())
            orconditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['ema50']))

        if self.sell_adx_enabled.value:
            conditions.append(dataframe['adx'] < self.buy_adx.value)

        if self.sell_dm_enabled.value:
            conditions.append(dataframe['dm_delta'] < 0)

        if self.sell_macd_enabled.value:
            orconditions.append(qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))


        # build the dataframe using the conditions
        r1 = False
        r2 = False
        if conditions:
             r1 = reduce(lambda x, y: x & y, conditions)

        if orconditions:
            r2 = reduce(lambda x, y: x | y, orconditions)

        dataframe.loc[(r1 | r2), 'sell'] = 1

        if orconditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, orconditions),
                'sell'] = 1

        return dataframe