import logging

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade
from datetime import datetime

logger = logging.getLogger(__name__)


class TrendMomoClassifier(IStrategy):

    plot_config = {
        "main_plot": {},
        "subplots": {
            "Up_or_down": {
                '&s-up_or_down': {'color': 'green'},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    custom_info = {
        'risk_reward_ratio': 2.5,
        'set_to_break_even_at_profit': 1,
    }

    process_only_new_candles = True
    stoploss = -0.03
    use_custom_stoploss = True

    use_exit_signal = True
    can_short = True

    buy_stoch_rsi = IntParameter(
        low=1, high=20, default=10, space='buy', optimize=True, load=True)
    sell_stoch_rsi = IntParameter(
        low=80, high=100, default=90, space='sell', optimize=True, load=True)
    short_stoch_rsi = IntParameter(
        low=80, high=100, default=90, space='sell', optimize=True, load=True)
    exit_short_stoch_rsi = IntParameter(
        low=1, high=20, default=10, space='buy', optimize=True, load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
            custom_stoploss using a risk/reward ratio https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/FixedRiskRewardLoss.py
        """
        result = break_even_sl = takeprofit_sl = -1
        custom_info_pair = self.custom_info.get(pair)

        if custom_info_pair is not None:
            # using current_time/open_date directly via custom_info_pair[trade.open_daten]
            # would only work in backtesting/hyperopt.
            # in live/dry-run, we have to search for nearest row before it
            open_date_mask = custom_info_pair.index.unique().get_loc(
                trade.open_date_utc, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            # trade might be open too long for us to find opening candle
            if(len(open_df) != 1):
                return -1  # won't update current stoploss

            initial_sl_abs = open_df['stoploss_rate']

            # calculate initial stoploss at open_date
            initial_sl = initial_sl_abs/current_rate-1

            # calculate take profit treshold
            # by using the initial risk and multiplying it
            risk_distance = trade.open_rate-initial_sl_abs
            reward_distance = risk_distance * \
                self.custom_info['risk_reward_ratio']
            # take_profit tries to lock in profit once price gets over
            # risk/reward ratio treshold
            take_profit_price_abs = trade.open_rate+reward_distance
            # take_profit gets triggerd at this profit
            take_profit_pct = take_profit_price_abs/trade.open_rate-1

            # break_even tries to set sl at open_rate+fees (0 loss)
            break_even_profit_distance = risk_distance * \
                self.custom_info['set_to_break_even_at_profit']
            # break_even gets triggerd at this profit
            break_even_profit_pct = (
                break_even_profit_distance+current_rate)/current_rate-1

            result = initial_sl
            if(current_profit >= break_even_profit_pct):
                break_even_sl = (
                    trade.open_rate*(1+trade.fee_open+trade.fee_close) / current_rate)-1
                result = break_even_sl

            if(current_profit >= take_profit_pct):
                takeprofit_sl = take_profit_price_abs/current_rate-1
                result = takeprofit_sl

        return result

    def feature_engineering_expand_all(self, df: DataFrame, period, **kwargs):

        df[["%-stoch_rsi_K-period", "%-stoch_rsi_D-period"]] = ta.STOCHRSI(
            df, timeperiod=period)[["fastk", "fastd"]]

        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])
               ) / (df['high'] - df['low']) * df['volume']

        df["%-A/D-period"] = mfv.rolling(period).sum()

        df["%-relative_volume-period"] = (df["volume"] /
                                          df["volume"].rolling(period).mean())

        return df

    def feature_engineering_expand_basic(self, df: DataFrame, **kwargs):

        df["%-pct-change"] = df["close"].pct_change()
        df["%-raw_volume"] = df["volume"]
        df["%-raw_price"] = df["close"]
        return df

    def feature_engineering_standard(self, df: DataFrame, **kwargs):

        df["%-day_of_week"] = df["date"].dt.dayofweek
        df["%-hour_of_day"] = df["date"].dt.hour

        return df

    def set_freqai_targets(self, df: DataFrame, **kwargs):

        df['&s-up_or_down'] = np.where(df["close"].shift(-50) >
                                       df["close"], 'up', 'down')

        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:

        df = self.freqai.start(df, metadata, self)

        df[["stoch_rsi_K", "stoch_rsi_D"]] = ta.STOCHRSI(
            df)[["fastk", "fastd"]]

        df['stoploss_rate'] = df['close']-(ta.ATR(df)*2)

        self.custom_info[metadata['pair']] = df[[
            'date', 'stoploss_rate']].copy().set_index('date')

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                # Signal: stochRSI below 20
                (df["stoch_rsi_K"] <= self.buy_stoch_rsi.value) &
                (qtpylib.crossed_above(df["stoch_rsi_K"], df["stoch_rsi_D"])) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                # Make sure Freqai is confident in the prediction
                (df['do_predict'] == 1) &
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            'enter_long'] = 1

        df.loc[
            (
                # Signal: stochRSI above 80
                (df["stoch_rsi_K"] >= self.short_stoch_rsi.value) &
                (qtpylib.crossed_above(df["stoch_rsi_D"], df["stoch_rsi_K"])) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                # Make sure Freqai is confident in the prediction
                (df['do_predict'] == 1) &
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'down')
            ),
            'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                # Signal: stochRSI above 80
                (df["stoch_rsi_K"] <= self.sell_stoch_rsi.value) &
                (qtpylib.crossed_above(df["stoch_rsi_D"], df["stoch_rsi_K"])) &
                (df['volume'] > 0)
            ),
            'exit_long'] = 1

        df.loc[
            (
                # Signal: stoch RSI below 20
                (df["stoch_rsi_K"] <= self.exit_short_stoch_rsi.value) &
                (qtpylib.crossed_above(df["stoch_rsi_K"], df["stoch_rsi_D"])) &
                (df['volume'] > 0)
            ),
            'exit_short'] = 1

        return df