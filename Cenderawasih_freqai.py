import logging
from functools import reduce
import math
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.freqai.strategy_bridge import CustomModel
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, merge_informative_pair
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class Cenderawasih_freqai(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.model = CustomModel(self.config)
    self.model.bridge.start(dataframe, metadata)

    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "prediction": {"prediction": {"color": "blue"}},
            "target_roi": {
                "target_roi": {"color": "brown"},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = False

    def informative_pairs(self):
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue  # avoid duplication
                informative_pairs.append((pair, tf))
        return informative_pairs

    def bot_start(self):
        self.model = CustomModel(self.config)

    def populate_any_indicators(self, metadata, pair, df, tf, informative=None,
            coin="", set_generalized_indicators=False):
        """
        Function designed to automatically generate, name and merge features
        from user indicated timeframes in the configuration file. User controls the indicators
        passed to the training/prediction by prepending indicators with `'%-' + coin `
        (see convention below). I.e. user should not prepend any supporting metrics
        (e.g. bb_lowerband below) with % unless they explicitly want to pass that metric to the
        model.
        :params:
        :pair: pair to be used as informative
        :df: strategy dataframe which will receive merges from informatives
        :tf: timeframe of the dataframe which will modify the feature names
        :informative: the dataframe associated with the informative pair
        :coin: the name of the coin which will modify the feature names.
        """

        with self.model.bridge.lock:
            if informative is None:
                informative = self.dp.get_pair_dataframe(pair, tf)

            # first loop is automatically duplicating indicators for time periods
            for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

                t = int(t)
                # informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
                # informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
                # informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, window=t)
                # informative[f"{coin}20sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
                informative[f"{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)
                informative[f"{coin}hma-period_{t}"] = tv_hma(informative, timeperiod=t)
                # informative[f"%-{coin}close_over_20sma-period_{t}"] = (
                #     informative["close"] / informative[f"{coin}20sma-period_{t}"]
                # )

                informative[f"%-{coin}close_below_ema-period_{t}"] = (
                    informative["close"] / informative[f"{coin}ema-period_{t}"]
                )

                informative[f"%-{coin}close_below_hma-period_{t}"] = (
                    informative["close"] / informative[f"{coin}hma-period_{t}"]
                )

                # informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)

                # bollinger = qtpylib.bollinger_bands(
                #     qtpylib.typical_price(informative), window=t, stds=2.2
                # )
                # informative[f"{coin}bb_lowerband-period_{t}"] = bollinger["lower"]
                # informative[f"{coin}bb_middleband-period_{t}"] = bollinger["mid"]
                # informative[f"{coin}bb_upperband-period_{t}"] = bollinger["upper"]

                # informative[f"%-{coin}bb_width-period_{t}"] = (
                #     informative[f"{coin}bb_upperband-period_{t}"]
                #     - informative[f"{coin}bb_lowerband-period_{t}"]
                # ) / informative[f"{coin}bb_middleband-period_{t}"]
                # informative[f"%-{coin}close-bb_lower-period_{t}"] = (
                #     informative["close"] / informative[f"{coin}bb_lowerband-period_{t}"]
                # )

                # informative[f"%-{coin}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

                # informative[f"%-{coin}relative_volume-period_{t}"] = (
                #     informative["volume"] / informative["volume"].rolling(t).mean()
                # )

            informative[f"%-{coin}rsi-period_14"] = ta.RSI(informative, timeperiod=14)
            informative[f"%-{coin}rsi-period_4"] = ta.RSI(informative, timeperiod=4)
            informative[f"%-{coin}pct-change"] = informative["close"].pct_change()
            informative[f"%-{coin}raw_volume"] = informative["volume"]
            informative[f"%-{coin}raw_price"] = informative["close"]

            indicators = [col for col in informative if col.startswith("%")]
            # This loop duplicates and shifts all indicators to add a sense of recency to data
            for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
                if n == 0:
                    continue
                informative_shift = informative[indicators].shift(n)
                informative_shift = informative_shift.add_suffix("_shift-" + str(n))
                informative = pd.concat((informative, informative_shift), axis=1)

            df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
            skip_columns = [
                (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
            ]
            df = df.drop(columns=skip_columns)

            # Add generalized indicators here (because in live, it will call this
            # function to populate indicators during training). Notice how we ensure not to
            # add them multiple times
            if set_generalized_indicators:
                df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
                df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

                # user adds targets here by prepending them with &- (see convention below)
                # If user wishes to use multiple targets, a multioutput prediction model
                # needs to be used such as templates/CatboostPredictionMultiModel.py
                df["&-s_close"] = (
                    df["close"]
                    .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
                    .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
                    .mean()
                    / df["close"]
                    - 1
                )

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]
        self.pair = metadata["pair"]
        sgi = True
        # the following loops are necessary for building the features
        # indicated by the user in the configuration file.
        # All indicators must be populated by populate_any_indicators() for live functionality
        # to work correctly.
        for tf in self.freqai_info["feature_parameters"]["include_timeframes"]:
            dataframe = self.populate_any_indicators(
                metadata,
                self.pair,
                dataframe.copy(),
                tf,
                coin=self.pair.split("/")[0] + "-",
                set_generalized_indicators=sgi,
            )
            sgi = False
            for pair in self.freqai_info["feature_parameters"]["include_corr_pairlist"]:
                if metadata["pair"] in pair:
                    continue  # do not include whitelisted pair twice if it is in corr_pairlist
                dataframe = self.populate_any_indicators(
                    metadata, pair, dataframe.copy(), tf, coin=pair.split("/")[0] + "-"
                )

        # the model will return 4 values, its prediction, an indication of whether or not the
        # prediction should be accepted, the target mean/std values from the labels used during
        # each training period.
        dataframe = self.model.bridge.start(dataframe, metadata, self)

        dataframe["target_roi"] = dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * 1.25
        dataframe["sell_roi"] = dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * 1.25
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [df["do_predict"] == 1, df["&-s_close"] > df["target_roi"]]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        # enter_short_conditions = [df["do_predict"] == 1, df["&-s_close"] < df["sell_roi"]]

        # if enter_short_conditions:
        #     df.loc[
        #         reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        #     ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-s_close"] < df["sell_roi"] * 0.25]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        # exit_short_conditions = [df["do_predict"] == 1, df["&-s_close"] > df["target_roi"] * 0.25]
        # if exit_short_conditions:
        #     df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
            rate: float, time_in_force: str, exit_reason: str, current_time, **kwargs,) -> bool:

        entry_tag = trade.enter_tag
        follow_mode = self.config.get("freqai", {}).get("follow_mode", False)
        if not follow_mode:
            pair_dict = self.model.bridge.dd.pair_dict
        else:
            pair_dict = self.model.bridge.dd.follower_dict

        with self.model.bridge.lock:
            pair_dict[pair]["prediction" + entry_tag] = 0
            if not follow_mode:
                self.model.bridge.dd.save_drawer_to_disk()
            else:
                self.model.bridge.dd.save_follower_dict_to_disk()

        return True

def tv_wma(df, length = 9) -> DataFrame:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + df.shift(i) * weight

    tv_wma = (sum / norm) if norm > 0 else 0
    return tv_wma

def tv_hma(dataframe, length = 9, field = 'close') -> DataFrame:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : HMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    h = 2 * tv_wma(dataframe[field], math.floor(length / 2)) - tv_wma(dataframe[field], length)

    tv_hma = tv_wma(h, math.floor(math.sqrt(length)))
    # dataframe.drop("h", inplace=True, axis=1)

    return tv_hma