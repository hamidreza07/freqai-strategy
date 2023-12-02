import logging
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
import numpy as np
from freqtrade.enums import (ExitCheckTuple, ExitType, RPCMessageType, RunMode, SignalDirection,
                             State, TradingMode)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import technical.indicators as ftt
import pandas_ta as pta

logger = logging.getLogger(__name__)

# inserire la differenza del valore t0-t1, t1-t2, t2-t3...

# l'hyperopt deve essere eseguito SOLO su periodo di movimento laterale e TESTATO su periodi di medio bull e di medio bear
# (SPIEGAZIONE: se faccio hyperopt su lungo periodo di tempo, potrebbe cercare di ottimizzare determinate condizioni estreme come pump e dump per estremizzare i profit su singoli trade: 
# la questione è che si potrebbe concentrare su eventi specifici e rari che vanno a compensare tutto il resto)

# inserire questi indicatori: https://github.com/just-nilux/legendary_ta/tree/main

# modificherei il target per cercare di arrivare all'innesco del trailing positivo e diminuirei lo shift -50 che sono troppe candele nel futuro!!! -> vediamo se da più importanza ai timeframe più piccoli invece dei 15m diminuendo il -50
# modificare il target per gestire long e short 

class Proton(IStrategy):
    """
    


    """
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v0.0.001"

    timeframe = '3m'

    max_trade_duration = 0

    # Futures
    custom_leverage = 1.0

    use_half_label_period_candle = True
    dynamic_target = False

    # target della differenza di prezzo per la predizione
    prediction_diff_perc_1 = 0.001
    prediction_diff_perc_2 = prediction_diff_perc_1 * 2

    num_label_1 = 5
    num_label_2 = 5

    #stoploss = -0.02 * custom_leverage
    stoploss = -0.2 * custom_leverage

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 30
    can_short = True
    
    minimal_roi = {
        "0": 1+prediction_diff_perc_2 * custom_leverage - 0.0011
    }

    plot_config = {
        'main_plot': {
        },
        'subplots': {
            "variazione percentuale": {
                'perc-price-variation-abs':  {'color': 'green'},
                'perc-price-variation-soglia-1': {'color': 'red'},
                'perc-price-variation-soglia-2': {'color': 'red'},
            },
            "Up_or_down": {
                '&s-up_or_down': {'color': 'green'},
            },
            "predict": {
                'do_predict': {'color': 'black'}
            },
            "prediction": {
                'prediction': {'color': 'blue'}
            },
            "prediction-stat": {
                'prediction_error-pct': {'color': 'red'},
                'prediction_mov_done_error-pct': {'color': 'blue'},
                'prediction_mov_correctness_error-pct': {'color': 'green'},
            }

        }
    }

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        self.custom_leverage = self.config["proton_parameters"]["custom_leverage"]

        self.dynamic_target = self.config["proton_parameters"]["dynamic_target"]

        self.prediction_diff_perc_1 = self.config["proton_parameters"]["minimal_price_perc_difference_1"]
        self.prediction_diff_perc_2 = self.config["proton_parameters"]["minimal_price_perc_difference_2"]

        self.use_half_label_period_candle = self.config["proton_parameters"]["use_half_label_period_candle"]

        self.num_label_2 = self.freqai_info["feature_parameters"]["label_period_candles"]
        if self.use_half_label_period_candle:
            self.num_label_1 = int(self.num_label_2 / 2)
        else:
            self.num_label_1 = self.num_label_2

        timeframe_in_minutes = int(self.timeframe[:-1])

        # Inizializzazione di parametri dipendenti da configurazione
        self.max_trade_duration = timeframe_in_minutes*(self.num_label_1)
        logger.info(f"--> self.max_trade_duration at {self.max_trade_duration} minutes")

        
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:
        """
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.
        
        Access metadata such as the current pair/timeframe/period with:

        `metadata["pair"]` `metadata["tf"]`  `metadata["period"]`

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        
                dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        
        """
        
        # https://www.avatrade.it/education/technical-analysis-indicators-strategies/mfi-indicator-trading-strategies
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        # MFI è un rsi che integra il volume e va accoppiato allo stocastico

        #
        ## PROVARE AD AGGIUNGERE: valutare l'importanza della feature:
        #
        #dataframe["%-relative_volume-period"] = (
        #    dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        #)


        return dataframe

    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.
        
        Access metadata such as the current pair/timeframe with:

        `metadata["pair"]` `metadata["tf"]`

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        """
        
        # https://www.avatrade.it/education/technical-analysis-indicators-strategies/stochastic-trading-strategies
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["%-fastd"] = stoch_fast['fastd']
        dataframe["%-fastk"] = stoch_fast['fastk']

        # Percentage changes
        dataframe["%-pct-change-1"] = dataframe["close"].pct_change(periods=self.num_label_1)
        dataframe["%-pct-volume-change-1"] = dataframe["volume"].pct_change(periods=self.num_label_1)

        if self.use_half_label_period_candle:
            dataframe["%-pct-change-2"] = dataframe["close"].pct_change(periods=self.num_label_2)
            dataframe["%-pct-volume-change-2"] = dataframe["volume"].pct_change(periods=self.num_label_2)

        # Percentage changes on mean of 3 last values
        #dataframe["%-pct-mean-change"] = dataframe["close"].shift(3).rolling(3).mean().pct_change()
        #dataframe["%-pct-mean-volume-change"] = dataframe["volume"].shift(3).rolling(3).mean().pct_change()
        
        # tratto da come utilizzare MFI https://www.comefaretradingonline.com/money-flow-index/#Come_Fare_Trading_con_il_Money_Flow_Index
        #if metadata["tf"] != self.timeframe:
        dataframe["distance-close-sma"] = dataframe["close"] - ta.SMA(dataframe, timeperiod=10)
        dataframe["%-pct-distance-close-sma"] = dataframe["distance-close-sma"].pct_change(periods=self.num_label_1)

        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).
        
        Access metadata such as the current pair with:

        `metadata["pair"]`

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        """

        ### Indicatori TradingView https://www.tradingview.com/symbols/BTCUSDT/technicals/?exchange=BINANCE
        ## 1 = buy, 0 = neutral, -1 = sell

        ## Moving Averages
        # Buy — MA value < price
        # Sell — MA value > price
        # Neutral — MA value = price

        # ema(10)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe.loc[(dataframe['ema10'] < dataframe['close']), '%-ema10-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-ema10-up-down']), '%-ema10-up-down'] = -1

        # sma(10)
        dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe.loc[(dataframe['sma10'] < dataframe['close']), '%-sma10-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-sma10-up-down']), '%-sma10-up-down'] = -1

        # ema(20)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe.loc[(dataframe['ema20'] < dataframe['close']), '%-ema20-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-ema20-up-down']), '%-ema20-up-down'] = -1

        # sma(20)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe.loc[(dataframe['sma20'] < dataframe['close']), '%-sma20-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-sma20-up-down']), '%-sma20-up-down'] = -1

        # ema(30)
        dataframe['ema30'] = ta.EMA(dataframe, timeperiod=30)
        dataframe.loc[(dataframe['ema30'] < dataframe['close']), '%-ema30-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-ema30-up-down']), '%-ema30-up-down'] = -1

        # sma(30)
        dataframe['sma30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe.loc[(dataframe['sma30'] < dataframe['close']), '%-sma30-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-sma30-up-down']), '%-sma30-up-down'] = -1

        # ema(50)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe.loc[(dataframe['ema50'] < dataframe['close']), '%-ema50-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-ema50-up-down']), '%-ema50-up-down'] = -1

        # sma(50)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe.loc[(dataframe['sma50'] < dataframe['close']), '%-sma50-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-sma50-up-down']), '%-sma50-up-down'] = -1

        # ema(100)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe.loc[(dataframe['ema100'] < dataframe['close']), '%-ema100-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-ema100-up-down']), '%-ema100-up-down'] = -1

        # sma(100)
        dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe.loc[(dataframe['sma100'] < dataframe['close']), '%-sma100-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-sma100-up-down']), '%-sma100-up-down'] = -1

        # ema(200)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe.loc[(dataframe['ema200'] < dataframe['close']), '%-ema200-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-ema200-up-down']), '%-ema200-up-down'] = -1

        # sma(200)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe.loc[(dataframe['sma200'] < dataframe['close']), '%-sma200-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-sma200-up-down']), '%-sma200-up-down'] = -1

        # ichimoku base line (9,26,52,26)
        # Ichimoku Cloud
        # Buy — base line < price and conversion line crosses price from below and lead line 1 > price and lead line 1 > lead line 2
        # Sell — base line > price and conversion line crosses price from above and lead line 1 < price and lead line 1 < lead line 2
        # Neutral — neither Buy nor Sell

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=9, base_line_periods=26, laggin_span=52, displacement=26)
        #dataframe['chikou_span'] = ichimoku['chikou_span'] NON UTILIZZARE: ha bias

        # conversion line
        #dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        # base line
        #dataframe['kijun_sen'] = ichimoku['kijun_sen']
        #
        #dataframe['senkou_a'] = ichimoku['senkou_span_a']
        #dataframe['senkou_b'] = ichimoku['senkou_span_b']
        # lead line 1
        #dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        # lead line 2
        #dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        #
        #dataframe['cloud_green'] = ichimoku['cloud_green']
        #dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe.loc[
            ((ichimoku['kijun_sen'] < dataframe['close']) &
             (qtpylib.crossed_above(ichimoku['tenkan_sen'], dataframe['close'])) &
             (ichimoku['leading_senkou_span_a'] > ichimoku['leading_senkou_span_b']))
            , '%-ichimoku-up-down'] = 1

        dataframe.loc[
            ((ichimoku['kijun_sen'] > dataframe['close']) &
             (qtpylib.crossed_below(ichimoku['tenkan_sen'], dataframe['close'])) &
             (ichimoku['leading_senkou_span_a'] < ichimoku['leading_senkou_span_b']))
            , '%-ichimoku-up-down'] = -1

        dataframe.loc[np.isnan(dataframe['%-ichimoku-up-down']), '%-ichimoku-up-down'] = 0

        # vwma(20)
        dataframe['vwma20'] = ftt.vwma(dataframe, 20)
        dataframe.loc[(dataframe['vwma20'] < dataframe['close']), '%-vwma20-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-vwma20-up-down']), '%-vwma20-up-down'] = -1

        # hma(9) Hull Moving Average
        dataframe['hma9'] = ftt.hull_moving_average(dataframe, 9)
        dataframe.loc[(dataframe['hma9'] < dataframe['close']), '%-hma9-up-down'] = 1
        dataframe.loc[np.isnan(dataframe['%-hma9-up-down']), '%-hma9-up-down'] = -1

        ## Oscillators

        # rsi(14)
        # Buy — indicator < 30 and rising
        # Sell — indicator > 70 and falling
        # Neutral — neither Buy nor Sell
        dataframe['rsi14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe.loc[((dataframe['rsi14'] < 30) & (dataframe['rsi14'] > dataframe['rsi14'].shift(1))), '%-rsi14-up-down'] = 1
        dataframe.loc[((dataframe['rsi14'] > 70) & (dataframe['rsi14'] < dataframe['rsi14'].shift(1))), '%-rsi14-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-rsi14-up-down']), '%-rsi14-up-down'] = 0

        # stoch(14,3,3)
        # Buy — main line < 20 and main line crosses over the signal line
        # Sell — main line > 80 and main line crosses under the signal line
        # Neutral — neither Buy nor Sell

        # nota: lo STOCH slow non funziona, uso il fast
        #stoch = ta.STOCHF(dataframe)
        #dataframe.loc[((stoch['fastk'] < 20) & qtpylib.crossed_above(stoch['fastk'], stoch['fastd'])), '%-stoch-fast-14-3-3-up-down'] = 1
        #dataframe.loc[((stoch['fastk'] > 80) & qtpylib.crossed_below(stoch['fastk'], stoch['fastd'])), '%-stoch-fast-14-3-3-up-down'] = -1
        #dataframe.loc[np.isnan(dataframe['%-stoch-fast-14-3-3-up-down']), '%-stoch-fast-14-3-3-up-down'] = 0

        stoch = ta.STOCH(dataframe)
        dataframe.loc[((stoch['slowk'] < 20) & qtpylib.crossed_above(stoch['slowk'], stoch['slowd'])), '%-stoch-14-3-3-up-down'] = 1
        dataframe.loc[((stoch['slowk'] > 80) & qtpylib.crossed_below(stoch['slowk'], stoch['slowd'])), '%-stoch-14-3-3-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-stoch-14-3-3-up-down']), '%-stoch-14-3-3-up-down'] = 0

        # CCI(20)
        # Buy — indicator < -100 and rising
        # Sell — indicator > 100 and falling
        # Neutral — neither Buy nor Sell
        dataframe['cci20'] = ta.CCI(dataframe, 20)
        dataframe.loc[((dataframe['cci20'] < -100) & (dataframe['cci20'] > dataframe['cci20'].shift(1))), '%-cci20-up-down'] = 1
        dataframe.loc[((dataframe['cci20'] > 100) & (dataframe['cci20'] < dataframe['cci20'].shift(1))), '%-cci20-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-cci20-up-down']), '%-cci20-up-down'] = 0

        # Average Directional Index (14)
        # Buy — indicator > 20 and +DI line crosses over -DI line
        # Sell — indicator > 20 and +DI line crosses under -DI line
        # Neutral — neither Buy nor Sell
        dataframe['adx'] = ta.ADX(dataframe, 14)
        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe.loc[((dataframe['adx'] > 20) & (qtpylib.crossed_above(dataframe['plus_di'], dataframe['minus_di']))), '%-adx14-up-down'] = 1
        dataframe.loc[((dataframe['adx'] > 20) & (qtpylib.crossed_below(dataframe['plus_di'], dataframe['minus_di']))), '%-adx14-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-adx14-up-down']), '%-adx14-up-down'] = 0

        # Awesome Oscillator ( https://www.avatrade.it/education/technical-analysis-indicators-strategies/awesome-oscillator-indicator-strategies )
        # Buy — saucer and values are greater than 0, or cross over the zero line
        # Sell — saucer and values are lower than 0, or cross under the zero line
        # Neutral — neither Buy nor Sell
        #TODO analizzare perchè awesome oscillator va in errore se faccio "plot"
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        # dataframe.loc[(
        #         #saucer
        #         (
        #         # Due barre rosse consecutive dell’istogramma, la seconda più corta della prima
        #         (dataframe['ao'].shift(3) > dataframe['ao'].shift(2)) #prima barra rossa
        #         & (dataframe['ao'].shift(2) > dataframe['ao'].shift(1)) #seconda barra rossa
        #         # La terza barra dell’istogramma è verde e più alta della seconda
        #         & (dataframe['ao'] > dataframe['ao'].shift(1))
        #         # Awesome Oscillator sopra lo zero
        #          & (dataframe['ao'] > 0)
        #         )
        #         | ((dataframe['ao'] > 0) & (dataframe['ao'].shift(1) < 0))
        # ), '%-ao-up-down'] = 1
        # dataframe.loc[(
        #     #saucer
        #         (
        #             # due barre verdi consecutive, di cui la seconda maggiore della prima
        #                 (dataframe['ao'].shift(3) < dataframe['ao'].shift(2)) #prima barra verde
        #                 & (dataframe['ao'].shift(2) < dataframe['ao'].shift(1)) #seconda barra verde
        #                 # La terza barra dell’istogramma è rossa e inferiore alla seconda
        #                 & (dataframe['ao'] < dataframe['ao'].shift(1))
        #                 # Awesome Oscillator sopra lo zero
        #                 & (dataframe['ao'] < 0)
        #         )
        #         | ((dataframe['ao'] < 0) & (dataframe['ao'].shift(1) > 0))
        # ), '%-ao-up-down'] = -1
        # dataframe.loc[np.isnan(dataframe['%-ao-up-down']), '%-ao-up-down'] = 0

        # Momentum (10)
        # Buy — indicator values are rising
        # Sell — indicator values are falling
        # Neutral — neither Buy nor Sell
        dataframe['mom'] = ta.MOM(dataframe, 10)

        dataframe.loc[(dataframe['mom'] > dataframe['mom'].shift(1)), '%-mom10-up-down'] = 1
        dataframe.loc[(dataframe['mom'] < dataframe['mom'].shift(1)), '%-mom10-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-mom10-up-down']), '%-mom10-up-down'] = 0

        # MACD Level (12, 26)
        # Buy — main line values > signal line values
        # Sell — main line values < signal line values
        # Neutral — neither Buy nor Sell
        macd = ta.MACD(dataframe, 12, 6)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']
        dataframe.loc[(dataframe['macd'] > dataframe['macdsignal']), '%-macd-12-26-up-down'] = 1
        dataframe.loc[(dataframe['macd'] < dataframe['macdsignal']), '%-macd-12-26-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-macd-12-26-up-down']), '%-macd-12-26-up-down'] = 0

        # Stochastic RSI Fast (3, 3, 14, 14)
        # Buy — downtrend and K and D lines < 20 and K line crosses over D line
        # Sell — uptrend and K and D lines > 80 and K line crosses under D line
        # Neutral — neither Buy nor Sell
        #TODO inserire i parametri corretti: timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=14)
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']
        dataframe.loc[(
            # TODO downtrend stochrsi (il segnale o il prezzo???)
            # K and D lines < 20
            (dataframe['fastk_rsi'] < 20)
            & (dataframe['fastd_rsi'] < 20)
            & qtpylib.crossed_above(dataframe['fastk_rsi'], dataframe['fastd_rsi'])
        ), '%-stochrsi-up-down'] = 1
        dataframe.loc[(
            # TODO uptrend stochrsi (il segnale o il prezzo???)
            # K and D lines > 80
            (dataframe['fastk_rsi'] > 80)
            & (dataframe['fastd_rsi'] > 80)
            & qtpylib.crossed_below(dataframe['fastk_rsi'], dataframe['fastd_rsi'])
        ), '%-stochrsi-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-stochrsi-up-down']), '%-stochrsi-up-down'] = 0

        # Williams Percent Range (14) (valori tra -100 e 0)
        # Buy — indicator < lower band and rising
        # Sell — indicator > upper band and falling
        # Neutral — neither Buy nor Sell
        # ipotizzo che band sia tra -20 e -80 https://www.guidatradingonline.net/indicatore-williams-percent-range/
        dataframe['willr'] = ta.WILLR(dataframe, 14)
        dataframe.loc[(
            (dataframe['willr'] < -80)
            & (dataframe['willr'] > dataframe['willr'].shift(-1))
        ), '%-willr-up-down'] = 1
        dataframe.loc[(
            (dataframe['willr'] > -20)
            & (dataframe['willr'] < dataframe['willr'].shift(-1))
        ), '%-willr-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-willr-up-down']), '%-willr-up-down'] = 0

        #TODO Bull Bear Power
        # Buy — uptrend and BearPower < zero and BearPower is rising
        # Sell — downtrend and BullPower > zero and BullPower is falling
        # Neutral — neither Buy nor Sell

        # https://skilling.com/row/en/blog/trading-articles/what-is-the-bulls-and-bears-power-indicator/

        # Bull Power = High price - Exponential Moving Average.
        #   It simply measures whether the current price is higher than the previous one and if it is, then the bulls are considered to be ‘winning’.
        #   This is done by taking the difference between the highest price of the current bar, and an exponential moving average (13-period by default)

        # Bear Power = Low price - Exponential Moving Average
        #    you take the difference between the lowest price and the exponential moving average (13-period by default)

        # uptrend e downtrend con SMA : https://www.babypips.com/learn/forex/using-moving-averages :
        #   puoi usare 2 o più sma per esempio sma(10) e sma(20)
        #   faster MA over slower MA in an uptrend, slower MA over faster MA in a downtrend




        # Ultimate Oscillator (7, 14, 28)
        # Buy — UO > 70
        # Sell — UO < 30
        # Neutral — neither Buy nor Sell
        dataframe['uo'] = ta.ULTOSC(dataframe)
        dataframe.loc[(dataframe['uo'] > 70), '%-uo-up-down'] = 1
        dataframe.loc[(dataframe['uo'] < 30), '%-uo-up-down'] = -1
        dataframe.loc[np.isnan(dataframe['%-uo-up-down']), '%-uo-up-down'] = 0


        # Indicatore TradingView
        # [-1.0 ≥ value < -0.5] — Strong Sell
        # [-0.5 ≥ value < -0.1] — Sell
        # [-0.1 ≥ value ≤ 0.1] — Neutral
        # [0.1 > value ≤ 0.5] — Buy
        # [0.5 > value ≤ 1.0] — Strong Buy
        # The overall rating of the indicator includes two large groups of indicators.
        # The first consists of SMAs and EMAs with different lengths (MA lengths are 10, 20, 30, 50, 100 and 200),
        # the Ichimoku Cloud (9, 26, 52), VWMA (20) and HullMA (9).
        # The second one is calculated on the following oscillators: RSI (14), Stochastic (14, 3, 3), CCI (20),
        # ADX (14, 14), AO, Momentum (10), MACD (12, 26, 9), Stochastic RSI (3, 3, 14, 14), Williams %R (14),
        # Bulls and Bears Power and UO (7,14,28).
        # Each group's rating is calculated separately: oscillators e MovingAverages

        dataframe['tw-movingaverages-avg'] = dataframe[[
            '%-ema10-up-down', '%-sma10-up-down',
            '%-ema20-up-down', '%-sma20-up-down',
            '%-ema30-up-down', '%-sma30-up-down',
            '%-ema50-up-down', '%-sma50-up-down',
            '%-ema100-up-down', '%-sma100-up-down',
            '%-ema200-up-down', '%-sma200-up-down',
            '%-ichimoku-up-down', '%-vwma20-up-down',
            '%-hma9-up-down'
        ]].mean(axis=1)

        dataframe.loc[(-1 <= dataframe['tw-movingaverages-avg']) & (dataframe['tw-movingaverages-avg'] < -0.5), '%-tw-movingaverages-up-down'] = -2
        dataframe.loc[(-0.5 <= dataframe['tw-movingaverages-avg']) & (dataframe['tw-movingaverages-avg'] < -0.1), '%-tw-movingaverages-up-down'] = -1
        dataframe.loc[(-0.1 <= dataframe['tw-movingaverages-avg']) & (dataframe['tw-movingaverages-avg'] < 0.1), '%-tw-movingaverages-up-down'] = 0
        dataframe.loc[(0.1 <= dataframe['tw-movingaverages-avg']) & (dataframe['tw-movingaverages-avg'] < 0.5), '%-tw-movingaverages-up-down'] = 1
        dataframe.loc[(0.5 <= dataframe['tw-movingaverages-avg']) & (dataframe['tw-movingaverages-avg'] < 1), '%-tw-movingaverages-up-down'] = 2

        dataframe['tw-oscillators-avg'] = dataframe[[
            '%-rsi14-up-down', '%-stoch-14-3-3-up-down',
            '%-cci20-up-down', '%-adx14-up-down',
            #'%-ao-up-down',
            '%-mom10-up-down', '%-macd-12-26-up-down',
            '%-stochrsi-up-down', '%-willr-up-down',
            #bull e bear power
            '%-uo-up-down'
        ]].mean(axis=1)

        dataframe.loc[(-1 <= dataframe['tw-oscillators-avg']) & (dataframe['tw-oscillators-avg'] < -0.5), '%-tw-oscillators-up-down'] = -2
        dataframe.loc[(-0.5 <= dataframe['tw-oscillators-avg']) & (dataframe['tw-oscillators-avg'] < -0.1), '%-tw-oscillators-up-down'] = -1
        dataframe.loc[(-0.1 <= dataframe['tw-oscillators-avg']) & (dataframe['tw-oscillators-avg'] < 0.1), '%-tw-oscillators-up-down'] = 0
        dataframe.loc[(0.1 <= dataframe['tw-oscillators-avg']) & (dataframe['tw-oscillators-avg'] < 0.5), '%-tw-oscillators-up-down'] = 1
        dataframe.loc[(0.5 <= dataframe['tw-oscillators-avg']) & (dataframe['tw-oscillators-avg'] < 1), '%-tw-oscillators-up-down'] = 2

        dataframe['tradingview-avg'] = dataframe[[
            '%-ema10-up-down', '%-sma10-up-down',
            '%-ema20-up-down', '%-sma20-up-down',
            '%-ema30-up-down', '%-sma30-up-down',
            '%-ema50-up-down', '%-sma50-up-down',
            '%-ema100-up-down', '%-sma100-up-down',
            '%-ema200-up-down', '%-sma200-up-down',
            '%-ichimoku-up-down', '%-vwma20-up-down',
            '%-hma9-up-down',

            '%-rsi14-up-down', '%-stoch-14-3-3-up-down',
            '%-cci20-up-down', '%-adx14-up-down',
            #'%-ao-up-down',
            '%-mom10-up-down', '%-macd-12-26-up-down',
            '%-stochrsi-up-down', '%-willr-up-down',
            #bull e bear power
            '%-uo-up-down'
        ]].mean(axis=1)

        dataframe.loc[(-1 <= dataframe['tradingview-avg']) & (dataframe['tradingview-avg'] < -0.5), '%-tradingview-up-down'] = -2
        dataframe.loc[(-0.5 <= dataframe['tradingview-avg']) & (dataframe['tradingview-avg'] < -0.1), '%-tradingview-up-down'] = -1
        dataframe.loc[(-0.1 <= dataframe['tradingview-avg']) & (dataframe['tradingview-avg'] < 0.1), '%-tradingview-up-down'] = 0
        dataframe.loc[(0.1 <= dataframe['tradingview-avg']) & (dataframe['tradingview-avg'] < 0.5), '%-tradingview-up-down'] = 1
        dataframe.loc[(0.5 <= dataframe['tradingview-avg']) & (dataframe['tradingview-avg'] < 1), '%-tradingview-up-down'] = 2

        return dataframe


    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        self.freqai.class_names = ['-2', '-1', '0', '1', '2']

        if not self.dynamic_target:  # target dinamico disabilitato

            dataframe['&s-up_or_down'] = np.where(
                (
                    (dataframe["close"].shift(-self.num_label_1) > (dataframe["close"]*(1+self.prediction_diff_perc_1)))
                    & (dataframe['high'].shift(-self.num_label_2).rolling(self.num_label_2).max() >= (dataframe["close"]*(1+self.prediction_diff_perc_1)))
                ),
                '1', '0')

            dataframe['&s-up_or_down'] = np.where(
                (
                        (dataframe["close"].shift(-self.num_label_1) > (dataframe["close"]*(1+self.prediction_diff_perc_2)))
                        & (dataframe['high'].shift(-self.num_label_2).rolling(self.num_label_2).max() >= (dataframe["close"]*(1+self.prediction_diff_perc_2)))
                ),
                '2', dataframe['&s-up_or_down'])

            dataframe['&s-up_or_down'] = np.where(
                (
                        (dataframe["close"].shift(-self.num_label_1) < (dataframe["close"]*(1-self.prediction_diff_perc_1)))
                        & (dataframe['low'].shift(-self.num_label_2).rolling(self.num_label_2).min() <= (dataframe["close"]*(1-self.prediction_diff_perc_1)))
                ),
                '-1', dataframe['&s-up_or_down'])

            dataframe['&s-up_or_down'] = np.where(
                (
                        (dataframe["close"].shift(-self.num_label_1) < (dataframe["close"]*(1-self.prediction_diff_perc_2)))
                        & (dataframe['low'].shift(-self.num_label_2).rolling(self.num_label_2).min() <= (dataframe["close"]*(1-self.prediction_diff_perc_2)))
                ),
                '-2', dataframe['&s-up_or_down'])

        else:
            ## INIZIO Tentativo di target dinamico (non fa alcuna predizione
            dataframe['pct-change'] = dataframe["close"].pct_change(periods=self.num_label_1)
            dataframe.loc[np.isnan(dataframe['pct-change']), 'pct-change'] = 0

            dataframe['target_1'] = ta.SMA(abs(dataframe['pct-change']), self.num_label_1)
            # imposto un  minimo
            dataframe.loc[dataframe['target_1'] < self.prediction_diff_perc_1, 'target_1'] = self.prediction_diff_perc_1
            dataframe['target_2'] = dataframe['target_1']*1.5
            dataframe.loc[np.isnan(dataframe['target_1']), 'target_1'] = 1
            dataframe.loc[np.isnan(dataframe['target_2']), 'target_2'] = 1

            dataframe['&s-up_or_down'] = np.where(
                #dataframe["%-pct-change-1_gen_BTC/USDT:USDT_3m"].shift(-self.num_label_1) > self.prediction_diff_perc_1,
                #dataframe["close"].shift(-self.num_label_1) > (dataframe["close"]*(1+self.prediction_diff_perc_1)),
                dataframe["pct-change"].shift(-self.num_label_1) > dataframe['target_1'],
                '1', '0')

            dataframe['&s-up_or_down'] = np.where(
                #dataframe["%-pct-change-1_gen_BTC/USDT:USDT_3m"].shift(-self.num_label_1) > self.prediction_diff_perc_2,
                #dataframe["close"].shift(-self.num_label_1) > (dataframe["close"]*(1+self.prediction_diff_perc_2)),
                dataframe["pct-change"].shift(-self.num_label_1) > dataframe['target_2'],
                '2', dataframe['&s-up_or_down'])

            dataframe['&s-up_or_down'] = np.where(
                #dataframe["%-pct-change-1_gen_BTC/USDT:USDT_3m"].shift(-self.num_label_1) < self.prediction_diff_perc_1,
                #dataframe["close"].shift(-self.num_label_1) < (dataframe["close"]*(1-self.prediction_diff_perc_1)),
                dataframe["pct-change"].shift(-self.num_label_1) < -dataframe['target_1'],
                '-1', dataframe['&s-up_or_down'])

            dataframe['&s-up_or_down'] = np.where(
                #dataframe["%-pct-change-1_gen_BTC/USDT:USDT_3m"].shift(-self.num_label_1) < self.prediction_diff_perc_2,
                #dataframe["close"].shift(-self.num_label_1) < (dataframe["close"]*(1-self.prediction_diff_perc_2)),
                dataframe["pct-change"].shift(-self.num_label_1) < -dataframe['target_2'],
                '-2', dataframe['&s-up_or_down'])

            ## FINE target dinamico

#        dataframe['&s-up_or_down'] = np.where(
#            dataframe["close"]
#            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
#            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
#            .mean() >  dataframe["close"], '1', '-1')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901

        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

        dataframe = self.freqai.start(dataframe, metadata, self)

        if self.dp.runmode == RunMode.BACKTEST:
            #######################################################
            # Verifica predizione
            #######################################################


            #( (valore finale - valore originale) / valore originale) x 100%

            #todo rivedere questo calcolo
            dataframe["perc-price-variation"] = dataframe["close"].pct_change(periods=self.num_label_1) * 100

            dataframe["perc-price-variation-abs"] = abs(dataframe["perc-price-variation"])
            dataframe['perc-price-variation-soglia-1'] = abs(self.prediction_diff_perc_1 * 100)
            dataframe['perc-price-variation-soglia-2'] = abs(self.prediction_diff_perc_2 * 100)

            dataframe.loc[
                (
                    (
                    (dataframe["perc-price-variation"] >= dataframe['perc-price-variation-soglia-1']) &
                    (dataframe['&s-up_or_down'].shift(self.num_label_1) == '1') &
                    (dataframe['do_predict'].shift(self.num_label_1) == 1)
                    ) |
                    (
                    (dataframe["perc-price-variation"] >= dataframe['perc-price-variation-soglia-1']) &
                    (dataframe['&s-up_or_down'].shift(self.num_label_1) == '2') &
                    (dataframe['do_predict'].shift(self.num_label_1) == 1)
                    ) |
                    (
                    (dataframe["perc-price-variation"] <= -dataframe['perc-price-variation-soglia-1']) &
                    (dataframe['&s-up_or_down'].shift(self.num_label_1) == '-1') &
                    (dataframe['do_predict'].shift(self.num_label_1) == 1)
                    ) |
                    (
                    (dataframe["perc-price-variation"] <= -dataframe['perc-price-variation-soglia-1']) &
                    (dataframe['&s-up_or_down'].shift(self.num_label_1) == '-2') &
                    (dataframe['do_predict'].shift(self.num_label_1) == 1)
                    ) |
                    (
                    (dataframe["perc-price-variation"] <= dataframe['perc-price-variation-soglia-1']) &
                    (dataframe["perc-price-variation"] >= -dataframe['perc-price-variation-soglia-1']) &
                    (dataframe['&s-up_or_down'].shift(self.num_label_1) == '0') &
                    (dataframe['do_predict'].shift(self.num_label_1) == 1)
                    ) |
                    (dataframe['do_predict'].shift(self.num_label_1) < 1)
                ) & dataframe["DI_values"] > 0,
                'prediction'
                          ] = 1

            dataframe.loc[np.isnan(dataframe["prediction"]), 'prediction'] = 0
            #dataframe.loc[
            #    (
            #        ((dataframe["perc-price-variation"] < dataframe['perc-price-variation-soglia-1']) & (dataframe['&s-up_or_down'].shift(self.num_label_1) == '1') & (dataframe['do_predict'].shift(self.num_label_1) == 1)) |
            #        ((dataframe["perc-price-variation"] < dataframe['perc-price-variation-soglia-2']) & (dataframe['&s-up_or_down'].shift(self.num_label_1) == '2') & (dataframe['do_predict'].shift(self.num_label_1) == 1)) |
            #        ((dataframe["perc-price-variation"] > -dataframe['perc-price-variation-soglia-1']) & (dataframe['&s-up_or_down'].shift(self.num_label_1) == '-1') & (dataframe['do_predict'].shift(self.num_label_1) == 1)) |
            #        ((dataframe["perc-price-variation"] > -dataframe['perc-price-variation-soglia-2']) & (dataframe['&s-up_or_down'].shift(self.num_label_1) == '-2') & (dataframe['do_predict'].shift(self.num_label_1) == 1)) |
            #        (((dataframe["perc-price-variation"] > dataframe['perc-price-variation-soglia-1']) | (dataframe["perc-price-variation"] < -dataframe['perc-price-variation-soglia-1'])) & (dataframe['&s-up_or_down'].shift(self.num_label_1) == '0') & (dataframe['do_predict'].shift(self.num_label_1) == 1))
            #    ),'prediction'] = 0

            dataframe.loc[
                (dataframe["DI_values"] > 0), 'prediction_index_calc'
            ] = 1
            dataframe['prediction_index'] = dataframe['prediction_index_calc'].cumsum() - dataframe['prediction_index_calc']
            dataframe['prediction_sum'] = dataframe['prediction'].cumsum() - dataframe['prediction']
            dataframe['prediction_error'] = dataframe['prediction_index'] - dataframe['prediction_sum']
            dataframe['prediction_error-pct'] = (dataframe['prediction_sum'] / dataframe['prediction_index']) * 100
            #######################################################
            # Only prediction movement
            #
            dataframe.loc[
                ((dataframe['prediction_index_calc'] == 1) &
                 (abs(dataframe["perc-price-variation"]) > abs(dataframe['perc-price-variation-soglia-1']))) & dataframe["DI_values"] > 0,
                'prediction_mov_index_calc'] = 1
            dataframe.loc[np.isnan(dataframe["prediction_mov_index_calc"]), 'prediction_mov_index_calc'] = 0
            dataframe['prediction_mov_index'] = dataframe['prediction_mov_index_calc'].cumsum() - dataframe['prediction_mov_index_calc']

            dataframe.loc[
                (
                        (
                                (dataframe["perc-price-variation"] >= dataframe['perc-price-variation-soglia-1']) &
                                (dataframe['&s-up_or_down'].shift(self.num_label_1) == '1') &
                                (dataframe['do_predict'].shift(self.num_label_1) == 1)
                        ) |
                        (
                                (dataframe["perc-price-variation"] >= dataframe['perc-price-variation-soglia-1']) &
                                (dataframe['&s-up_or_down'].shift(self.num_label_1) == '2') &
                                (dataframe['do_predict'].shift(self.num_label_1) == 1)
                        ) |
                        (
                                (dataframe["perc-price-variation"] <= -dataframe['perc-price-variation-soglia-1']) &
                                (dataframe['&s-up_or_down'].shift(self.num_label_1) == '-1') &
                                (dataframe['do_predict'].shift(self.num_label_1) == 1)
                        ) |
                        (
                                (dataframe["perc-price-variation"] <= -dataframe['perc-price-variation-soglia-1']) &
                                (dataframe['&s-up_or_down'].shift(self.num_label_1) == '-2') &
                                (dataframe['do_predict'].shift(self.num_label_1) == 1)
                        )
                ),
                'prediction_mov'
            ] = 1
            dataframe.loc[np.isnan(dataframe["prediction_mov"]), 'prediction_mov'] = 0
            dataframe['prediction_mov_sum'] = dataframe['prediction_mov'].cumsum() - dataframe['prediction_mov']
            # numero di errori di predizione e/o predizioni mancanti
            dataframe['prediction_mov_error'] = dataframe['prediction_mov_index'] - dataframe['prediction_mov_sum']
            # percentuale di correttezza di predizioni (quelle mancanti sono conteggiate errate)
            dataframe['prediction_mov_error-pct'] = (dataframe['prediction_mov_sum'] / dataframe['prediction_mov_index']) * 100

            ######################################################################
            ## aggiungere solo errori fatti su predizioni di movimento effettuate
            # tutte le righe con &s-up_or_down <> 0
            dataframe.loc[
                (
                 (dataframe['&s-up_or_down'].shift(self.num_label_1) != '0') &
                 (dataframe['do_predict'].shift(self.num_label_1) == 1) &
                 (dataframe["DI_values"].shift(self.num_label_1) > 0)
                ),
                'prediction_mov_correctness_index_calc'
            ] = 1
            dataframe.loc[np.isnan(dataframe["prediction_mov_correctness_index_calc"]), 'prediction_mov_correctness_index_calc'] = 0
            dataframe['prediction_mov_correctness_index'] = dataframe['prediction_mov_correctness_index_calc'].cumsum() - dataframe['prediction_mov_correctness_index_calc']

        # se prediction_mov == 1 allora e' corretta la predizione
            dataframe.loc[(dataframe['prediction_mov_correctness_index_calc'] == 1) & (dataframe["prediction_mov"] == 1),
                'prediction_mov_correctness'
            ] = 1
            dataframe.loc[np.isnan(dataframe["prediction_mov_correctness"]), 'prediction_mov_correctness'] = 0
            dataframe['prediction_mov_correctness_sum'] = dataframe['prediction_mov_correctness'].cumsum() - dataframe['prediction_mov_correctness']
            # numero di errori di predizione e/o predizioni mancanti
            dataframe['prediction_mov_correctness_error'] = dataframe['prediction_mov_correctness_index'] - dataframe['prediction_mov_correctness_sum']
            # percentuale di correttezza di predizioni (quelle mancanti sono conteggiate errate)
            dataframe['prediction_mov_correctness_error-pct'] = (dataframe['prediction_mov_correctness_sum'] / dataframe['prediction_mov_correctness_index']) * 100
            # percentuale di predizioni fatte sul numero di predizioni possibili
            dataframe['prediction_mov_done_error-pct'] = (dataframe['prediction_mov_correctness_index'] / dataframe['prediction_mov_index']) * 100

            # utilizzo iloc -2 perche' l'ultimo record non va tenuto in considerazione
            # logger.info
            print(f"--> Numero di candele totali: {dataframe['prediction_index'].iloc[-2]}")
            print(f"--> Percentuale di predizioni corrette:                  {dataframe['prediction_error-pct'].iloc[-2]} % -- numero posizioni possibili: {dataframe['prediction_index'].iloc[-2]}  -- numero posizioni corrette: {dataframe['prediction_sum'].iloc[-2]} ")
            # print(f"NB: tiene conto come errate solo quelle errate, quelle mancate sono conteggiate di default in questo caso")

            # print(f"--> Percentuale di predizioni corrette (solo movimenti): {dataframe['prediction_mov_error-pct'].iloc[-2]} % -- numero posizioni possibili: {dataframe['prediction_mov_index'].iloc[-2]}  -- numero posizioni corrette: {dataframe['prediction_mov_sum'].iloc[-2]} ")
            # print(f"NB: predizioni possibili verso quelle corrette, tiene conto come errate sia quelle mancate che quelle errate")

            print(f"--> Percentuale di predizioni fatte (solo movimenti):    {dataframe['prediction_mov_done_error-pct'].iloc[-2]} % -- numero predizioni possibili: {dataframe['prediction_mov_index'].iloc[-2]}  -- numero predizioni fatte: {dataframe['prediction_mov_correctness_index'].iloc[-2]} ")

            print(f"--> Percentuale di predizioni corrette (solo movimenti): {dataframe['prediction_mov_correctness_error-pct'].iloc[-2]} % -- numero predizioni fatte: {dataframe['prediction_mov_correctness_index'].iloc[-2]}  -- numero posizioni corrette: {dataframe['prediction_mov_correctness_sum'].iloc[-2]} ")
            # print(f"NB: predizioni fatte verso quelle corrette")

            print(f"**** migliorare la percentuale di predizioni fatte :: {dataframe['prediction_mov_done_error-pct'].iloc[-2]} % -- il numero di predizioni dei movimenti ( {dataframe['prediction_mov_correctness_index'].iloc[-2]} ) verso le possibilità di posizioni ( {dataframe['prediction_mov_index'].iloc[-2]} ) sul totale di {dataframe['prediction_index'].iloc[-2]} candele")
            print(f"**** migliorare la percentuale di correttezza delle predizioni dei movimenti :: {dataframe['prediction_mov_correctness_error-pct'].iloc[-2]} %  ")

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        min_confidence_level = 0.2  # 1 / numero di classi
        df.loc[
            (
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['1'] > min_confidence_level) &
                ((df['&s-up_or_down'] == '1'))
            ),
            ['enter_long', 'enter_tag']] = (1, 'long-1')

        df.loc[
            (
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['2'] > min_confidence_level) &
                ((df['&s-up_or_down'] == '2'))
            ),
            ['enter_long', 'enter_tag']] = (1, 'long-2')

        df.loc[
            (
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['-1'] > min_confidence_level) &
                ((df['&s-up_or_down'] == '-1'))
            ),
            ['enter_short', 'enter_tag']] = (1, 'short-1')

        df.loc[
            (
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['-2'] > min_confidence_level) &
                ((df['&s-up_or_down'] == '-2'))
            ),
            ['enter_short', 'enter_tag']] = (1, 'short-2')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        return df

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        #TODO aggiungere che se l'ultima candela è inversa alla direzione, allora deve uscire
        
        #TODO aggiungere che se nel dataframe ci sono segnali nella stessa direzione, allora non esco
        
        # trade expired
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if (((trade_duration >= self.max_trade_duration) &
                (self.use_half_label_period_candle) & (current_profit > 0)) |
                ((trade_duration >= self.max_trade_duration) & (not self.use_half_label_period_candle))):
            # logger.info(f"{pair} trade_expired at {current_profit*100}")
            return "trade_expired"

        if trade_duration >= (self.max_trade_duration*2):
            return "trade_expired_x2"

        # esco se supero la media del pct-change
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dataframe['custom_exit-pct-change'] = dataframe["close"].pct_change(periods=self.num_label_1)
        dataframe.loc[np.isnan(dataframe['custom_exit-pct-change']), 'custom_exit-pct-change'] = 0

        dataframe['custom_exit-target_1'] = ta.SMA(abs(dataframe['custom_exit-pct-change']), self.num_label_1)
        dataframe.loc[dataframe['custom_exit-target_1'] < self.prediction_diff_perc_1, 'custom_exit-target_1'] = self.prediction_diff_perc_1
        current_candle = dataframe.iloc[-1].squeeze()
        if current_profit > current_candle['custom_exit-target_1']*self.custom_leverage:
            return "early_profit"

        if current_profit < -(current_candle['custom_exit-target_1']*self.custom_leverage*3):
            return "stop_loss"

        if self.dp.runmode == RunMode.BACKTEST:

            if (trade.calc_profit_ratio(current_candle['high']) >= current_candle['custom_exit-target_1']) | (trade.calc_profit_ratio(current_candle['low']) >= current_candle['custom_exit-target_1']):
                return "early_profit_backtest"

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return self.custom_leverage

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        if exit_reason == 'force_exit' and trade.calc_profit_ratio(rate) < 0:
            # Reject force-sells with negative profit
            # This is just a sample, please adjust to your needs
            # (this does not necessarily make sense, assuming you know when you're force-selling)
            #return False
            return True
        return True

    ##TODO PER RISK MANAGEMENT
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # current_candle = dataframe.iloc[-1].squeeze()
        #
        # if current_candle['fastk_rsi_1h'] > current_candle['fastd_rsi_1h']:
        #     if self.config['stake_amount'] == 'unlimited':
        #         # Use entire available wallet during favorable conditions when in compounding mode.
        #         return max_stake
        #     else:
        #         # Compound profits during favorable conditions instead of using a static stake.
        #         return self.wallets.get_total_stake_amount() / self.config['max_open_trades']

        # Use default stake amount.
        return proposed_stake

    # UTILITY

#TODO implementare il parametro window
def is_rising(column_dataframe, window=3) -> bool:

    if column_dataframe > column_dataframe.shift(1) & column_dataframe.shift(1) > column_dataframe(2):

        return True

    return False


#TODO implementare il parametro window
def is_falling(self, column_dataframe, window=3) -> bool:

    if column_dataframe < column_dataframe.shift(1) & column_dataframe.shift(1) < column_dataframe(2):

        return True

    return False