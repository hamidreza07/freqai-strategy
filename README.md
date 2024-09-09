
# FreqAI Strategy

This repository contains strategies designed for the [FreqAI](https://github.com/freqtrade/freqai) module, an extension of the [FreqTrade](https://www.freqtrade.io/) cryptocurrency algorithmic trading framework. The strategies here leverage machine learning models to predict market movements and make informed buy/sell decisions.

## Features

- **Customizable Strategies**: Easily modify parameters to fine-tune performance.
- **Model Integration**: Integrates with a variety of machine learning models to make real-time predictions.
- **Backtesting**: Supports robust backtesting to evaluate the performance of strategies using historical data.
- **Hyperparameter Optimization**: Built-in support for optimizing model and strategy parameters using FreqTrade’s optimization tools.
- **High Compatibility**: Compatible with FreqTrade and FreqAI for seamless integration with existing strategies and tools.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/hamidreza07/freqai-strategy.git
    cd freqai-strategy
    ```

2. Install FreqTrade and FreqAI (if not already installed):
    ```bash
    # FreqTrade installation
    git clone https://github.com/freqtrade/freqtrade.git
    cd freqtrade
    ./setup.sh --install

    # FreqAI installation
    pip install freqai
    ```

3. Copy the strategy files from this repository into your FreqTrade `user_data/strategies` directory.

4. Edit the configuration file (config.json) to fit your exchange and trading pair needs.

## Usage

1. **Backtesting**: To backtest one of the strategies, use the following command:
    ```bash
    freqtrade backtesting --strategy YourStrategyName --timerange 20210101-20220101
    ```

2. **Hyperparameter Optimization**: To optimize the strategy, run:
    ```bash
    freqtrade hyperopt --hyperopt YourStrategyNameHyperopt --spaces buy sell roi trailing --timerange=20210101-20220101
    ```

3. **Running the bot**: After backtesting and optimization, you can run the bot with your chosen strategy:
    ```bash
    freqtrade trade --strategy YourStrategyName
    ```

## Strategy Details

The strategies in this repository aim to predict short-term market movements using machine learning models. By integrating with FreqAI, they can learn from historical data to adapt to changing market conditions and provide predictions based on trends, price action, and other market signals.

### Strategies Included:
- **ML-Based Strategy**: Utilizes a machine learning model to predict buy/sell signals based on a variety of market indicators.
- **Customizable Thresholds**: Adjust buy/sell thresholds to adapt to different market environments.
  
### Customization

Each strategy can be fine-tuned by adjusting parameters such as:
- Model type (e.g., Random Forest, Gradient Boosting)
- Buy/sell thresholds
- Indicators (RSI, MACD, EMA, etc.)
- Timeframes for analysis

## Contributing

Feel free to open issues or pull requests if you want to contribute to improving the strategies. All contributions are welcome!


## Contact

For any questions or suggestions, you can reach out to [Hamidreza Habibi](hamidreza07@gmail.com).

---

This README includes installation steps, usage instructions, and strategy details, giving potential users and contributors a clear understanding of the repository's purpose and how to get started. You can modify sections as needed depending on the specifics of your strategy implementations.
