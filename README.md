# crypto trading algos

Crypto trading analytic tools

[Data Processor API](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/utility_classes/historical_data_processor.py)
* Now support Deribit and FTX
 * Checkout this [Notebook](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/FTX_spreads.ipynb) to see how to use FTX Dataprocessor API to fetch historical spread data

[Backtester](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/utility_classes/backtester.py)
* Event driven backtesting class that can be incorporated with any strategies

[Sample Strategies](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/utility_classes/strategy.py)
* [Backtest Runner](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/backtest_runner.ipynb)
  * Backtest results for some of the directional trading sample strategies

[Statistical Arbitrage Research](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/statistical_arb.ipynb)
 * Pearson coefficient heatmap
   * ![0925 spread pearson](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/images/0925_pearson.PNG)
