# crypto trading algos

Crypto trading analytic tools

All the API classes used in research Notebooks are stored in the [utility_classes](https://github.com/dabaojian1992/crypto_trading_researches/tree/master/strategy_backtests/utility_classes) folder

[Data Processing API](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/utility_classes/historical_data_processor.py)
* Now support Deribit and FTX
 * Checkout this [Notebook](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/FTX_spreads_fetching.ipynb) to see how to use FTX Dataprocessor API to fetch historical spread data
* To get FTX historical perp, expired futures and spread OHLC data all at once, simply change the output file path and select 'run all cells' in [this](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/FTX_spreads_fetching.ipynb) Notebook

[Sample Strategies](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/utility_classes/strategy.py)
* [Backtest Runner](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/backtest_runner.ipynb)
  * Backtest results for some of the directional trading sample strategies

[Statistical Arbitrage Research](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/statistical_arb.ipynb)
 * This research finds the tokens on FTX that exibit cointegrated movements between perpetual and futures contracts 

[Garch model for spread volatility prediction](https://github.com/dabaojian1992/crypto_trading_researches/blob/master/strategy_backtests/garch_example.ipynb)
* This research implements rolling GARCH model to predict spread volatility 
