from abc import ABC, abstractmethod
from backtester import Backtester
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from historical_data_processor import DeribitDataProcessor as Deribit



class Strategy(ABC):

    @abstractmethod
    def generate_signal():
        raise NotImplementedError(
            'should be implementing generate_signal() from subclass')
    

class MAStrategy(Strategy, Backtester): 
    '''
    sample strategy, don't use it
    '''
    def __init__(self, Exchange, start_year, start_month,
                 start_day, holding_period, up_multiplier, down_multiplier, 
                 lookback_period1, lookback_period2):
        Backtester.__init__(self, Exchange, start_year, start_month,
                            start_day, holding_period, up_multiplier, down_multiplier)
        self.lookback_period1 = lookback_period1
        self.lookback_period2 = lookback_period2

    def generate_signal(self):
        df = self.df
        df['short_period_ma'] = df.close.rolling(self.lookback_period1).mean()
        df['long_period_ma'] = df.close.rolling(self.lookback_period2).mean()
        df['ma_deficit'] = df['short_period_ma'] - df['long_period_ma']
        
        
        df['long'] = ((df.ma_deficit > 50) & (df.ma_deficit.shift(1) < 0)) * 1
        df['short'] = ((df.ma_deficit < -50) & (df.ma_deficit.shift(1) > 0)) * -1

        df['entry'] = df['long'] + df['short']

        df['number_of_long_signals'] = df.entry.gt(0).cumsum()
        df['number_of_short_signals'] = df.entry.lt(0).cumsum()
        df['total_number_of_signals'] = df.entry.ne(0).cumsum()

        self.df = df
            

class HigherHighLowerLow(Strategy, Backtester):
    '''
    simple high low strategy for testing
    '''
    def __init__(self, Exchange, start_year, start_month,
                 start_day, holding_period, up_multiplier, down_multiplier, time_interval='60'):
        Backtester.__init__(self, Exchange, start_year, start_month,
                            start_day, holding_period, up_multiplier, down_multiplier, time_interval)

    def generate_signal(self):
        df = self.df

        df['long'] = ((df.close.shift(2) > df.high.shift(3)) & (
            df.high > df.high.shift(1)) & (df.high.shift(1) > df.high.shift(2))) * 1
        
        df['short'] = ((df.close.shift(2) > df.low.shift(3)) & (
            df.low > df.low.shift(1)) & (df.low.shift(1) > df.low.shift(2))) * -1
        df['entry'] = df['short'] + df['long']
        
        df['number_of_long_signals'] = df.entry.gt(0).cumsum()
        df['number_of_short_signals'] = df.entry.lt(0).cumsum()
        df['total_number_of_signals'] = df.entry.ne(0).cumsum()

        df.dropna(inplace=True)

        self.df = df


class RSI(Strategy, Backtester):
    '''
    simple momentum RSI strategy for testing
    '''
    def __init__(self, Exchange, start_year, start_month,
                 start_day, holding_period, up_multiplier, down_multiplier, time_interval='60'):
        Backtester.__init__(self, Exchange, start_year, start_month,
                            start_day, holding_period, up_multiplier, down_multiplier, time_interval)

    # def generate_signal(self):
        

if __name__ == '__main__': 
    ma = MAStrategy(Deribit, '2021', '01', '26', holding_period=24, up_multiplier=1.05,
                    down_multiplier=0.95, lookback_period1=7, lookback_period2=10)
    ma.run_backtester()
    # hh = HigherHighLowerLow(Deribit, '2021', '01', '26', holding_period=20,
    #                         up_multiplier=1.01, down_multiplier=0.99, time_interval='30')
    # hh.run_backtester()
    # hh.plot_performance()
    # breakpoint()
    # hh.df['returns'] = hh.returns
    # hh.df.to_csv('high_low_strategy')
