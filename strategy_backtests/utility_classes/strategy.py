from abc import ABC, abstractmethod
from backtester import Backtester
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from historical_data_processor import DeribitDataProcessor as Deribit
plt.style.use('ggplot')

class Strategy(ABC):

    @abstractmethod
    def generate_signal():
        raise NotImplementedError(
            'should be implementing generate_signal() from subclass')
    

class MAStrategy(Strategy, Backtester): 
    '''
    Sample MA strategy for testing
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
        
        
        df['long'] = ((df.ma_deficit > 30) & (df.ma_deficit.shift(1) < 0)) * 1
        df['short'] = ((df.ma_deficit < -30) & (df.ma_deficit.shift(1) > 0)) * -1

        df['entry'] = df['long'] + df['short']

        df['number_of_long_signals'] = df.entry.gt(0).cumsum()
        df['number_of_short_signals'] = df.entry.lt(0).cumsum()
        df['total_number_of_signals'] = df.entry.ne(0).cumsum()

        self.df = df
            

class HigherHighLowerLow(Strategy, Backtester):
    '''
    Sample high low strategy for testing
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


class MomentumRSI(Strategy, Backtester):
    '''
    Sample momentum RSI strategy for testing
    '''
    def __init__(self, Exchange, start_year, start_month,
                 start_day, holding_period, up_multiplier, down_multiplier, RSI_lookback_period, time_interval='60', RSI_short=30, RSI_long=80, MA_long_period=200, MA_short_period=50):
        Backtester.__init__(self, Exchange, start_year, start_month,
                            start_day, holding_period, up_multiplier, down_multiplier, time_interval)

        self.RSI_lookback_period = RSI_lookback_period
        self.RSI_long = RSI_long
        self.RSI_short = RSI_short

        self.MA_short_period = MA_short_period
        self.MA_long_period = MA_long_period
    
    def calc_RSI(self):
        df = self.df
        df['change'] = df.close - df.close.shift(1)
        df['up'] = [x if x > 0 else 0 for x in df.change]
        df['down'] = [abs(x) if x < 0 else 0 for x in df.change]
        up_ewma = df['up'].ewm(span=self.RSI_lookback_period,
                                min_periods=self.RSI_lookback_period-1).mean()
        down_ewma = df['down'].ewm(span=self.RSI_lookback_period,
                                    min_periods=self.RSI_lookback_period-1).mean()
        df['relative_strength'] = up_ewma / down_ewma                                                              
        df['RSI'] = 100 - (100 / (1+df.relative_strength))
        
        df.drop(['change', 'up', 'down', 'relative_strength'], axis = 1, inplace=True)
        df = self.df

    def calc_MA(self): 
        df = self.df 
        df['ma_long'] = df.close.ewm(
            span=self.MA_long_period, min_periods=self.MA_long_period-1).mean()
        df['ma_short'] = df.close.ewm(
            span=self.MA_short_period, min_periods=self.MA_short_period-1).mean()
        self.df = df 

    def generate_signal(self):
        df = self.df
        self.calc_RSI()
        self.calc_MA()
        df['long'] = ((df.RSI < self.RSI_long) & (abs(df.ma_short - df.ma_long) > 35) & (df.ma_short > df.ma_long)) * 1
        df['short'] = ((df.RSI > self.RSI_short) & (abs(df.ma_short - df.ma_long) > 35) & (df.ma_short < df.ma_long)) * -1
        df['entry'] = df['short'] + df['long']
        df.dropna(inplace=True)
        self.df = df

class BasicMeanReversion(Strategy, Backtester):
    '''
    Sample mean reversion strategy from Ernest Chan's book 'Algorithmic Trading: Winning Strategies and Their Rationale'
    '''
    def __init__(self, holding_period, up_multiplier, down_multiplier, up_trend_signal, down_trend_signal, short_lookback_period, long_lookback_period, Exchange=None, start_year=None, start_month=None,
                 start_day=None, time_interval='1', csv_path=False):

        if csv_path == False and (Exchange == None or start_year == None or start_month == None or start_day == None):
            raise Exception('Missing arguments')

        Backtester.__init__(self, Exchange=Exchange, start_year=start_year, start_month=start_month,
                            start_day=start_day, holding_period=holding_period, up_multiplier=up_multiplier, down_multiplier=down_multiplier, time_interval=time_interval, csv_path=csv_path)

        self.up_trend_signal = up_trend_signal
        self.down_trend_signal = down_trend_signal
        self.long_lookback_period = long_lookback_period
        self.short_lookback_period = short_lookback_period

    
    def generate_signal(self):

        df = self.df

        df['min_price'] = df.close.rolling(self.short_lookback_period).min()
        df['max_price'] = df.close.rolling(self.short_lookback_period).max()
        df['up_trend_level'] = df.close.shift(self.long_lookback_period)*self.up_trend_signal
        df['down_trend_level'] = df.close.shift(self.long_lookback_period)*self.down_trend_signal

        df['long'] = ((df.close <= df.min_price) & (
            df.close > df.up_trend_level)) * 1
        df['short'] = ((df.close >= df.max_price) & (
            df.close < df.down_trend_level)) * -1

        df['entry'] = df['short'] + df['long']
        df.dropna(inplace=True)
        self.df = df  

class PolynomialTrend(Strategy, Backtester):
    '''
    Sample second degree polynomial trend strategy constantly rolls out predicted price trajectory using OLS to find the beta
    OLS model details: https://www.fsb.miamioh.edu/lij14/411_note_matrix.pdf
    '''

    def __init__(self, holding_period, up_multiplier, down_multiplier, lookback_period, lookahead_period, long_threshold, short_threshold, Exchange=None, start_year=None, start_month=None,
                 start_day=None, time_interval='1', csv_path=False):

        if csv_path == False and (Exchange == None or start_year == None or start_month == None or start_day == None):
            raise Exception('Missing arguments')

        Backtester.__init__(self, Exchange=Exchange, start_year=start_year, start_month=start_month,
                            start_day=start_day, holding_period=holding_period, up_multiplier=up_multiplier, down_multiplier=down_multiplier, time_interval=time_interval, csv_path=csv_path)

        self.lookahead_period = lookahead_period
        self.lookback_period = lookback_period
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
    
    @staticmethod
    def poly_model(series, lookahead):
        '''
        β =(X.T @ X)^−1 @ X.T @y
        '''
        y = series.values.reshape(-1,1)
        t = np.arange(len(y))
        X = np.c_[np.ones_like(y), t, t**2]
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        lookahead_value = np.array([1, t[-1]+lookahead, (t[-1]+lookahead)**2])
        predicted_y = lookahead_value @ beta
        return predicted_y

    def generate_signal(self):    
        df = self.df
        df['predicted_val'] = df.close.rolling(self.lookback_period).apply(self.poly_model, args=(self.lookahead_period,))
        df['delta'] = (df.predicted_val/df.close) -1
        df['long'] = (df['delta'] > self.long_threshold) * 1
        df['short'] = (df['delta'] < self.short_threshold) * -1 
        df['entry'] = df['short'] + df['long']
        df.dropna(inplace=True)

        self.df = df 
    

if __name__ == '__main__': 
    # ma = BasicMeanReversion(Exchange=Deribit, start_year='2020', start_month='01', start_day='26', holding_period=300, up_multiplier=1.02,
    #                 down_multiplier=0.98, up_trend_signal = 1.03, down_trend_signal=0.97, long_lookback_period=60*24*5, short_lookback_period=60*12)
    # filename = 'new_data.csv'
    # ma = BasicMeanReversion(holding_period=300, up_multiplier=1.02,
    #                         down_multiplier=0.98, up_trend_signal=1.03, down_trend_signal=0.97, long_lookback_period=60*24*5, short_lookback_period=60*12, csv_path=filename)
    
    poly = PolynomialTrend(Exchange=Deribit, start_year='2021', start_month='01', start_day='26', 
                            holding_period=300, up_multiplier=1.02, lookback_period=48, lookahead_period=4, 
                            long_threshold=0.03, short_threshold=-0.03,
                           down_multiplier=0.98, time_interval='30')
    poly.run_backtester()

