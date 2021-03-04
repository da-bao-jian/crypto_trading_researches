from abc import ABC, abstractmethod
from backtester import Backtester
import numpy as np
import matplotlib.pyplot as plt


class Strategy(ABC):

    @abstractmethod
    def generate_signal():
        raise NotImplementedError('should be implementing calc_signal() from subclass')


class MAStrategy(Strategy, Backtester): 

    def __init__(self, Exchange, start_year, start_month,
                 start_day, holding_period, up_multiplier, down_multiplier):
        Backtester.__init__(self, Exchange, start_year, start_month,
                            start_day, holding_period, up_multiplier, down_multiplier)
    
    def generate_signal(self): 
        df = self.df
        df['20_ma'] = df.close.rolling(20).mean()
        df['50_ma'] = df.close.rolling(50).mean()
        df['ma_deficit'] = df.20_ma - df.50_ma
        df['long'] = 1 if (df.ma_deficit > 0 and df.ma_deficit.shift(1) < 0)
        df['short'] = -1 if (df.ma_deficit > 0 and df.ma_deficit.shift(1) > 0)
        df['number_of_entries'] = df.long + df.short
        self.df = df
    
    def plot_performance(self):
        results = np.array(self.returns)   
        all_returns = results.cumsum()
        plt.plot(all_returns)
        plt.show()