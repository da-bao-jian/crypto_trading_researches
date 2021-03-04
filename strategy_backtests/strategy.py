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
    
    @abstractmethod
    def plot_performance():
        raise NotImplementedError(
            'should be implementing plot_performance() from subclass')


class MAStrategy(Strategy, Backtester): 
    '''
    test Strategy, don't use it
    '''
    def __init__(self, Exchange, start_year, start_month,
                 start_day, holding_period, up_multiplier, down_multiplier):
        Backtester.__init__(self, Exchange, start_year, start_month,
                            start_day, holding_period, up_multiplier, down_multiplier)
    
    def generate_signal(self):
        df = self.df
        df['20ma'] = df.close.rolling(20).mean()
        df['50ma'] = df.close.rolling(50).mean()
        df['ma_deficit'] = df['20ma'] - df['50ma']
        
        
        df['long'] = ((df.ma_deficit > 0) & (df.ma_deficit.shift(1) < 0)) * 1
        df['short'] = ((df.ma_deficit < 0) & (df.ma_deficit.shift(1) > 0)) * -1
        
        df['entry'] = df['long'] + df['short']
        self.df = df
            
    def plot_performance(self):
        results = np.array(self.returns)  
        all_returns = results.cumsum()
        # breakpoint()
        plt.plot(all_returns, list(range(0, len(all_returns))))
        plt.show()

if __name__ == '__main__': 
    ma = MAStrategy(Deribit, '2021', '02', '26', holding_period=20, up_multiplier=1.06, down_multiplier=0.97)


    ma.run_backtester()
    print(ma.returns)
