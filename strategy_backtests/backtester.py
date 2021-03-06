import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from historical_data_processor import DeribitDataProcessor as Deribit
from datetime import datetime as dt


class Backtester: 

    def __init__(self, Exchange, start_year, start_month, start_day, holding_period, up_multiplier, down_multiplier, time_interval='60'):
        self.exchange = Exchange(
            start_year, start_month, start_day, time_interval = time_interval)
        self.df = self.exchange.REST_polling()
        
        self.holding_period = holding_period
        self.constant_holding = holding_period

        self.end = self.df.timestamp.values[-1]

        self.open_positions = False 
        self.take_profits = None
        self.stop_loss = None
        self.direction = None # long =1 short = -1
        self.entry_price = None

        self.up_multiplier = up_multiplier
        self.down_multiplier = down_multiplier

        self.returns = []

    def trade(self, price, long=False, short=False): 
        if long or short: 
            self.entry_price = price # assuming that using the closing price can be filled here
            if long: 
                self.open_positions = True
                self.direction = 1 
                self.take_profits = price * self.up_multiplier
                self.stop_loss = price * self.down_multiplier
                self.returns.append(0)
            else: 
                self.open_positions = True
                self.direction = -1
                self.take_profits = price * self.down_multiplier
                self.stop_loss = price * self.up_multiplier
                self.returns.append(0)
        elif long and short:
            raise TypeError("Cannot enter long and short for the same trade")
        else:
            raise TypeError("Need to indicate long or short trade")

    def reset(self):
        self.open_positions = False
        self.take_profits = None
        self.stop_loss = None
        self.direction = None  # long =1 short = -1
        self.entry_price = None
        self.holding_period = self.constant_holding

    def close_trade(self, price):
        pnl = self.direction * (price/self.entry_price-1)
        self.returns.append(pnl)
        self.reset()
    
    def price_control(self, price, time): 
        if price >= self.take_profits and self.direction == 1: 
            self.close_trade(price)
        elif price <= self.stop_loss and self.direction == 1: 
            self.close_trade(price)
        elif price <= self.take_profits and self.direction == -1:
            self.close_trade(price)
        elif price >= self.stop_loss and self.direction == -1: 
            self.close_trade(price)
        elif time == self.end:
            self.close_trade(price)
        elif self.holding_period <= 0:
            self.close_trade(price)
        else:
            self.holding_period = self.holding_period - 1
            self.returns.append(0)

    def generate_signal(self):
        if 'entry' not in self.df.columns:
            raise Exception('No trade entered')
    
    def plot_performance(self):
        results = np.array(self.returns)
        all_returns = results.cumsum()
        time = np.array(self.df.timestamp)
        plt.plot(time, all_returns)
        plt.show()

    def run_backtester(self, using_close_price=True):
        self.generate_signal()
        for row in self.df.itertuples():
            if row.entry == 1 and self.open_positions == False:
                if using_close_price:
                    self.trade(row.close, long=True)
                else:
                    self.trade(row.next_open, long=True)
            elif row.entry == -1 and self.open_positions == False: 
                if using_close_price:
                    self.trade(row.close, short=True)
                else:
                    self.trade(row.next_open, short=True)
            elif self.open_positions: 
                self.price_control(row.close, row.timestamp)
            else:
                self.returns.append(0)
        self.plot_performance()


