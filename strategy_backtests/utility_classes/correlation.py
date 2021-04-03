
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback
import operator
import numpy as np
import dateutil.parser as dp
import math
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import itertools


class CSVManager:

    def __init__(self, csv_path):

        self.data = pd.read_csv(csv_path, parse_dates=[
                                'timestamp'], index_col='timestamp')
        self.data.dropna(inplace=True)
        self.data.index.name = 'Starting time'
        self.data['timestamp'] = [i.isoformat()
                                  for i in self.data.index.to_pydatetime()]
        self.df = self.data.copy()

    def change_resolution(self, timeframe: str, file_type: str):

        time_symbols = ['T', 'H']

        if timeframe[-1] not in time_symbols:
            raise ValueError(
                'Only T(minute) and H(hour) timeframes are supported')

        if file_type == 'PERP':
            resample_dict = {'volume': 'sum', 'open': 'first',
                             'low': 'min', 'high': 'max',
                             'close': 'last', 'funding_rate': 'mean', 'timestamp': 'first'}
        elif file_type == 'SPOT':
            resample_dict = {'volume': 'sum', 'open': 'first',
                             'low': 'min', 'high': 'max',
                             'close': 'last', 'timestamp': 'first'}
        elif file_type == 'SPREAD':
            # timestamp,perp_volume,funding_rate,fut_volume,spread_open,spread_high,spread_low,spread_close
            resample_dict = {'perp_volume': 'sum', 'fut_volume': 'sum', 'spread_open': 'first',
                             'spread_low': 'min', 'spread_high': 'max',
                             'spread_close': 'last', 'spread_close_numerical': 'last', 'funding_rate': 'mean', 'timestamp': 'first'}
        elif file_type == 'FUTURE':
            resample_dict = {'volume': 'sum', 'open': 'first',
                             'low': 'min', 'high': 'max',
                             'close': 'last', 'timestamp': 'first'}
        self.df = self.data.resample(timeframe).agg(resample_dict)
        # self.timeframe = new_timeframe
        return self.df

class Correlation:
    def __init__(self, spread_folder_path: str, perp_folder_path: str=None, futures_folder_path: str=None):
        self.spread_folder_path = spread_folder_path
        self.perp_folder_path = perp_folder_path
        self.futures_folder_path = futures_folder_path

    def find_cointegration(self, df):
        n = df.shape[1]
        pvalue_matrix = np.ones((n, n))
        keys = df.keys()
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                result = coint(df[keys[i]], df[keys[j]])
                pvalue_matrix[i, j] = result[1]
                if result[1] < 0.05:
                    pairs.append((keys[i], keys[j]))
        return pvalue_matrix, pairs

    def pair_coint(self, all: bool, timeframe: str, *markets):
        df = pd.DataFrame()

        if all:
            markets = ['1INCH', 'AAVE', 'ADA', 'ALGO', 'ALT', 'ATOM', 'AVAX', 'BAL', 'BCH', 'BNB', 'BRZ', 'BSV', 'BTC', 'BTMX', 'CHZ', 'COMP', 'CREAM', 'CUSDT', 'DEFI', 'DMG', 'DOGE', 'DOT', 'DOTPRESPLIT', 'DRGN', 'EOS', 'ETC', 'ETH', 'EXCH', 'FIL', 'FLM', 'GRT', 'HNT',
                       'HT', 'KNC', 'LEO', 'LINK', 'LTC', 'MATIC', 'MID', 'MKR', 'MTA', 'NEO', 'OKB', 'OMG', 'PAXG', 'PRIV', 'RUNE', 'SHIT', 'SOL', 'SUSHI', 'SXP', 'THETA', 'TOMO', 'TRU', 'TRX', 'TRYB', 'UNI', 'UNISWAP', 'USDT', 'VET', 'WAVES', 'XAUT', 'XRP', 'XTZ', 'YFI', 'ZEC']

        else:
            markets = list(markets)

        for name in markets:
            for perp in os.scandir(self.perp_folder_path):
                if perp.path.split('/')[-1].split('-')[0] == name:
                    time_formated_perp = CSVManager(perp.path)
                    perp_file = time_formated_perp.change_resolution(
                        timeframe, 'PERP')
                    rows = []
                    for fut in os.scandir(self.futures_folder_path):
                        if fut.path.split('/')[-1].split('-')[0] == name:
                            
                            dates = fut.path.split('/')[-1].split('_')[0].split('-')[1]

                            time_formated_fut = CSVManager(fut.path)
                            fut_file = time_formated_fut.change_resolution(
                                timeframe, 'FUTURE')

                            perp_file.rename(
                                columns={'open': 'perp_open', 'high': 'perp_high', 'low': 'perp_low', 'close': 'perp_close', 'volume': 'perp_volume'}, inplace=True)
                            fut_file.rename(
                                columns={'open': 'fut_open', 'high': 'fut_high', 'low': 'fut_low', 'close': 'fut_close', 'volume': 'fut_volume'}, inplace=True)
                            
                            joint_df = pd.merge(perp_file, fut_file, how='inner', on=['timestamp'])

                            coint_result = coint(joint_df['perp_close'], joint_df['fut_close'])
                            rows.append([coint_result[1], dates])
            
                    new_df = pd.DataFrame(rows, columns=[name, 'futures_date'])
                    new_df.set_index('futures_date' ,drop=True, inplace=True)
            if len(df) == 0:
                df = new_df
            else:
                df = pd.merge(df, new_df, how='outer', on=['futures_date'])

        fig, ax = plt.subplots(figsize=(30, 5))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        mask = df.isnull()
        sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f", mask=mask, vmax=0.1,
                    vmin=0, ax=ax, annot_kws={"fontsize": 8})
        plt.title('Perp-fut spreads Cointegration')
        plt.show()


    def spreads_correlation_heatmap(self, futures_date: str, coint: bool, showing_only_below_threshold: bool=False, annot: bool = False, triangular: bool = False, min_cor: int = -1.0, max_cor: int = 1.0, timeframe: str = '1T'):

        time_symbols = ['T', 'H']

        if timeframe[-1] not in time_symbols:
            raise ValueError(
                'Only T(minute) and H(hour) timeframes are supported')

        spread_df = pd.DataFrame() #making a dataframe for spreads across different tokens
        errors=[]
        token_with_missing_values = []
        dates_aggregator_start = {} 
        dates_aggregator_end = {}
        
        # cuz futures might have different starting date, I record all the starting date here and only use the one date that appear the most
        for fut_data in os.scaspread_ndir(self.spread_folder_path):
            if fut_data.path.split('/')[-1].split('-')[1].split('_')[0] == futures_date:
                
                starting_time = pd.read_csv(fut_data.path)['timestamp'][0]
                ending_time = pd.read_csv(fut_data.path)['timestamp'].iloc[-1]
                
                if starting_time in dates_aggregator_start:
                    dates_aggregator_start[starting_time] += 1
                else:
                    dates_aggregator_start[starting_time] = 1

                if ending_time in dates_aggregator_end:
                    dates_aggregator_end[ending_time] += 1
                else:
                    dates_aggregator_end[ending_time] = 1

        starting_time = max(dates_aggregator_start.items(),
                            key=operator.itemgetter(1))[0]
        ending_time = max(dates_aggregator_end.items(),
                          key=operator.itemgetter(1))[0]
        try:
            for fut_data in os.scandir(self.spread_folder_path):

                date_in_filename = fut_data.path.split('/')[-1].split('-')[1].split('_')[0]
                if date_in_filename == futures_date and pd.read_csv(fut_data.path)['timestamp'][0] <= starting_time:
                    token_name = fut_data.path.split('/')[-1].split('-')[0]
                    
                    if timeframe == '1T':
                        all_spreads = pd.read_csv(fut_data.path)
                    else:
                        time_formated_file = CSVManager(fut_data.path)
                        all_spreads = time_formated_file.change_resolution(timeframe, 'SPREAD')
                    spread_df[token_name] = all_spreads.loc[(all_spreads['timestamp'] >= starting_time) & (all_spreads['timestamp'] <= ending_time)]['spread_close'].reset_index(drop=True)
                    # spread_df[token_name] = all_spreads.loc[(all_spreads['timestamp'] >= starting_time) & (
                    #     all_spreads['timestamp'] <= ending_time)]['spread_close_numerical'].reset_index(drop=True)
                else:
                    pass

            for (token_name, token_spread) in spread_df.tail(1).iteritems():
                if math.isnan(token_spread.values[0]):
                    token_with_missing_values.append(token_name)

            spread_df = spread_df.dropna()
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            if coint:
                corr_matrix, pairs = self.find_cointegration(spread_df)
                fig, ax = plt.subplots(figsize=(40, 40))

                if showing_only_below_threshold:
                    graph = sns.heatmap(corr_matrix, xticklabels=spread_df.columns,
                                yticklabels=spread_df.columns, cmap=cmap, annot=annot, fmt=".2f", mask=(corr_matrix >= 0.05))
                else:
                    graph = sns.heatmap(corr_matrix, xticklabels=spread_df.columns,
                                        yticklabels=spread_df.columns, cmap=cmap, annot=annot, fmt=".2f", mask=(corr_matrix >= 0.99))

                graph.tick_params(top=True, labeltop=True)
                plt.title('{} Cointegration Matrix P-Value'.format(futures_date))
                # print(f'Pairs that have p-value larger than 0.5')
            else:
                corr_matrix = spread_df.pct_change().corr(method='pearson')
                if triangular:
                # Generate a mask for the upper triangle
                    mask = np.zeros_like(corr_matrix, dtype=bool)
                    mask[np.triu_indices_from(mask)] = True
                else:
                    mask = None
                fig, ax = plt.subplots(figsize=(40, 40))
                sns.heatmap(corr_matrix, cmap=cmap, vmax=max_cor,
                            vmin=min_cor, linewidths=0.5, annot=annot, fmt=".2f", mask=mask)

                plt.title('{} spreads Pearson correlation'.format(futures_date))
            plt.show()
                
        except:
            errors.append(traceback.format_exc())
            pass

        for e in errors:
            print(e)

        ','.join(token_with_missing_values)
        print('spreads from {} to {}'.format(starting_time, ending_time))
        print(f'{token_with_missing_values} have missing values')
    
    def plot_single_token(self, symbol: str, timeframe: str = '1T', file_type: str='SPREAD'):
        for ticker in os.scandir(self.spread_folder_path):
            if ticker.path.split('/')[-1].split('_')[0] == symbol:
                file = CSVManager(ticker.path)
                file = file.change_resolution(timeframe, file_type)
                plt.title('{} {} spread data'.format(
                    ticker.path.split('/')[-1].split('_')[0], timeframe))
                if file_type == 'SPREAD':
                    plt.plot(file['spread_close'])  
                elif file_type == 'PERP':
                    plt.plot(file['close'])
                elif file_type == 'FUTURE':
                    plt.plot(file['close'])
        plt.show()

    def plot_historical_spread(self, symbol: str, timeframe: str = '1T', numerical_values: bool=False):
        
        plt.figure(figsize=(20,15))

        count = 1
        for ticker in os.scandir(self.spread_folder_path):
            if ticker.path.split('/')[-1].split('-')[0] == symbol:
                spread = CSVManager(ticker.path)
                spread = spread.change_resolution(timeframe, 'SPREAD')
                plt.subplot(5,3,count)
                # set a hard row number here; might need to change later
                plt.title('{} {} spread data'.format(ticker.path.split('/')[-1].split('_')[0], timeframe))
                if numerical_values:
                    plt.plot(spread['spread_close_numerical'])
                else:
                    plt.plot(spread['spread_close'])
                count += 1
        plt.tight_layout()
        plt.show()

    def plot_spread_price_distribution(self, symbol: str, timeframe: str = 'H', histogram: bool = True):
        
        all_apread_dfs = []
        futures_date=[]
        for ticker in os.scandir(self.spread_folder_path):
            if ticker.path.split('/')[-1].split('-')[0] == symbol:
                spread = CSVManager(ticker.path)
                spread = spread.change_resolution(timeframe, 'SPREAD')
                all_apread_dfs.append(spread)
                futures_date.append(ticker.path.split('/')[-1].split('_')[0])
        
        palette = itertools.cycle(sns.color_palette())
        sns.set(style='darkgrid')
        fig, ax = plt.subplots(figsize=(40, 40))

        for i in range(len(all_apread_dfs)):
            
            sns.histplot(data=all_apread_dfs[i]['spread_close'],
                         color=next(palette), kde=histogram)
        ax.set_xticks(range(-10, 10))
        plt.show()

if __name__ == '__main__':
    corr = Correlation(
        spread_folder_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_spreads',
        perp_folder_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_perps',
        futures_folder_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/expired_futures_data')
    corr.plot_spread_price_distribution('BTC')
