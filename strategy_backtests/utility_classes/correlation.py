
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback
import operator
import numpy as np
import dateutil.parser as dp


class Correlation:

    def __init__(self, spread_folder_path: str):
        self.spread_folder_path = spread_folder_path
    
    def spreads_correlation_heatmap(self, futures_date: str, annot: bool=False, triangular: bool=False, min_cor: int = -1.0, max_cor: int=1.0):

        spread_df = pd.DataFrame() #making a dataframe for spreads across different tokens
        errors=[]
        dates_aggregator_start = {} 
        dates_aggregator_end = {}
        
        # cuz futures might have different starting date, I record all the starting date here and only use the one date that appear the most
        for fut_data in os.scandir(self.spread_folder_path):
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
                    all_spreads = pd.read_csv(
                        fut_data.path)['spread_close']
                    breakpoint()
                    spread_df[token_name] = all_spreads.loc[(
                        all_spreads['timestamp'] >= starting_time) & (all_spreads['timestamp'] <= ending_time)]
                else:
                    pass
            breakpoint()
            corr_matrix = spread_df.corr(method='pearson')
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            if triangular:
            # Generate a mask for the upper triangle
                mask = np.zeros_like(corr_matrix, dtype=np.bool)
                mask[np.triu_indices_from(mask)] = True
            else:
                mask = None

            sns.heatmap(corr_matrix, cmap=cmap, vmax=max_cor,
                        vmin=min_cor, linewidths=0.5, annot=annot, mask=mask)
            plt.title('{} spreads Pearson correlation'.format(futures_date))
            plt.show()
                
        except:
            errors.append(traceback.format_exc())
            pass

        for e in errors:
            print(e)


if __name__ == '__main__':
    corr = Correlation(
        spread_folder_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_spreads')
    corr.spreads_correlation_heatmap(
            futures_date='0925', triangular=True, min_cor=0)
