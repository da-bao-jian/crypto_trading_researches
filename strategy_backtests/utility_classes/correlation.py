
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback

class Correlation:

    def __init__(self, spread_folder_path: str):
        self.spread_folder_path = spread_folder_path
    
    def spreads_correlation_heatmap(self, futures_date: str, annot: bool=False):

        spread_df = pd.DataFrame() #making a dataframe for spreads across different tokens
        errors=[]

        try:
            for fut_data in os.scandir(self.spread_folder_path):
                    if fut_data.path.split('/')[-1].split('-')[1].split('_')[0] == futures_date:
                        token_name = fut_data.path.split('/')[-1].split('-')[0]
                        spread_df[token_name] = pd.read_csv(fut_data.path)['spread_close']

            corr_matrix = spread_df.corr(method='pearson')
            # f, ax = plt.subplots(figsize=(11, 9))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Generate a mask for the upper triangle
            #mask = np.zeros_like(corr_matrix, dtype=np.bool)
            #mask[np.triu_indices_from(mask)] = True

            

            sns.heatmap(corr_matrix, cmap=cmap, vmax=1.0,
                        vmin=-1.0, linewidths=0.5, annot=annot)
            plt.title('{} spreads Pearson correlation'.format(futures_date))
            plt.show()
                
        except:
            errors.append(traceback.format_exc())
            pass

        for e in errors:
            print(e)


        # return corr_matrix
