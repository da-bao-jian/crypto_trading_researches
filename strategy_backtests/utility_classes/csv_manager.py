import pandas as pd
import dateutil.parser as dp


class CSVManager:

    def __init__(self, csv_path):

        self.data = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        self.data.dropna(inplace=True)
        self.df = self.data.copy()

    def change_resolution(self, timeframe: str, file_type: str):
        
        time_symbols = ['T', 'H']
        
        if timeframe[-1] not in time_symbols:
            raise ValueError('Only T(minute) and H(hour) timeframes are supported')
        
        if file_type == 'PERP':
            resample_dict = {'volume': 'sum', 'open': 'first',
                            'low': 'min', 'high': 'max',
                             'close': 'last', 'funding_rate': 'mean'}
        else if file_type == 'SPREAD':
            # timestamp,perp_volume,funding_rate,fut_volume,spread_open,spread_high,spread_low,spread_close
            resample_dict = {'perp_volume': 'sum', 'fut_volume': 'sum', 'spread_open': 'first',
                             'spread_low': 'min', 'spread_high': 'max',
                             'spread_close': 'last', 'funding_rate': 'mean'}
        else if file_type == 'FUTURE':
            resample_dict = {'perp_volume': 'sum', 'fut_volume': 'sum', 'spread_open': 'first',
                             'spread_low': 'min', 'spread_high': 'max',
                             'spread_close': 'last'}
        self.df = self.data.resample(timeframe).agg(resample_dict)
        # self.timeframe = new_timeframe
        return self.df

if __name__ == '__main__':
    perp = CSVManager(
        '/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_perps/WAVES-PERP_historical_data.csv')
    perp.change_resolution('H', 'PERP')
