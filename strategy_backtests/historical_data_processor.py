import json
import websockets
import asyncio
from keys import BINANCE_API_KEY, BINANCE_API_SECRET_KEY
from binance.client import Client
import csv
import os
import sys
import time
from datetime import datetime as dt
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Binance OHCL data
class BinanceDataProcessor:

    def __init__(self, key, secret):
        self.key = key
        self.secret = secret
        self.client = Client(self.key, self.secret)

    def binance_historical_data_recorder(self, name_of_csv, symbol="BTCUSDT"):
        ohlc_data = open(name_of_csv, 'w', newline='')
        ohlc_writer = csv.writer(ohlc_data, delimiter=',')

        ohlc = self.client.get_historical_klines(
            symbol, Client.KLINE_INTERVAL_1MINUTE, "1 Dec, 2016", "24 Feb, 2021")

        for candlestick in ohlc:
            ohlc_writer.writerow(candlestick)

        ohlc_data.close()


# cctx ftx
# import ccxt  # ccxt only run on python 3.8.5 but not 3.9.0
# import pandas as pd
# import datetime
# import asyncio
# import csv

# ftx = ccxt.ftx()

# def fetch_data_thru_cctx(symbol='BTC-PERP', time_interval='1m', since='2021-02-25T01:29:00.000Z', limit=1500, params={}):

#     since = ftx.parse8601(since)

#     data = ftx.fetch_ohlcv(symbol, time_interval, since, limit)
#     data_framed = pd.DataFrame(data)

#     data_framed.columns = (
#         ['UTC timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

#     def parsing_UNIX(unix):  # parsing unix timestamp
#         return ftx.iso8601(unix)
#     data_framed['UTC timestamp'] = data_framed['UTC timestamp'].apply(
#         parsing_UNIX)
#     file = open('recent_kline_data.csv', 'w', newline='')
#     candlestick_writer = csv.writer(file, delimiter=',')
#     print(data_framed)
#     # print(data_framed.iloc[:1])
#     # print(data_framed.iloc[:1].iloc[:, 1])


# API Doc: https://docs.deribit.com/?python#public-get_instrument
class DeribitDataProcessor:

    def __init__(self, start_year, start_month, start_day, end_year=None, end_month=None, end_day=None, 
                symbol="BTC-PERPETUAL", time_interval='60'):
        
        str_start_time = f'{start_day}/{start_month}/{start_year}'
        self.start = dt.timestamp(
            dt.strptime(str_start_time, "%d/%m/%Y"))

        if end_year == None and end_month == None and end_day == None:
            self.end = dt.timestamp(dt.now())
        else:
            str_end_time = f'{end_day}/{end_month}/{end_year}'
            self.end = dt.timestamp(
                dt.strptime(str_end_time, "%d/%m/%Y"))

        self.symbol = symbol
        self.time_interval = time_interval

        self.msg = {
            "jsonrpc": "2.0",
            "id": 833,
            "method": "public/get_tradingview_chart_data",
            "params": {
                "instrument_name": self.symbol,
                "start_timestamp": int(round(self.start))*1000,
                "end_timestamp": int(round(self.end))*1000,
                "resolution": self.time_interval
            }
        }
        # print(self.msg)

    async def call_api(self, msg):
        async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
            await websocket.send(msg)
            while websocket.open:
                response = await websocket.recv()
                return response

    def api_loop(self, api_func, msg):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete((api_func(msg)))

    def retrieve_data(self):
        response = self.api_loop(self.call_api, json.dumps(self.msg))
        return response   # raw form, needs json.loads() later

    def to_pandas_df(self, response):
        jsoned_response = json.loads(response)
        df = pd.DataFrame(jsoned_response['result'])
        df['ticks'] = df.ticks/1000
        df['timestamp'] = [dt.fromtimestamp(time) for time in df.ticks]
        return df

    def deribit_historical_data_recorder(self, name_of_csv):
        df = self.to_pandas_df(self.retrieve_data())
        return df.to_csv(name_of_csv, encoding='utf-8', index=False)
    
    def df_column_orgnizer(self, df):
        needed_columns = ['volume', 'open',
                          'low', 'high', 'close', 'timestamp', 'next_open']
        for col in df.columns:
            if col not in needed_columns:
                del df[col]
        return df
    
    def REST_polling(self, write_file = False, name_of_csv='new_data', cleaned_column=True):

        day_span = (dt.now()-dt.fromtimestamp(self.start)).days
        df = pd.DataFrame()
        new_day = dt.fromtimestamp(self.start)

        for d in range(day_span): 

            unix_past_day = dt.timestamp(new_day)
            new_day += datetime.timedelta(days=1)
            unix_future_day = dt.timestamp(new_day)

            new_request = {
                "jsonrpc": "2.0",
                "id": 833,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": self.symbol,
                    "start_timestamp": unix_past_day*1000,
                    "end_timestamp": unix_future_day*1000,
                    "resolution": self.time_interval
                }
            }

            response = self.api_loop(self.call_api, json.dumps(new_request))
            pandaed = self.to_pandas_df(response)
            pandaed['next_open'] = pandaed.open.shift(-1)
            pandaed.drop(pandaed.tail(1).index,
                    inplace=True)
            
            if cleaned_column:
                pandaed = self.df_column_orgnizer(pandaed)
            
            df = df.append(pandaed, ignore_index=True)

            # print(f'showing data of {new_day}')
            # print(pandaed)
            time.sleep(0.01)

        if write_file:  
            return df.to_csv(name_of_csv)
        else:
            return df



if __name__ == '__main__':

    deribit = DeribitDataProcessor('2021', '02', '26', time_interval='30')
    # df = res.to_pandas_df(res.retrieve_data())
    complete_data = deribit.REST_polling(True, 'BTCPerp-09-26-20-to-03-01-21')
    # dataframe = pd.read_csv('BTCPerp-09-26-20-to-03-01-21.csv')
    # plt.plot(dataframe.timestamp, dataframe.close)
