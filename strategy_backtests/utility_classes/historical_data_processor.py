import json
import websockets
import asyncio
from keys import BINANCE_API_KEY, BINANCE_API_SECRET_KEY, FTX_API_KEY, FTX_API_SECRET
from binance.client import Client
import csv
import os
import sys
import time
from datetime import datetime as dt
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List 
import urllib.parse  
from requests import Request, Session, Response
import hmac
import dateutil.parser as dp
import traceback



def timestamp_to_unix(year, month, day):
    return dt(year, month, day).timestamp()


def unix_to_timestamp(uni):
    return dt.fromtimestamp(uni)


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
    
    def REST_polling(self, write_file = True, name_of_csv='new_data.csv', cleaned_column=True):

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
            pandaed['next_open'] = pandaed.open.shift(-1) #change this to the random price within one standard deviation from last ohlc 
            pandaed.drop(pandaed.tail(1).index,
                    inplace=True)
            
            if cleaned_column:
                pandaed = self.df_column_orgnizer(pandaed)
            
            df = df.append(pandaed, ignore_index=True)

            # print(f'showing data of {new_day}')
            # print(pandaed)
            time.sleep(0.01)

        if write_file:  
            df.to_csv(name_of_csv)
            return df
        else:
            return df


class FTXDataProcessor:

    _ENDPOINT = 'https://ftx.com/api/'

    def __init__(self, api_key=None, api_secret=None, subaccount_name=None):
        self._session = Session()
        self._api_key = api_key
        self._api_secret = api_secret
        self._subaccount_name = subaccount_name

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('GET', path, params=params)

    def _post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('POST', path, json=params)

    def _delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('DELETE', path, json=params)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        request = Request(method, self._ENDPOINT + path, **kwargs)
        self._sign_request(request)
        response = self._session.send(request.prepare())
        return self._process_response(response)

    def _sign_request(self, request: Request) -> None:
        ts = int(time.time() * 1000)
        prepared = request.prepare()
        signature_payload = f'{ts}{prepared.method}{prepared.path_url}'.encode(
        )
        if prepared.body:
            signature_payload += prepared.body
        signature = hmac.new(self._api_secret.encode(),
                             signature_payload, 'sha256').hexdigest()
        request.headers['FTX-KEY'] = self._api_key
        request.headers['FTX-SIGN'] = signature
        request.headers['FTX-TS'] = str(ts)
        if self._subaccount_name:
            request.headers['FTX-SUBACCOUNT'] = urllib.parse.quote(
                self._subaccount_name)

    def _process_response(self, response: Response) -> Any:
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        else:
            if not data['success']:
                raise Exception(data['error'])
            return data['result']

    def list_futures(self):
        return self._get('futures')

    def list_markets(self) -> List[dict]:
        return self._get('markets')

    def get_orderbook(self, market: str, depth: int = None):
        return self._get(f'markets/{market}/orderbook', {'depth': depth})

    def get_trades(self, market: str):
        return self._get(f'markets/{market}/trades')

    def get_account_info(self):
        return self._get(f'account')

    def get_open_orders(self, market: str = None):
        return self._get(f'orders', {'market': market})

    def get_order_history(self, market: str = None, side: str = None, order_type: str = None, start_time: float = None, end_time: float = None) -> List[dict]:
        return self._get(f'orders/history', {'market': market, 'side': side, 'orderType': order_type, 'start_time': start_time, 'end_time': end_time})

    def get_conditional_order_history(self, market: str = None, side: str = None, type: str = None, order_type: str = None, start_time: float = None, end_time: float = None) -> List[dict]:
        return self._get(f'conditional_orders/history', {'market': market, 'side': side, 'type': type, 'orderType': order_type, 'start_time': start_time, 'end_time': end_time})

    def modify_order(
        self, existing_order_id: Optional[str] = None,
        existing_client_order_id: Optional[str] = None, price: Optional[float] = None,
        size: Optional[float] = None, client_order_id: Optional[str] = None,
    ):
        assert (existing_order_id is None) ^ (existing_client_order_id is None), \
            'Must supply exactly one ID for the order to modify'
        assert (price is None) or (
            size is None), 'Must modify price or size of order'
        path = f'orders/{existing_order_id}/modify' if existing_order_id is not None else \
            f'orders/by_client_id/{existing_client_order_id}/modify'
        return self._post(path, {
            **({'size': size} if size is not None else {}),
            **({'price': price} if price is not None else {}),
            ** ({'clientId': client_order_id} if client_order_id is not None else {}),
        })

    def get_conditional_orders(self, market: str = None):
        return self._get(f'conditional_orders', {'market': market})

    def place_order(self, market: str, side: str, price: float, size: float, type: str = 'limit',
                    reduce_only: bool = False, ioc: bool = False, post_only: bool = False,
                    client_id: str = None):
        return self._post('orders', {'market': market,
                                     'side': side,
                                     'price': price,
                                     'size': size,
                                     'type': type,
                                     'reduceOnly': reduce_only,
                                     'ioc': ioc,
                                     'postOnly': post_only,
                                     'clientId': client_id,
                                     })

    def place_conditional_order(
        self, market: str, side: str, size: float, type: str = 'stop',
        limit_price: float = None, reduce_only: bool = False, cancel: bool = True,
        trigger_price: float = None, trail_value: float = None
    ):
        """
        To send a Stop Market order, set type='stop' and supply a trigger_price
        To send a Stop Limit order, also supply a limit_price
        To send a Take Profit Market order, set type='trailing_stop' and supply a trigger_price
        To send a Trailing Stop order, set type='trailing_stop' and supply a trail_value
        """
        assert type in ('stop', 'take_profit', 'trailing_stop')
        assert type not in ('stop', 'take_profit') or trigger_price is not None, \
            'Need trigger prices for stop losses and take profits'
        assert type not in ('trailing_stop',) or (trigger_price is None and trail_value is not None), \
            'Trailing stops need a trail value and cannot take a trigger price'

        return self._post('conditional_orders',
                          {'market': market, 'side': side, 'triggerPrice': trigger_price,
                           'size': size, 'reduceOnly': reduce_only, 'type': 'stop',
                           'cancelLimitOnTrigger': cancel, 'orderPrice': limit_price})

    def cancel_order(self, order_id: str):
        return self._delete(f'orders/{order_id}')

    def cancel_orders(self, market_name: str = None, conditional_orders: bool = False,
                      limit_orders: bool = False):
        return self._delete(f'orders', {'market': market_name,
                                        'conditionalOrdersOnly': conditional_orders,
                                        'limitOrdersOnly': limit_orders,
                                        })

    def get_fills(self):
        return self._get(f'fills')

    def get_balances(self):
        return self._get('wallet/balances')

    def get_deposit_address(self, ticker: str):
        return self._get(f'wallet/deposit_address/{ticker}')

    def get_positions(self, show_avg_price: bool = False):
        return self._get('positions', {'showAvgPrice': show_avg_price})

    def get_position(self, name: str, show_avg_price: bool = False):
        return next(filter(lambda x: x['future'] == name, self.get_positions(show_avg_price)), None)

    def get_all_trades(self, market: str, start_time: float = None, end_time: float = None):
        '''
         {'id': 576467907, 'liquidation': False, 'price': 53727.0, 'side': 'sell', 'size': 0.0037, 'time': '2021-03-10T04:59:57.855187+00:00'}
        '''
        ids = set()
        limit = 100
        results = []
        while True:
            response = self._get(f'markets/{market}/trades', {
                'end_time': end_time,
                'start_time': start_time
            })
            deduped_trades = [r for r in response if r['id'] not in ids]
            results.extend(deduped_trades)
            ids |= {r['id'] for r in deduped_trades}
            print(
                f'Adding {len(response)} trades with end time {dt.fromtimestamp(int(end_time))}')
            if len(response) == 0:
                break
            end_time = min(dt.fromisoformat(t['time'])
                           for t in response).timestamp()
            if len(response) < limit:
                break

        df = pd.DataFrame(results)
        df = df.drop(columns=['time'])
        df = df.rename(columns={"startTime": "timestamp"})
        df['next_open'] = df.open.shift(-1)
        return df

        return df

    def get_all_OHCL(self, market: str, resolution: int = 60, start_time: float = None, end_time: float = None, limit: int = 5000):
        '''
        {'close': 49483.0, 'high': 49510.0, 'low': 49473.0, 'open': 49475.0, 'startTime': '2021-03-07T05:00:00+00:00', 'time': 1615093200000.0, 'volume': 649052.5699}
        '''
        if end_time == None:
            end_time = time.time()

        if start_time == None:
            start_time = 1557288000
        
        unix_times = set()
        limit = 100
        results = []
        
        while True:

            response = self._get(f'markets/{market}/candles', {
                'end_time': end_time,
                'start_time': start_time,
                'resolution': resolution,
                'limit': 5000
            })
            deduped_candles = [r for r in response if r['time'] not in unix_times]


            results = deduped_candles + results
            unix_times |= {r['time'] for r in deduped_candles}
            print(
                f'Adding {len(response)} candles with start time {dt.fromtimestamp(int(end_time))}')
            if len(response) == 0:
                break
            end_time = min(dt.fromisoformat(t['startTime'])
                           for t in response).timestamp()
            if len(response) < limit:
                break

        df = pd.DataFrame(results)
        df = df.drop(columns=['time'])
        df = df.rename(columns={"startTime": "timestamp"})
        df['next_open'] = df.open.shift(-1)
        
        return df

    def get_PERP_OHCL(self, market: str, resolution: int = 60, start_time: float = None, end_time: float = None, limit: int = 5000):
        '''
        {'close': 49483.0, 'high': 49510.0, 'low': 49473.0, 'open': 49475.0, 'startTime': '2021-03-07T05:00:00+00:00', 'time': 1615093200000.0, 'volume': 649052.5699}
        '''
        
        if market[-4:] == 'PERP': 
        
            if end_time == None:
                end_time = time.time()

            if start_time == None:
                start_time = 1557288000

            unix_times = set()
            limit = 100
            results = []

            while True:

                response = self._get(f'markets/{market}/candles', {
                    'end_time': end_time,
                    'start_time': start_time,
                    'resolution': resolution,
                    'limit': 5000
                })

                deduped_candles = [
                    r for r in response if r['time'] not in unix_times]

                if len(deduped_candles) > 0: 
                    funding_response = self._get(f'/funding_rates', {
                        'end_time': int(deduped_candles[-1]['time']/1000),
                        'start_time': int(deduped_candles[0]['time']/1000)-3600,
                        'future': market
                    })

                for data in deduped_candles:
                    for funding_period in funding_response:
                        if data['startTime'][:13] == funding_period['time'][:13]:
                            data['funding_rate'] = funding_period['rate']

                results = deduped_candles + results
                unix_times |= {r['time'] for r in deduped_candles}
                print(
                    f'Adding {len(response)} candles with start time {dt.fromtimestamp(int(end_time))}')
                if len(response) == 0:
                    break
                end_time = min(dt.fromisoformat(t['startTime'])
                            for t in response).timestamp()
                if len(response) < limit:
                    break

            df = pd.DataFrame(results)
            df = df.drop(columns=['time'])
            df = df.rename(columns={"startTime": "timestamp"})
            df['next_open'] = df.open.shift(-1)

            return df
        else:
            raise ValueError('Please only enter a valid perpetual contract to use this function')



    def get_expired_futures_OHCL(self, market: str, resolution: int = 60, start_time: float = None, end_time: float = None, limit: int = 5000):
        try:
            int(market[-4:])
        except:
            raise ValueError('Please specify the expiration date')
        
        if end_time == None:
            expired_futures = self._get('expired_futures')
            for ticker in expired_futures:
                if ticker['name'] == market:
                    end_time = int(dp.parse(ticker['expiry']).timestamp())
        res = self.get_all_OHCL(market = market, resolution = resolution, end_time = end_time)
        return res 

    def get_historical_funding(self, market: str, path:str, start_time: float = None, end_time: float = None, limit: int=5000):
        try:
            market[-4:] == 'PERP'
        except:
            raise ValueError('Please choose a valid futures contract that has funding')
        
        if end_time == None:
            end_time = time.time()

        if start_time == None:
            start_time = 1557288000
        
        unix_times = set()
        limit = 100
        results = []
        while True:

            response = self._get(f'/funding_rates', {
                'end_time': end_time,
                'start_time': start_time,
                'future': market
            })
            deduped_candles = [
                r for r in response if r['time'] not in unix_times]
            results = results + deduped_candles
            unix_times |= {r['time'] for r in deduped_candles}
            stamp = dp.parse(results[-1]['time']).timestamp()
            print(
                f'Adding {len(response)} funding data with start time {dt.fromtimestamp(int(stamp))}')
            if len(response) == 0:
                break
            m=min(t['time'] for t in response)
            end_time = int(dp.parse(m).timestamp())
            if len(response) < limit:
                break
        
        df = pd.DataFrame(results)
        df = df.iloc[::-1]
        df = df.rename(columns={"time": "timestamp"})

        file_path = os.path.join(
            path, "{}_historical_funding_data.csv".format(market))
        df.to_csv(file_path, index=False)

        return df

    def get_all_historical_funding_rates(self, path: str):
        errors=[]
        all_tickers = self.get_all_perp_tickers()
        for ticker in all_tickers:
            ticker = ticker+'-PERP'
            try:
                self.get_historical_funding(market = ticker, path = path)
            except:
                errors.append(f'Error occured while fetching {ticker} data')
                pass
        for e in errors:
            print(e)

    def get_expired_futures_dates(self):
        '''
        expiration date: 1225 | Date December 2020
        expiration date: 0925 | Date September 2020
        expiration date: 0626 | Date June 2020
        expiration date: 20200327 | Date March 2020
        expiration date: 20191227 | Date December 2019
        expiration date: 20190927 | Date September 2019
        expiration date: 20190628 | Date June 2019
        expiration date: 20190329 | Date March 2019
        '''
        expired_futures = self._get('expired_futures')
        for ticker in expired_futures:
            if ticker['underlying'] == 'ETH':
                time_stamp = ticker['name'][4:]
                month_year = ticker['expiryDescription']
                print(f'expiration date: {time_stamp} | Date {month_year}')
        print('BTC-0626 data is missing from FTX end')

    def get_all_perp_tickers(self):
        all_tickers=[]
        response = self._get('futures')
        for ticker in response:
            if ticker['perpetual'] and ticker['name'] not in all_tickers:
                all_tickers.append(ticker['underlying'])
        return all_tickers

    def get_all_expired_futures_that_have_perps(self) -> List:
        expired_futures_arr = []
        expired_futures = self._get('expired_futures')
        current_perp_futures = self.get_all_perp_tickers()
        for ticker in expired_futures:
            if ticker['underlying'] in current_perp_futures and ticker['name'][4:8] != 'MOVE' and ticker['expiryDescription'] != 'March 2019' and ticker['expiryDescription'] != 'June 2019':
                expired_futures_arr.append(ticker['name'])

        return expired_futures_arr
    
    def write_all_expired_futures_OHCL(self, path: str, resolution: int=60):

        expired_futures = self.get_all_expired_futures_that_have_perps()
        
        errors = []

        for ticker in expired_futures:
            try:
                if ticker[-4:] != 'PERP':
                    expired_future_dataframe = self.get_expired_futures_OHCL(market=ticker, resolution=resolution)
                    file_path = os.path.join(
                        path, "{}_{}_data.csv".format(ticker, resolution))
                    expired_future_dataframe.to_csv(file_path, index=False)
            except:
                errors.append('There has been an error in writing file {}'.format(ticker))
                pass
        
        for e in errors:
            print(e)

    def write_all_PERPs_OHCL(self, path: str, resolution: int=60):
        all_tickers = []
        response = self._get('futures')
        for perp in response:
            if perp['perpetual'] and perp['name'] not in all_tickers:
                all_tickers.append(perp['name'])

        errors = []

        for ticker in all_tickers:
            try:
                perp_dataframe = self.get_PERP_OHCL(
                    market=ticker, resolution=resolution)
                file_path = os.path.join(
                    path, "{}_historical_data.csv".format(ticker))
                perp_dataframe.to_csv(file_path, index=False)        
            except:
                errors.append(traceback.format_exc())
                pass

        for e in errors:
            print(e)

    def get_spreads(self, perp_path: str, futures_path: str):
        '''
        given perp and futures data, calculate the spreads
        '''

        perp_data_df = pd.read_csv(perp_path).drop(columns=['next_open'])
        futures_data_df = pd.read_csv(futures_path).drop(columns=['next_open'])

        perp_data_df.rename(
            columns={'open': 'perp_open', 'high': 'perp_high', 'low':'perp_low', 'close':'perp_close', 'volume': 'perp_volume'}, inplace=True)
        futures_data_df.rename(
            columns={'open': 'fut_open', 'high': 'fut_high', 'low': 'fut_low', 'close': 'fut_close', 'volume': 'fut_volume'}, inplace=True)
        
        joint_df = pd.merge(perp_data_df, futures_data_df, how='inner', on=['timestamp'])

        joint_df['spread_open'] = (
            (joint_df['perp_open'] - joint_df['fut_open']) / joint_df['perp_open']) * 100
        joint_df['spread_high'] = ((
            joint_df['perp_high'] - joint_df['fut_high']) / joint_df['perp_high']) * 100
        joint_df['spread_low'] = ((
            joint_df['perp_low'] - joint_df['fut_low']) / joint_df['perp_low']) * 100
        joint_df['spread_close'] = ((
            joint_df['perp_close'] - joint_df['fut_close']) / joint_df['perp_close']) * 100
        joint_df['spread_close_numerical'] = (joint_df['perp_close'] - joint_df['fut_close'])

        joint_df.drop(columns=['perp_open', 'perp_high',
                               'perp_low', 'perp_close', 'fut_open', 'fut_high', 'fut_low', 'fut_close'], inplace=True)
        return joint_df
    
    def write_all_spreads(self, perp_folder_path: str, futures_folder_path: str, output_path: str):
        errors=[]
        for fut_data in os.scandir(futures_folder_path):
            try:
                future_name = fut_data.path.split('/')[-1].split('_')[0]
                perp_name = fut_data.path.split('/')[-1].split('-')[0]

                for perp_data in os.scandir(perp_folder_path):
                    if perp_name == perp_data.path.split('/')[-1].split('-')[0]:
                        spread_df = self.get_spreads(perp_path=perp_data.path, futures_path=fut_data.path)

                        file_path = os.path.join(
                            output_path, "{}_spread_data.csv".format(future_name))
                        spread_df.to_csv(file_path, index=False)

                        print(f'Writing spread for {future_name}')
            except:
                errors.append(traceback.format_exc())
                pass

        for e in errors:
            print(e)

if __name__ == '__main__':

    # deribit = DeribitDataProcessor('2021', '02', '26', time_interval='30')
    acc = FTXDataProcessor(api_key=FTX_API_KEY, api_secret=FTX_API_SECRET)
    # res = acc._request('GET', 'markets/BTC-PERP/candles?resolution=60&limit=500')
    # acc.write_all_PERPs_OHCL(path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_perps')
    # acc.write_all_spreads(
    #     perp_folder_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_perps',
    #     futures_folder_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/expired_futures_data',
    #     output_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_spreads')

    acc.get_spreads(
        perp_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/all_perps/ETH-PERP_historical_data.csv', 
        futures_path='/home/harry/trading_algo/crypto_trading_researches/strategy_backtests/historical_data/expired_futures_data/ETH-0925_60_data.csv')
