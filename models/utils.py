import os
import time
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict
import json

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionStyle(Enum):
    EURO = "EURO"
    AMER = "AMER"

def classify_option(ticker: str):
    call_maturities = ['A', 'B', 'C' ,'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    put_maturities = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    if ticker[4] in call_maturities:
        return OptionType.CALL
    elif ticker[4] in put_maturities:
        return OptionType.PUT
    else:
        raise ValueError("Ticker does not specify option type.")

def get_prefixo_ticker(ticker: str):
    return ticker[:4]

colors = ['black', 'red', 'green', 'blue', 'olive', 'purple', 'orange', 'brown', 'pink', 'gray']

options_data = lambda database: f'../data/Histórico B3/Negociações {database}.csv'


def gen_date_list(ini_date: str, end_date: str):
  start_date = datetime.strptime(ini_date, '%Y-%m-%d')
  end_date = datetime.strptime(end_date, '%Y-%m-%d')
  delta = timedelta(days=1)

  date_list = []
  while start_date <= end_date:
    date_list.append(start_date.strftime('%Y%m%d'))
    start_date += delta

  return date_list


def gen_valid_date_list(ini_date: str, ndays: int):
    start_date = datetime.strptime(ini_date, '%Y-%m-%d')
    delta = timedelta(days=1)
    counter = 0
    date_list = []
    while counter < ndays:
        if os.path.exists(options_data(database=start_date.strftime('%Y%m%d'))):
            date_list.append(start_date.strftime('%Y%m%d'))
            counter += 1
        start_date += delta

    return date_list

def ndays(database:str, ndays: int):
    start_date = datetime.strptime(database, '%Y-%m-%d')

    return (start_date + timedelta(days=ndays)).strftime('%Y-%m-%d')

def measure(func):
    start_time = time.time()
    res = func()
    end_time = time.time()
    diff = end_time - start_time
    if diff < 60 * 2:
        print(f"Elapsed time for {func.__name__}: {round(diff, 2)} seconds")
    else:
        print(f"Elapsed time for {func.__name__}: {round((diff)/60, 2)} minutes")
    return res

def get_option_data(asset_ticker: str, start_date: str, end_date: str, style: OptionStyle = OptionStyle.EURO, type: OptionType = OptionType.CALL) -> pd.DataFrame:
    """
    Placeholder function to retrieve option prices.
    
    Parameters:
    ticker (str): option ticker symbol
    start_date (str): starat date for data retrieval %Y-%m-%d
    end_date (str): end date for data retrieval %Y-%m-%d
    
    Returns:
    np.ndarray: array of option prices
    """
    
    databases = gen_date_list(start_date, end_date)
    full_prices = pd.DataFrame()
    for database in databases:
        file_path = options_data(database.replace('-', ''))
        prices = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
        if prices.empty: continue
        prices = prices[prices['Ticker'].str.contains(get_prefixo_ticker(asset_ticker))]
        prices['Asset Price'] = prices[prices['Ticker'] == asset_ticker]['LastPrice'].values[0]
        full_prices = pd.concat([full_prices, prices], ignore_index=True)
    full_prices.dropna(subset=['Strike'], inplace=True)
    full_prices['Days to Maturity'] = days_to_maturity(full_prices['TradeDate'].tolist(), full_prices['Maturity'].tolist())
    full_prices = full_prices[full_prices['Days to Maturity'] > 0.0] 
    type_value = "CALL" if type == OptionType.CALL else "PUTT"
    full_prices = full_prices[full_prices['Type'] == type_value]
    full_prices = full_prices[full_prices['Style'] == style.value]
    return full_prices


def get_asset_prices(asset_ticker: str, start_date: str, end_date: str, col: str = 'LastPrice') -> float:
    """
    Placeholder function to retrieve option prices.
    
    Parameters:
    asset_ticker (str): ticker name
    date (str): start date for price retrieval %Y-%m-%d 
    col (str): column name to retrieve the price from, default is 'LastPrice'
    Returns:
    float: asset price
    """
    
    databases = gen_date_list(start_date, end_date)
    full_prices = pd.DataFrame()
    for database in databases:
        file_path = options_data(database.replace('-', ''))
        prices = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
        if prices.empty: continue
        prices = prices[prices['Ticker'] == asset_ticker]
        full_prices = pd.concat([full_prices, prices], ignore_index=True)
    full_prices = full_prices[['TradeDate', col]]
    full_prices.rename(columns={col: 'Asset Price'}, inplace=True)
    return full_prices


def days_to_maturity(trade_date: List[str], maturity_date: List[str]):
    date_format = "%Y-%m-%d"
    trade_dates = [datetime.strptime(date, date_format) for date in trade_date]
    maturity_dates = [datetime.strptime(date, date_format) for date in maturity_date]
    return np.array([(maturity - trade).days / 365 for trade, maturity in zip(trade_dates, maturity_dates)])

def load_params(model: str, database: str, ticker) -> Dict:
    if not os.path.exists("calibrated_params.json"):
        return {}
    with open("calibrated_params.json", "r") as f:
        all_params = json.load(f)
        return all_params.get(model, {}).get(database, {}).get(ticker, {})
    
def save_params(model: str, database: str, ticker: str, params: Dict):
    existed_params = {}
    if os.path.exists("calibrated_params.json"):
        with open("calibrated_params.json", "r") as f:
            existed_params = json.load(f)
    if model not in existed_params:
        existed_params[model] = {}

    if database not in existed_params[model]:
        existed_params[model][database] = {}

    existed_params[model][database][ticker] = params

    with open("calibrated_params.json", "w") as f:
        json.dump(existed_params, f, indent=2)    