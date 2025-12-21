from scipy.optimize import minimize
from functools import partial
import numpy as np
from heston_model import heston_price, heston_price_stable
from kou_jump_diffusion import kou_option_price
import pandas as pd
from utils import options_data, gen_date_list, classify_option , OptionType, ndays, measure, get_prefixo_ticker
from typing import List, Dict, Callable
from datetime import datetime
import random
import json
import os

def squared_error(model, prices: List[float], params):
    """
    Calculate the squared error between model predictions and observed data.
    
    Parameters:
    model (function): the model function to generate predictions, receive params as input
        black-scholes: [S, K, T, r, sigma, option_type]
        kou: [S0, K, r, sigma, T, eta1, eta2, p, lambd, option_type]
        heston: [S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r]
    prices (np.ndarray): observed data points
    params (np.ndarray): parameters for the model function
    
    Returns:
    float: squared error
    """
    penality = 0
    ## Fazer isso para cada maturidade, strike e taxa
    # print(model(params)[:5])
    # print(prices[:5])
    return np.sum((model(params) - prices) ** 2) + penality


def get_option_data(asset_ticker: str, start_date: str, end_date: str):
    """
    Placeholder function to retrieve option prices.
    
    Parameters:
    ticker (str): option ticker symbol
    start_date (str): start date for data retrieval %Y-%m-%d
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
        full_prices = pd.concat([full_prices, prices], ignore_index=True)
    full_prices.dropna(subset=['Strike'], inplace=True)
    full_prices['Days to Maturity'] = days_to_maturity(full_prices['Data Base'].tolist(), full_prices['Maturity Date'].tolist())
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
    full_prices = full_prices[['Data Base', col]]
    full_prices.rename(columns={col: 'Asset Price'}, inplace=True)
    return full_prices


def days_to_maturity(trade_date: List[str], maturity_date: List[str]):
    date_format = "%Y-%m-%d"
    trade_dates = [datetime.strptime(date, date_format) for date in trade_date]
    maturity_dates = [datetime.strptime(date, date_format) for date in maturity_date]
    return np.array([(maturity - trade).days / 365 for trade, maturity in zip(trade_dates, maturity_dates)])

def filter_calls(pricing_table: pd.DataFrame) -> pd.DataFrame:
    return pricing_table[pricing_table['Ticker'].apply(classify_option) == OptionType.CALL]

def listify_model(model: Callable, market_params: List[Dict], optmizing_params_keys: List[str]) -> Callable:
    def func(calibrating_params):
        named_calibrating_params = {key: value for key, value in zip(optmizing_params_keys, calibrating_params)}
        return [partial(model, **params)(**named_calibrating_params) for params in market_params]
    return func

def load_params(model: str, database: str) -> Dict:
    if not os.path.exists("calibrated_params.json"):
        return {}
    with open("calibrated_params.json", "r") as f:
        all_params = json.load(f)
        return all_params.get(model, {}).get(database, {})
    
def save_params(model: str, database: str, params: Dict):
    existed_params = {}
    if os.path.exists("calibrated_params.json"):
        with open("calibrated_params.json", "r") as f:
            existed_params = json.load(f)
    if model not in existed_params:
        existed_params[model] = {}
    existed_params[model][database] = params

    with open("calibrated_params.json", "w") as f:
        json.dump(existed_params, f, indent=2)    


def validate_heston_model(database: str, _ndays: int = 5):
    params = load_params("heston", database)
    data_end = ndays(database, _ndays)
    options_b3 = get_option_data("VALE", database, data_end)
    asset_prices = get_asset_prices("VALE3", database, data_end)
    options_full_data = filter_calls(options_b3.join(asset_prices.set_index('Data Base'), on='Data Base'))

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': 0.10, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    listified_model = listify_model(heston_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'Squared error for Heston model on {database} to {data_end}\nwith params {params}\nMSE: {sqr_err}')
    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(heston_price(**market_params[i], **params), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))

def calibrate_heston_model(database: str = "2020-09-10", _ndays = 5):
    r = 0.10

    params = {
        "v0": {"x0": 0.1, "limits": [1e-3,0.5]},
        "kappa": {"x0": 3, "limits": [1e-3,5]},
        "theta": {"x0": 0.05, "limits": [1e-3,0.5]},
        "sigma": {"x0": 0.3, "limits": [1e-2,0.5]},
        "rho": {"x0": -0.8, "limits": [-1,1]},
        "lambd": {"x0": 0.03, "limits": [-1,1]},
    }
    data_ini = ndays(database, -1*_ndays)
    print(data_ini, database)
    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]
    options_b3 = get_option_data("VALE", data_ini, database)
    asset_prices = get_asset_prices("VALE3", data_ini, database)
    options_full_data = filter_calls(options_b3.join(asset_prices.set_index('Data Base'), on='Data Base'))

    prices = options_full_data['LastPrice'].values

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print(market_params[:5])

    heston_model_listified = listify_model(heston_price_stable, market_params, list(params.keys()))
    result = minimize(partial(squared_error, heston_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    print("params: ", {**result_params})
    save_params("heston", database, result_params)

def calibrate_heston_model(ticker: str, database: str = "2020-09-10", _ndays = 5):
    r = 0.10

    params = {
        "v0": {"x0": 0.1, "limits": [1e-3,0.5]},
        "kappa": {"x0": 3, "limits": [1e-3,5]},
        "theta": {"x0": 0.05, "limits": [1e-3,0.5]},
        "sigma": {"x0": 0.3, "limits": [1e-2,0.5]},
        "rho": {"x0": -0.8, "limits": [-1,1]},
        "lambd": {"x0": 0.03, "limits": [-1,1]},
    }
    data_ini = ndays(database, -1*_ndays)
    print('Dates: ', data_ini, database)
    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]

    options_b3 = get_option_data(ticker, data_ini, database)
    asset_prices = get_asset_prices(ticker, data_ini, database)
    options_full_data = filter_calls(options_b3.join(asset_prices.set_index('Data Base'), on='Data Base'))

    prices = options_full_data['LastPrice'].values

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print('Market params: ', market_params[:5])

    heston_model_listified = listify_model(heston_price, market_params, list(params.keys()))
    result = minimize(partial(squared_error, heston_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    print("params: ", {**result_params})
    save_params("heston", database, result_params)


def validate_kou_model(database: str, _ndays: int = 5):
    params = load_params("kou", database)
    data_end = ndays(database, _ndays)
    options_b3 = get_option_data("VALE", database, data_end)
    asset_prices = get_asset_prices("VALE3", database, data_end)
    options_full_data = filter_calls(options_b3.join(asset_prices.set_index('Data Base'), on='Data Base'))

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': 0.10, 'T': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    listified_model = listify_model(kou_option_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'Squared error for Kou model on {database} to {data_end}\nwith params {params}\nMSE: {sqr_err}')

    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(kou_option_price(**market_params[i], **params), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))

def calibrate_kou_model(database: str = "2020-09-10", _ndays = 5):
    r = 0.10

    params = {
        "sigma": {"x0": 0.3, "limits": [1e-2,0.5]},
        "eta1": {"x0": 0.5, "limits": [1e-2,50]},
        "eta2": {"x0": 0.5, "limits": [1e-2,50]},
        "p": {"x0": 0.5, "limits": [1e-2,1]},
        "lambd": {"x0": 0.5, "limits": [1e-2,15]},
    }
    data_ini = ndays(database, -1*_ndays)
    print(data_ini, database)

    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]
    options_b3 = get_option_data("VALE", data_ini, database)
    asset_prices = get_asset_prices("VALE3", data_ini, database)
    options_full_data = filter_calls(options_b3.join(asset_prices.set_index('Data Base'), on='Data Base'))

    prices = options_full_data['LastPrice'].values

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'T': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print(market_params[:5])

    kou_model_listified = listify_model(kou_option_price, market_params, list(params.keys()))
    result = minimize(partial(squared_error, kou_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    print("params: ", {**result_params})
    save_params("kou", database, result_params)

if __name__ == "__main__":
    database = '2025-05-01'
    measure(lambda: calibrate_heston_model('VALE3', database, _ndays=7))
    validate_heston_model(database, _ndays=5)
    # measure(lambda: calibrate_kou_model(database, _ndays=7))
    # validate_kou_model(database, _ndays=7)

