from scipy.optimize import minimize
from functools import partial
import numpy as np
from heston_model import heston_price
import pandas as pd
from utils import options_data, gen_date_list
from typing import List, Dict, Callable
from datetime import datetime
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
        prices = prices[prices['Ticker'].str.contains(asset_ticker)]
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

def listify_model(model: Callable, market_params: List[Dict], optmizing_params_keys: List[str]) -> Callable:
    def func(calibrating_params):
        named_calibrating_params = {key: value for key, value in zip(optmizing_params_keys, calibrating_params)}
        return [partial(model, **params)(**named_calibrating_params) for params in market_params]
    return func

def calibrate_heston_model():
    r = 0.10

    params = {
        "v0": {"x0": 0.1, "limits": [1e-3,0.1]},
        "kappa": {"x0": 3, "limits": [1e-3,5]},
        "theta": {"x0": 0.05, "limits": [1e-3,0.1]},
        "sigma": {"x0": 0.3, "limits": [1e-2,1]},
        "rho": {"x0": -0.8, "limits": [-1,0]},
        "lambd": {"x0": 0.03, "limits": [-1,1]},
    }
    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]
    options_b3 = get_option_data("VALE", "2020-09-03", "2020-09-10")
    asset_prices = get_asset_prices("VALE3", "2020-09-03", "2020-09-10")
    options_full_data = options_b3.join(asset_prices.set_index('Data Base'), on='Data Base')
    
    asset_prices = options_full_data['Asset Price'].values
    strikes = options_full_data['Strike'].values
    maturities = days_to_maturity(options_full_data['Data Base'].tolist(), options_full_data['Maturity Date'].tolist())
    prices = options_full_data['LastPrice'].values

    market_params = [{ 'S0': asset_price, 'K': strike, 'r': r, 'tau': maturitie } for asset_price, strike, maturitie in zip(asset_prices, strikes, maturities)]

    heston_model_listified = listify_model(heston_price, market_params, list(params.keys()))
    result = minimize(partial(squared_error, heston_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    print(result.x)
    
    print(heston_model_listified(*result.x))

if __name__ == "__main__":
    # options_b3=get_option_data("VALE", "2020-09-03", "2020-09-10")
    # asset_prices = get_asset_prices("VALE3", "2020-09-03", "2020-09-10")
    # print(asset_prices)
    # print(options_b3)
    # print(options_b3.join(asset_prices.set_index('Data Base'), on='Data Base'))
    
    calibrate_heston_model()

