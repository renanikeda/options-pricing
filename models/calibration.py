from scipy.optimize import minimize
from functools import partial
import numpy as np
from heston_model import heston_price
import pandas as pd
from utils import options_data, gen_date_list
from typing import List
from datetime import datetime


def squared_error(model, prices, params):
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
    return np.sum((model(*params) - prices) ** 2) + penality


def get_option_prices(asset_ticker: str, start_date: str, end_date: str):
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
        prices = pd.read_csv(options_data(database.replace('-', '')))
        prices = prices[prices['Ticker'].str.contains(asset_ticker)]
        full_prices = pd.concat([full_prices, prices], ignore_index=True)

    return full_prices

def days_to_maturity(trade_date: List[str], maturity_date: List[str]):
    date_format = "%Y-%m-%d"
    trade_dates = [datetime.strptime(date, date_format) for date in trade_date]
    maturity_dates = [datetime.strptime(date, date_format) for date in maturity_date]
    return np.array([(maturity - trade).days / 365 for trade, maturity in zip(trade_dates, maturity_dates)])

def calibrate_heston_model():
    S0 = 100
    K = 100
    r = 0.07
    tau = 1.0

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
    options_b3 = get_option_prices("VALE", "2020-09-03", "2020-09-10")
    S0 = options_b3[options_b3['Ticker'] == 'VALE3']['LastPrice'].iloc[0]
    
    options_b3.dropna(subset=['Strike'], inplace=True)
    prices = options_b3['LastPrice'].values
    strikes = options_b3['Strike'].values
    maturities = days_to_maturity(options_b3['Data Base'].tolist(), options_b3['Maturity Date'].tolist())
    # prices= 10.5  # Example observed prices
    heston_price_partial = partial(heston_price, S0, K, r=r, tau=tau)
    result = minimize(partial(squared_error, heston_price_partial, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    print(result.x)
    
    print(heston_price_partial(*result.x))

if __name__ == "__main__":
    # calibrate_heston_model()
    options_b3=get_option_prices("VALE", "2020-09-03", "2020-09-03")
    print(options_b3[options_b3['Ticker'] == 'VALE3']['LastPrice'].iloc[0])
    options_b3.dropna(subset=['Strike'], inplace=True)
    print(options_b3)
    print(days_to_maturity(options_b3['Data Base'].tolist(), options_b3['Maturity Date'].tolist()))