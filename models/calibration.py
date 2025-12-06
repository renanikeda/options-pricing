from scipy.optimize import minimize
from functools import partial
import numpy as np
from heston_model import heston_price
import pandas as pd


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
    
    prices = pd.read_csv("../market options/interested_merged_deals.csv", delimiter=",")
    prices = prices[prices['Ticker'].str.contains(asset_ticker, na=False) & (prices['TradeDate'] >= start_date) & (prices['TradeDate'] <= end_date)]
    return prices['price']

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

    prices= 10.5  # Example observed prices
    heston_price_partial = partial(heston_price, S0, K, r=r, tau=tau)
    result = minimize(partial(squared_error, heston_price_partial, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    print(result.x)
    
    print(heston_price_partial(*result.x))

if __name__ == "__main__":
    calibrate_heston_model()