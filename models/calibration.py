from scipy.optimize import minimize
from functools import partial
import numpy as np
from heston_model import heston_price
from kou_jump_diffusion import kou_option_price
from utils import ndays, measure, get_option_data, save_params, load_params
from typing import List, Dict, Callable
import random

def squared_error(model, prices: List[float], params):
    """
    Calculate the squared error between model predictions and observed data.
    
    Parameters:
    model (function): the model function to generate predictions, receive params as input
        black-scholes: [S, K, T, r, sigma, option_type]
        kou: [S0, K, r, sigma, T, eta1, eta2, p, lambd, option_type]
        heston: [S0, K, v0, kappa, theta, sigma, rho, tau, r]
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

def listify_model(model: Callable, market_params: List[Dict], optmizing_params_keys: List[str]) -> Callable:
    def func(calibrating_params):
        named_calibrating_params = {key: value for key, value in zip(optmizing_params_keys, calibrating_params)}
        return [partial(model, **params)(**named_calibrating_params) for params in market_params]
    return func

def validate_heston_model(asset_ticker: str, database: str, _ndays: int = 5):
    params = load_params("heston", database)
    data_end = ndays(database, _ndays)
    options_full_data = get_option_data(asset_ticker, database, data_end)
    
    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': 0.10, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    listified_model = listify_model(heston_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'Squared error for Heston model on {database} to {data_end}\nwith params {params}\nMSE: {sqr_err}')
    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(heston_price(**market_params[i], **params), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))

def calibrate_heston_model(ticker: str, database: str = "2020-09-10", _ndays = 5):
    r = 0.10

    params = {
        "v0": {"x0": 0.1, "limits": [1e-3,0.5]},
        "kappa": {"x0": 0.5, "limits": [1e-3,3]},
        "theta": {"x0": 0.05, "limits": [1e-3,0.5]},
        "sigma": {"x0": 0.3, "limits": [1e-2,0.5]},
        "rho": {"x0": -0.8, "limits": [-1,1]},
    }

    data_ini = ndays(database, -1*_ndays)
    print('Dates: ', data_ini, database)
    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]

    options_full_data = get_option_data(ticker, data_ini, database)

    prices = options_full_data['LastPrice'].values

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print('Market params: ', market_params[:5])

    heston_model_listified = listify_model(heston_price, market_params, list(params.keys()))
    result = minimize(partial(squared_error, heston_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    print("params: ", {**result_params})
    save_params("heston", database, result_params)


def validate_kou_model(asset_ticker: str, database: str, _ndays: int = 5):
    params = load_params("kou", database)
    data_end = ndays(database, _ndays)
    options_full_data = get_option_data(asset_ticker, database, data_end)

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': 0.10, 'T': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    listified_model = listify_model(kou_option_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'Squared error for Kou model on {database} to {data_end}\nwith params {params}\nMSE: {sqr_err}')

    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(kou_option_price(**market_params[i], **params), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))

def calibrate_kou_model(asset_ticker: str, database: str = "2020-09-10", _ndays = 5):
    r = 0.10

    params = {
        "sigma": {"x0": 0.3, "limits": [1e-2,0.5]},
        "eta1": {"x0": 2, "limits": [1e-2,50]},
        "eta2": {"x0": 2, "limits": [1e-2,50]},
        "p": {"x0": 0.5, "limits": [1e-2,1]},
        "lambd": {"x0": 0.5, "limits": [1e-2,15]},
    }
    data_ini = ndays(database, -1*_ndays)
    print(data_ini, database)

    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]
    options_full_data = get_option_data(asset_ticker, data_ini, database)

    prices = options_full_data['LastPrice'].values

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'T': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print(market_params[:5])

    kou_model_listified = listify_model(kou_option_price, market_params, list(params.keys()))
    result = minimize(partial(squared_error, kou_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    print("params: ", {**result_params})
    save_params("kou", database, result_params)

if __name__ == "__main__":
    database = '2025-05-02'
    ticker = "VALE3"
    # print(get_option_data(ticker, database, ndays(database, 7)))
    # measure(lambda: calibrate_heston_model(ticker, database, _ndays=7))
    validate_heston_model(ticker, database, _ndays=1)
    # measure(lambda: calibrate_kou_model(ticker, database, _ndays=7))
    validate_kou_model(ticker, database, _ndays=7)

