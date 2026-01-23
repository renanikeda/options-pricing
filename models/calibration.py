from scipy.optimize import minimize
from functools import partial
import numpy as np
from heston_model import heston_price
from kou_jump_diffusion import kou_option_price
from utils import diff_days, nworkdays, measure, get_option_data, save_params, load_params, OptionType, get_selic
from black_scholes import black_scholes, implied_vol
from typing import List, Dict, Callable
import random
import pandas as pd

def squared_error(model: Callable, prices: List[float], params: Dict) -> float:
    """
    Calculate the squared error between model predictions and observed data.
    
    Parameters:
    model (function): the model function to generate predictions, receive params as input
        black-scholes: [S, K, tau, r, sigma, option_type]
        kou: [S0, K, r, sigma, tau, eta1, eta2, p, lambd, option_type]
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

def estimate_v0(options_data: pd.DataFrame, data_trade: str, spread_price: float = 0.5, min_trade_qty: int = 10, r: float = 0.1, option_type = OptionType.CALL, default_vol = 0.1) -> pd.DataFrame:
    filtered_data = options_data[options_data['TradeDate'] == data_trade]
    filtered_data = filtered_data[filtered_data['OscnPctg'] <= spread_price]
    filtered_data = filtered_data[filtered_data['TradeQty'] >= min_trade_qty]
    filtered_data['ATM'] = abs(filtered_data['Asset Price']/filtered_data['Strike'] - 1)

    # filtered_data = filtered_data.sort_values(by=['ATM', 'Days to Maturity'], ascending=[True, True])
    filtered_data = filtered_data.sort_values(by=['Days to Maturity', 'ATM'], ascending=[True, True])
    if filtered_data.empty:
        raise ValueError("No options meet the filtering criteria.")
    
    filtered_data = filtered_data.iloc[0]
    vol = implied_vol(filtered_data['LastPrice'], filtered_data['Asset Price'], filtered_data['Strike'], filtered_data['Days to Maturity'], r=r, option_type=option_type)
    vol = default_vol if vol is np.nan else vol
    return vol ** 2

def estimate_sigma(options_data_base: pd.DataFrame, data_trade: str, new_strike: float, new_maturity: str, spread_price: float = 0.5, min_trade_qty: int = 10, r: float = 0.1, option_type = OptionType.CALL, default_vol: float = 0.1) -> float:
    """
    Estimate implied volatility for a given strike and maturity by finding the closest match in the dataset.
    
    Parameters:
    options_data_base (pd.DataFrame): full options dataset
    data_trade (str): trade date to filter
    new_strike (float): target strike price
    new_maturity (str): target maturity date
    spread_price (float): maximum oscillation percentage
    min_trade_qty (int): minimum trade quantity
    r (float): risk-free rate
    option_type (OptionType): call or put option
    default_vol (float): default volatility if calculation fails
    
    Returns:
    float: estimated implied volatility
    """
    # Filter by trade date and quality criteria
    filtered_data = options_data_base[options_data_base['TradeDate'] == data_trade]
    filtered_data = filtered_data[filtered_data['OscnPctg'] <= spread_price]
    filtered_data = filtered_data[filtered_data['TradeQty'] >= min_trade_qty]
    
    if filtered_data.empty:
        raise ValueError("No options meet the filtering criteria.")
    
    filtered_data = filtered_data.copy()
    filtered_data['Diff_Strike'] = abs(filtered_data['Strike'] / new_strike - 1)
    filtered_data['Diff_Maturity'] = filtered_data['Maturity'].apply(lambda maturity: abs(diff_days(maturity, new_maturity)))
    
    # Normalize both differences to [0, 1] range for combined distance
    max_strike_diff = filtered_data['Diff_Strike'].max()
    max_maturity_diff = filtered_data['Diff_Maturity'].max()
    
    if max_strike_diff > 0:
        filtered_data['Norm_Strike'] = filtered_data['Diff_Strike'] / max_strike_diff
    else:
        filtered_data['Norm_Strike'] = 0
        
    if max_maturity_diff > 0:
        filtered_data['Norm_Maturity'] = filtered_data['Diff_Maturity'] / max_maturity_diff
    else:
        filtered_data['Norm_Maturity'] = 0
    
    filtered_data['Euclidean_Distance'] = np.sqrt(
        filtered_data['Norm_Strike']**2 + filtered_data['Norm_Maturity']**2
    )
    
    
    filtered_data = filtered_data.sort_values(by='Euclidean_Distance', ascending=True)
    closest_option = filtered_data.iloc[0]
    
    # Calculate implied volatility
    vol = implied_vol(
        closest_option['LastPrice'],
        closest_option['Asset Price'],
        closest_option['Strike'],
        closest_option['Days to Maturity'],
        r=r,
        option_type=option_type
    )
    
    # Return default if calculation fails
    # return default_vol if (vol is None or np.isnan(vol)) else vol
    return vol

def validate_heston_model(asset_ticker: str, database: str, _ndays: int = 5):
    params = load_params("heston", database, asset_ticker)
    database =  nworkdays(database, 2)
    data_end = nworkdays(database, _ndays)
    print('Dates: ', database, database)
    options_full_data = get_option_data(asset_ticker, database, data_end)
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100

    # market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'], 'v0': estimate_v0(options_full_data, row['TradeDate'], r=r) } for _, row in options_full_data.iterrows()]
    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print('Random Market params: ', random.sample(market_params, min(5, len(market_params))))

    listified_model = listify_model(heston_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'MSE Heston model on {database} to {data_end}: {sqr_err}')
    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(heston_price(**market_params[i], **params ), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))

def calibrate_heston_model(ticker: str, database: str = "2020-09-10", _ndays = 5):
    params = {
        "v0": {"x0": 0.1, "limits": [1e-3,0.5]},
        "kappa": {"x0": 0.5, "limits": [1e-3,5]},
        "theta": {"x0": 0.1, "limits": [1e-3,0.8]},
        "sigma": {"x0": 0.1, "limits": [1e-3,0.8]},
        "rho": {"x0": -0.2, "limits": [-1,1]},
    }

    data_ini = nworkdays(database, -1*_ndays)
    print('Dates: ', data_ini, database)
    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]

    options_full_data = get_option_data(ticker, data_ini, database)

    prices = options_full_data['LastPrice'].values
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    v0 = estimate_v0(options_full_data, options_full_data.iloc[0]['TradeDate'], r=r)
    params['v0']['x0'] = v0
    # market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'], 'v0': estimate_v0(options_full_data, row['TradeDate'], r=r) } for _, row in options_full_data.iterrows()]
    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print('Random Market params: ', random.sample(market_params, min(5, len(market_params))))

    heston_model_listified = listify_model(heston_price, market_params, list(params.keys()))
    result = minimize(partial(squared_error, heston_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    print("params: ", {**result_params})
    save_params("heston", database, ticker, result_params)


def validate_kou_model(asset_ticker: str, database: str, _ndays: int = 5):
    params = load_params("kou", database, asset_ticker)
    database =  nworkdays(database, 2)
    data_end = nworkdays(database, _ndays)
    options_full_data = get_option_data(asset_ticker, database, data_end)
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    listified_model = listify_model(kou_option_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'MSE Kou model on {database} to {data_end}: {sqr_err}')

    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(kou_option_price(**market_params[i], **params), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))

def calibrate_kou_model(asset_ticker: str, database: str = "2020-09-10", _ndays = 5):

    params = {
        "sigma": {"x0": 0.3, "limits": [1e-2,0.5]},
        "eta1": {"x0": 2, "limits": [1e-2,50]},
        "eta2": {"x0": 2, "limits": [1e-2,50]},
        "p": {"x0": 0.5, "limits": [1e-2,1]},
        "lambd": {"x0": 0.5, "limits": [1e-2,15]},
    }
    data_ini = nworkdays(database, -1*_ndays)
    print(data_ini, database)

    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]
    options_full_data = get_option_data(asset_ticker, data_ini, database)

    prices = options_full_data['LastPrice'].values
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    print('Random Market params: ', random.sample(market_params, min(5, len(market_params))))

    kou_model_listified = listify_model(kou_option_price, market_params, list(params.keys()))
    result = minimize(partial(squared_error, kou_model_listified, prices), initial_params, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    print("params: ", {**result_params})
    save_params("kou", database, asset_ticker, result_params)

def validate_black_scholes_model(asset_ticker: str, database: str = "2020-09-10", _ndays = 5):
    database =  nworkdays(database, 2)
    data_end = nworkdays(database, _ndays)
    options_full_data = get_option_data(asset_ticker, database, data_end)
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'], 'sigma': estimate_sigma(options_full_data, database, row['Strike'], row['Maturity']) } for _, row in options_full_data.iterrows()]
    print('Random params: ', random.sample(params, min(5, len(params))))
    listified_model = listify_model(black_scholes, params, [])
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, [])
    print(f'MSE Black-Scholes on {database} to {data_end}: {sqr_err}')

    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(black_scholes(**params[i]), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))

if __name__ == "__main__":
    # database = '2025-05-02'
    database = '2025-01-30'
    # database = '2020-10-16'
    ticker = "PETR4"
    validate_black_scholes_model(ticker, database, _ndays=5)
    measure(lambda: calibrate_heston_model(ticker, database, _ndays=5))
    validate_heston_model(ticker, database, _ndays=5)
    # measure(lambda: calibrate_kou_model(ticker, database, _ndays=5))
    # validate_kou_model(ticker, database, _ndays=5)

