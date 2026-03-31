from imp_vol import vol_surface, estimate_v0
from scipy.optimize import minimize
from functools import partial
import numpy as np
from heston_model import heston_price
from kou_jump_diffusion import kou_option_price
from utils import diff_days, estimate_sigma_hist, nworkdays, measure, get_option_data, save_params, load_params, OptionType, get_selic
from black_scholes import black_scholes, black_scholes_vega, implied_vol
from typing import List, Dict, Callable, Literal
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

    return np.sum((prices - model(params)) ** 2) + penality

def minimize_prices(model: Callable, market_params: List[Dict], optmizing_params_keys: List[str], optmizing_params_values: List[str]) -> float:
    err = 0
    eps = 1e-3
    for market_param in market_params:
        price = market_param['price']
        named_calibrating_params = {key: value for key, value in zip(optmizing_params_keys, optmizing_params_values)}
        model_price = model(**market_param, **named_calibrating_params)
        # weight = 1
        weight = 1/np.maximum(price, eps)**2
        # imp_vol = implied_vol(price, market_param['S0'], market_param['K'], market_param['tau'], r=market_param['r'])
        # imp_vol = eps if np.isnan(imp_vol) else imp_vol
        # weight = 1/np.maximum(black_scholes_vega(market_param['S0'], market_param['K'], sigma=imp_vol, tau=market_param['tau'], r=market_param['r']), eps)**2
        # err = np.append(err, weight*((np.log(price) - np.log(model_price)) ** 2))
        err += weight*((price - model_price) ** 2)
    return err

def minimize_imp_vol(model: Callable, market_params: List[Dict], optmizing_params_keys: List[str], optmizing_params_values: List[str]) -> float:
    err = np.array([])
    for market_param in market_params:
        market_imp_vol = implied_vol(market_param['price'], market_param['S0'], market_param['K'], market_param['tau'], r=market_param['r'])
        named_calibrating_params = {key: value for key, value in zip(optmizing_params_keys, optmizing_params_values)}
        model_imp_vol = implied_vol(model(**market_param, **named_calibrating_params), market_param['S0'], market_param['K'], market_param['tau'], r=market_param['r'])
        err = np.append(err, (market_imp_vol - model_imp_vol) ** 2)
    return np.sum(err)

def listify_model(model: Callable, market_params: List[Dict], optmizing_params_keys: List[str]) -> Callable:
    def func(calibrating_params):
        named_calibrating_params = {key: value for key, value in zip(optmizing_params_keys, calibrating_params)}
        return [model(**params, **named_calibrating_params) for params in market_params]
    return func


def estimate_sigma(asset_ticker: str, data_base: str, new_strike: float, new_maturity: str, spread_price: float = 0.75, min_trade_qty: int = 10, r: float = 0.1, option_type = OptionType.CALL, default_vol: float = 0.1) -> float:
    """
    Estimate implied volatility for a given strike and maturity by finding the closest match in the dataset.
    
    Parameters:
    options_data_base (pd.DataFrame): full options dataset
    data_base (str): trade date to filter
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
    options_data_base = get_option_data(asset_ticker, data_base, data_base)
    filtered_data = options_data_base[options_data_base['TradeDate'] == data_base]
    filtered_data = filtered_data[filtered_data['OscnPctg'] <= spread_price*100]
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
    return vol if vol is not np.nan else default_vol

def validate_heston_model(asset_ticker: str, database: str, _ndays: int = 5, moneyness_spread: float = 0.15):
    params = load_params("heston", database, asset_ticker, params_file=f'calibrated_params {moneyness_spread*100}%.json')
    print(params)
    data_start =  nworkdays(database, 2)
    data_end = nworkdays(data_start, _ndays)
    options_full_data = get_option_data(asset_ticker, data_start, data_end, moneyness_divergence=moneyness_spread)
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100

    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    # print('Random Market params: ', random.sample(market_params, min(5, len(market_params))))

    listified_model = listify_model(heston_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'MSE Heston model on {data_start} to {data_end}: {sqr_err}')
    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(heston_price(**market_params[i], **params ), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))
    return round(sqr_err, 8)

def calibrate_heston_model(ticker: str, database: str = "2020-09-10", _ndays = 5, moneyness_spread: float = 0.15):
    params = {
        # "v0": {"x0": 0.05, "limits": [1e-3,1]},
        "kappa": {"x0": 1, "limits": [1e-3,10]},
        "theta": {"x0": 0.1, "limits": [1e-3,1]},
        "sigma": {"x0": 0.1, "limits": [1e-3,2.5]},
        "rho": {"x0": -0.4, "limits": [-0.95,0.5]},
    }

    def feller_constraint(x):
        """Retorna valor >= 0 quando a condição é satisfeita."""
        kappa, theta, sigma = x[1], x[2], x[3]
        return 2 * kappa * theta - sigma**2 - 0.01

    constraints =  {'type': 'ineq', 'fun': feller_constraint}

    data_ini = nworkdays(database, -1*_ndays)
    print('Dates: ', data_ini, database)
    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]

    options_full_data = get_option_data(ticker, data_ini, database, moneyness_divergence=moneyness_spread)
    print(f'Moneyness spread: {moneyness_spread*100}%, Options data length: {len(options_full_data)}')
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    v0 = estimate_v0(options_full_data, options_full_data.iloc[0]['TradeDate'], r=r)
    market_params = [{ 'price': row['LastPrice'], 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'], 'v0': v0 } for _, row in options_full_data.iterrows()]
    # print('Random Market params: ', random.sample(market_params, min(5, len(market_params))))
    

    result = minimize(partial(minimize_prices, heston_price, market_params, params.keys()), initial_params, tol = 1e-4, method='SLSQP', options={'maxiter': 1e3 }, bounds=limit_params, constraints=constraints)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}
    result_params['v0'] = v0

    save_params("heston", database, ticker, result_params, params_file=f'calibrated_params {moneyness_spread*100}%.json')

def validate_kou_model(asset_ticker: str, database: str, _ndays: int = 5, moneyness_spread: float = 0.15):
    params = load_params("kou", database, asset_ticker, params_file=f'calibrated_params {moneyness_spread*100}%.json')
    print(params)
    database =  nworkdays(database, 2)
    data_end = nworkdays(database, _ndays)
    options_full_data = get_option_data(asset_ticker, database, data_end, moneyness_divergence=moneyness_spread)
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    market_params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    listified_model = listify_model(kou_option_price, market_params, list(params.keys()))
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, list(params.values()))
    print(f'MSE Kou model on {database} to {data_end}: {sqr_err}')

    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(kou_option_price(**market_params[i], **params), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))
    return round(sqr_err, 8)


def validate_model_imp_vol(asset_ticker: str, database: str, model: Literal['heston', 'kou'] = 'kou', moneyness_spread: float = 0.15):

    surface_market = vol_surface(asset_ticker, database, 'market', moneyness_divergence=moneyness_spread)
    maturity = surface_market['Maturity'].value_counts().sort_values(ascending=False).index[0]
    surface_model = vol_surface(asset_ticker, database, model, moneyness_divergence=moneyness_spread)
    surface_model = surface_model[surface_model['Maturity'] == maturity].sort_values(by='Strike')
    surface_market = surface_market[surface_market['Maturity'] == maturity].sort_values(by='Strike')
    
    vol_imp_market = surface_market['Implied Volatility'].values
    vol_imp_model = surface_model['Implied Volatility'].values

    #Mask filtering nan
    valid_mask = ~np.isnan(vol_imp_market)
    vol_imp_market = vol_imp_market[valid_mask]
    vol_imp_model = vol_imp_model[valid_mask]
    
    sqr_err = (1/len(vol_imp_market)) * np.sum((vol_imp_market - vol_imp_model) ** 2)
    print(f'MSE {model.capitalize()} model {model} implied vol on {database}: {sqr_err}')
    return round(sqr_err, 8)

def calibrate_kou_model(asset_ticker: str, database: str = "2020-09-10", _ndays = 5, moneyness_spread: float = 0.15):

    params = {
        "sigma": {"x0": 0.3, "limits": [1e-2,0.8]},
        "eta1": {"x0": 15, "limits": [1e-2,50]},
        "eta2": {"x0": 10, "limits": [1e-2,50]},
        "p": {"x0": 0.4, "limits": [1e-2,0.6]},
        "lambd": {"x0": 0.5, "limits": [1e-2,15]},
    }
    data_ini = nworkdays(database, -1*_ndays)
    print(data_ini, database)

    initial_params = [param["x0"] for key, param in params.items()]
    limit_params = [param["limits"] for key, param in params.items()]
    options_full_data = get_option_data(asset_ticker, data_ini, database, moneyness_divergence=moneyness_spread)
    print(f'Moneyness spread: {moneyness_spread*100}%, Options data length: {len(options_full_data)}')
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    market_params = [{ 'price': row['LastPrice'], 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] } for _, row in options_full_data.iterrows()]
    # print('Random Market params: ', random.sample(market_params, min(5, len(market_params))))

    result = minimize(partial(minimize_prices, kou_option_price, market_params, params.keys()), initial_params, tol = 1e-4, method='L-BFGS-B', options={'maxiter': 1e3 }, bounds=limit_params)
    result_params = {key: value for key, value in zip(params.keys(), result.x)}

    # print("params: ", {**result_params})
    save_params("kou", database, asset_ticker, result_params, params_file=f'calibrated_params {moneyness_spread*100}%.json')

def validate_black_scholes_model_imp_vol(asset_ticker: str, database: str = "2020-09-10", _ndays = 5):
    data_start =  nworkdays(database, 2)
    data_end = nworkdays(data_start, _ndays)
    print('Dates: ', data_start, data_end)
    options_full_data = get_option_data(asset_ticker, data_start, data_end)
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'], 'sigma': estimate_sigma(asset_ticker, database, new_strike=row['Strike'], new_maturity=row['Maturity'], r=r) } for _, row in options_full_data.iterrows()]
    # print('Random params: ', random.sample(params, min(5, len(params))))
    listified_model = listify_model(black_scholes, params, [])
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, [])
    print(f'MSE Black-Scholes Imp vol on {data_start} to {data_end}: {sqr_err}')

    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(black_scholes(**params[i]), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))
    return round(sqr_err, 8)

def validate_black_scholes_model_hist_vol(asset_ticker: str, database: str = "2020-09-10", _ndays = 5):
    data_start =  nworkdays(database, 2)
    data_end = nworkdays(data_start, _ndays)
    print('Dates: ', data_start, data_end)
    options_full_data = get_option_data(asset_ticker, data_start, data_end)
    r = get_selic(options_full_data.iloc[0]['TradeDate']) / 100
    sigma_hist = estimate_sigma_hist(asset_ticker, database, _ndays=(_ndays+2)).values.flatten()[-1]
    print('Estimated historical volatility: ', sigma_hist)
    params = [{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'], 'sigma': sigma_hist } for _, row in options_full_data.iterrows()]

    listified_model = listify_model(black_scholes, params, [])
    sqr_err = (1/len(options_full_data)) * squared_error(listified_model, options_full_data['LastPrice'].values, [])
    print(f'MSE Black-Scholes hist vol on {data_start} to {data_end}: {sqr_err}')

    for i in random.sample(range(len(options_full_data)), 5):
        print('Estimates price: ', round(black_scholes(**params[i]), 2))
        print('Real price: ', round(options_full_data['LastPrice'].iloc[i], 2))
    return round(sqr_err, 8)

def results_to_csv():
    results = []
    for ticker in ['PETR4', 'VALE3', 'BOVA11']:
    # for ticker in ['VALE3']:
        for database in ['2020-07-13', '2022-04-18', '2025-06-10']:
        # for database in ['2020-07-13', '2022-04-18']:
            for moneyness_spread in [0.15, 0.2, 0.5, 0.6, 0.7]:
            # for moneyness_spread in [0.15, 0.6]:
                _ndays=5
                sqr_model_heston = validate_heston_model(ticker, database, _ndays=_ndays, moneyness_spread=moneyness_spread) 
                sqr_model_kou = validate_kou_model(ticker, database, _ndays=_ndays, moneyness_spread=moneyness_spread)
                sqr_bs_hist_vol = validate_black_scholes_model_hist_vol(ticker, database, _ndays=_ndays)
                sqr_bs_imp_vol = validate_black_scholes_model_imp_vol(ticker, database, _ndays=_ndays)
                sqr_model_heston_imp_vol = validate_model_imp_vol(ticker, database, model='heston', moneyness_spread=moneyness_spread)
                sqr_model_kou_imp_vol = validate_model_imp_vol(ticker, database, model='kou', moneyness_spread=moneyness_spread)
                results.append({
                    'ticker': ticker,
                    'database': database,
                    'sqr_bs_hist_vol': sqr_bs_hist_vol,
                    'sqr_bs_imp_vol': sqr_bs_imp_vol,
                    'sqr_model_heston': sqr_model_heston,
                    'sqr_model_kou': sqr_model_kou,
                    'sqr_model_heston_imp_vol': sqr_model_heston_imp_vol,
                    'sqr_model_kou_imp_vol': sqr_model_kou_imp_vol,
                    'moneyness': f'{moneyness_spread*100}%',
                })
    df_results = pd.DataFrame(results)
    df_results.to_csv('model_validation_results.csv', index=False)
            

if __name__ == "__main__":
    # database = '2023-11-01'
    # database = '2025-01-30'
    # database = '2020-10-16'
    # ticker = "PETR4"
    # ticker = "VALE3"
    # results_to_csv()
    # raise Exception

    # get_option_data('PETR4', '2020-07-07', '2020-07-13', moneyness_divergence=0.7)
    for database in ['2020-07-13', '2021-04-20', '2022-04-18', '2023-11-01', '2025-01-30', '2025-06-10']:
    # for database in ['2025-01-30', '2025-06-10']:
        for moneyness_spread in [0.15, 0.2, 0.5, 0.6, 0.7]:
        # for moneyness_spread in [0.2, 0.6]:
            # for ticker in ['PETR4', 'VALE3', 'BOVA11']:
            for ticker in ['BOVA11']:
            # for ticker in ['BOVA11']:
                # validate_black_scholes_model_hist_vol(ticker, database, _ndays=5)
                # validate_black_scholes_model_imp_vol(ticker, database, _ndays=5)
                measure(lambda: calibrate_heston_model(ticker, database, _ndays=5, moneyness_spread=moneyness_spread))
                # validate_heston_model(ticker, database, _ndays=5)
                measure(lambda: calibrate_kou_model(ticker, database, _ndays=5, moneyness_spread=moneyness_spread))
                # validate_kou_model(ticker, database, _ndays=5)
            
                # validate_model_imp_vol(ticker, '2020-07-13', model='kou')
                # validate_model_imp_vol(ticker, '2020-07-13', model='heston')
