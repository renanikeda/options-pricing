
import pandas as pd
import numpy as np
from typing import Literal

from black_scholes import implied_vol
from heston_model import heston_model, heston_price
from kou_jump_diffusion import kou_option_price, kou_process
from utils import OptionType, get_option_data, get_selic, load_params

def estimate_v0(options_data: pd.DataFrame, data_trade: str, r: float = 0.1, option_type = OptionType.CALL, default_vol = 0.1) -> pd.DataFrame:
    filtered_data = options_data[options_data['TradeDate'] == data_trade].copy()
    filtered_data['ATM'] = abs(filtered_data['Asset Price']/filtered_data['Strike'] - 1)

    # filtered_data = filtered_data.sort_values(by=['ATM', 'Days to Maturity'], ascending=[True, True])
    filtered_data = filtered_data.sort_values(by=['Days to Maturity', 'ATM'], ascending=[True, True])
    if filtered_data.empty:
        raise ValueError("No options meet the filtering criteria.")
    
    filtered_data = filtered_data.iloc[0]
    vol = implied_vol(filtered_data['LastPrice'], filtered_data['Asset Price'], filtered_data['Strike'], filtered_data['Days to Maturity'], r=r, option_type=option_type)
    vol = default_vol if vol is np.nan else vol
    return vol ** 2

def vol_surface(ticker: str, database: str, type: Literal['heston', 'kou', 'market'] = 'market', moneyness_divergence: float = 0.15, verbose=False):
    options_data = get_option_data(ticker, database, database, moneyness_divergence=moneyness_divergence)
    options_data = options_data[options_data['Asset Ticker'] == ticker]
    imp_vols = []
    r = get_selic(options_data.iloc[0]['TradeDate']) / 100
    for (_, row) in options_data.iterrows():
        if type == 'heston':
            params = load_params("heston", database, ticker, params_file=f'calibrated_params {moneyness_divergence*100}%.json')
            if 'v0' not in params:
                params['v0'] = estimate_v0(options_data, row['TradeDate'], r=r)
            price = heston_price(**{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] }, **params)
            if verbose: print(f"Heston price: {price:.2f}, Market price: {row['LastPrice']}")
        elif type == 'kou':
            params = load_params("kou", database, ticker, params_file=f'calibrated_params {moneyness_divergence*100}%.json')
            price = kou_option_price(**{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] }, **params)
            if verbose: print(f"Kou price: {price:.2f}, Market price: {row['LastPrice']}")
        elif type == 'market':
            price = row['LastPrice']
            if verbose: print(f"Black-Scholes, Market price: {row['LastPrice']}")
        imp_vol = implied_vol(price, row['Asset Price'], row['Strike'], row['Days to Maturity'], r=r, option_type=OptionType.CALL if row.Type == "CALL" else OptionType.PUT)
        imp_vols.append(imp_vol)
    vol_surface = pd.DataFrame()
    vol_surface['Strike'] = options_data['Strike']
    # vol_surface['LastPrice'] = options_data['LastPrice']
    # vol_surface['Asset Price'] = options_data['Asset Price']
    vol_surface['Maturity'] = options_data['Maturity']    
    vol_surface['Days to Maturity'] = options_data['Days to Maturity']    
    vol_surface['Implied Volatility'] = imp_vols
    vol_surface['moneyness'] = options_data['moneyness']
    return vol_surface