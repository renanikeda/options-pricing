import os
import time
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
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

# colors = ['black', 'red', 'green', 'blue', 'olive', 'purple', 'orange', 'brown', 'pink', 'gray']
colors = ['darkred', 'darkgoldenrod', 'olive', 'darkcyan', 'indigo', 'darkmagenta', 'saddlebrown', 'teal', 'slategray', 'darkgreen']
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


def gen_valid_date_list(ini_date: str, ndays: int, return_format: str = '%Y%m%d'):
    start_date = datetime.strptime(ini_date, '%Y-%m-%d')
    delta = timedelta(days=1) if ndays >= 0 else timedelta(days=-1)
    counter = 0
    date_list = []
    while counter < abs(ndays):
        if os.path.exists(options_data(database=start_date.strftime('%Y%m%d'))):
            date_list.append(start_date.strftime(return_format))
            counter += 1
        start_date += delta

    return date_list

def nworkdays(database:str, ndays: int):
    date_list = gen_valid_date_list(database, ndays, '%Y-%m-%d')
    return date_list[-1]

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
    type_value = depara_option_type(type)
    full_prices = full_prices[full_prices['Type'] == type_value]
    full_prices = full_prices[full_prices['Style'] == style.value]
    return full_prices

def depara_option_type(type: OptionType) -> str:
    return "CALL" if type == OptionType.CALL else "PUTT"

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

def load_params(model: str, database: str, ticker: str) -> Dict:
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

def ewma_volatility(prices: pd.Series, window: int = None, span: int = None, 
                   halflife: int = None, alpha: float = None, 
                   annualize: bool = True, trading_days: int = 252) -> pd.DataFrame:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) volatility.
    
    EWMA gives more weight to recent observations, making it more responsive
    to recent market changes compared to simple moving average.
    
    Parameters:
    prices (pd.Series): Time series of asset prices
    window (int): Size of the moving window (for initial calculation)
    span (int): Specify decay in terms of span (alternative to alpha)
                alpha = 2 / (span + 1)
    halflife (int): Specify decay in terms of half-life (alternative to alpha)
                    alpha = 1 - exp(log(0.5) / halflife)
    alpha (float): Smoothing factor, 0 < alpha <= 1
                   Higher alpha = more weight to recent observations
    annualize (bool): If True, annualize the volatility (default: True)
    trading_days (int): Number of trading days per year (default: 252)
    
    Returns:
    pd.DataFrame: EWMA volatility (standard deviation)
    
    Examples:
    >>> prices = get_asset_prices('PETR4', '2024-01-01', '2024-12-31')
    >>> vol_span = ewma_volatility(prices['Asset Price'], span=30)
    >>> vol_halflife = ewma_volatility(prices['Asset Price'], halflife=21)
    >>> vol_alpha = ewma_volatility(prices['Asset Price'], alpha=0.94)
    """
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calculate EWMA variance
    if alpha is not None:
        ewma_var = log_returns.ewm(alpha=alpha, adjust=False).var()
    elif span is not None:
        ewma_var = log_returns.ewm(span=span, adjust=False).var()
    elif halflife is not None:
        ewma_var = log_returns.ewm(halflife=halflife, adjust=False).var()
    elif window is not None:
        ewma_var = log_returns.ewm(span=window, adjust=False).var()
    else:
        # Default: span=30 (approximately 1 month)
        ewma_var = log_returns.ewm(span=30, adjust=False).var()
    
    # Calculate standard deviation
    ewma_std = np.sqrt(ewma_var)
    
    # Annualize if requested
    if annualize:
        ewma_std = ewma_std * np.sqrt(trading_days)
    
    return pd.DataFrame({
        'Vol EWMA': ewma_std
    })

def get_asset_volatility(start_date: str, end_date: str, asset_ticker: Optional[str], asset_frame: Optional[pd.DataFrame] = None,
                        window: int = 30, alpha: float = 0.94, **kwargs) -> pd.Series:
    """
    Get asset volatility over a date range.
    
    Parameters:
    asset_ticker (str): ticker name
    start_date (str): start date %Y-%m-%d
    end_date (str): end date %Y-%m-%d
    window (int): size of the moving window
    method (str): volatility calculation method ('ewma', 'simple', 'garch')
    **kwargs: additional parameters for volatility calculation
    
    Returns:
    pd.Series: Asset volatility time series
    
    Example:
    >>> vol = get_asset_volatility('PETR4', '2024-01-01', '2024-12-31', 
    ...                            window=30, method='ewma', span=30)
    """
    # Get price data
    prices_df = asset_frame.copy() if asset_frame else get_asset_prices(asset_ticker, start_date, end_date) 
    
    if prices_df.empty:
        return pd.Series()
    
    # Set date as index
    prices_df['TradeDate'] = pd.to_datetime(prices_df['TradeDate'])
    prices_df.set_index('TradeDate', inplace=True)
    prices = prices_df['Asset Price']
    
    # Calculate volatility
    
    volatility = ewma_volatility(prices, window=window, alpha=alpha, **kwargs)
    
    return volatility

