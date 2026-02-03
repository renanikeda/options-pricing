
from brownian_motion import geometric_brownian_motion
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from typing import Literal

from black_scholes import implied_vol
from heston_model import heston_model, heston_price
from kou_jump_diffusion import kou_option_price, kou_process
from calibration import estimate_v0
from utils import OptionType, ewma_volatility, get_option_data, get_asset_prices, get_selic, load_params, nworkdays, estimate_sigma_hist, OptionStyle

def vol_surface(ticker: str, database: str, type: Literal['heston', 'kou', 'market'] = 'market', verbose=False):
    options_data = get_option_data(ticker, database, database)
    options_data = options_data[options_data['Asset Ticker'] == ticker]
    imp_vols = []
    r = get_selic(options_data.iloc[0]['TradeDate']) / 100
    for (_, row) in options_data.iterrows():
        if type == 'heston':
            params = load_params("heston", database, ticker)
            if 'v0' not in params:
                params['v0'] = estimate_v0(options_data, row['TradeDate'], r=r)
            price = heston_price(**{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] }, **params)
            if verbose: print(f"Heston price: {price:.2f}, Market price: {row['LastPrice']}")
        elif type == 'kou':
            params = load_params("kou", database, ticker)
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

def plot_returns(W: np.array, bins: int = 100) -> None:
    """
    Plot the returns distribution and QQ plot.
    Args:
        W (np.ndarray): Simulated stock prices matrix.
    """
    
    returns = np.diff(np.log(W), axis=0)
    flat_returns = returns.flatten()

    print('ploting returns shape:', flat_returns.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(211)
    xlim = np.std(flat_returns) * 4
    ax.set_xlim([-xlim, xlim])
    sns.histplot(flat_returns, bins=bins, color='darkolivegreen', stat='density', ax=ax)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, bins*2)
    p = norm.pdf(x, np.mean(flat_returns), np.std(flat_returns))
    plt.plot(x, p, color='r', linestyle='dashed', linewidth=3, label='Gaussian')
    plt.xlabel(f'GBM retorno diário')
    plt.ylabel('Frequência')
    plt.grid(linewidth=0.3)
    plt.legend()

    ax = fig.add_subplot(212)
    qqplot(flat_returns, line='s', ax=ax, markerfacecolor='darkolivegreen', fit=True)
    plt.xlabel('Quantil Teórico')
    plt.ylabel('Quantil Amostral')
    plt.xlim([-4.5,4.5])
    plt.grid(linewidth=0.3)
    plt.tight_layout()
    plt.show()

def filter_recent_maturity(options_df: pd.DataFrame, min_trade_qty=5, spread_price=0.75) -> pd.Timestamp:
    """
    Filter and retrieve the most recent maturity date from the options DataFrame.
    
    Parameters:
    options_df (pd.DataFrame): DataFrame containing option data with 'Maturity' column.
    min_trade_qty (int): Minimum trade quantity to filter options.
    spread_price (float): Maximum allowed spread price percentage (decimal format).
    
    Returns:
    pd.Timestamp: The most recent maturity date.
    """
    resultado = options_df[options_df["TradeQty"] >= min_trade_qty]
    resultado = resultado[resultado["OscnPctg"].abs() <= abs(spread_price * 100)]

    resultado = (
        resultado
        .loc[lambda x: x.groupby("TradeDate")["Days to Maturity"].idxmin()]
        .sort_values("TradeDate")
    )
    return resultado

def estimate_vol(asset_ticker: str, trade_date: str, r = 0.1, min_trade_qty: int = 5, spread_price: float = 0.75, option_type: OptionType = OptionType.CALL):
    """
    Get the implied volatility for the most recent maturity option on a given trade date.
    
    Parameters:
    asset_ticker (str): ticker name
    trade_date (str): trade date %Y-%m-%d
    r (float): risk-free interest rate
    option_type (OptionType): option type
    
    Returns:
    float: implied volatility
    """
    options_df = get_option_data(asset_ticker, trade_date, trade_date, type=option_type)
    if options_df.empty:
        return None
    
    prices_df = get_asset_prices(asset_ticker, trade_date, trade_date)
    asset_price = prices_df['Asset Price'].values[0]
    
    recent_maturity = filter_recent_maturity(options_df, min_trade_qty, spread_price)
    vol_surf = vol_surface(asset_ticker, trade_date, type='black')
    vol_surf = vol_surf.loc[vol_surf['Maturity'] == recent_maturity.iloc[0]['Maturity']]
    vol_surf['Price Difference'] = abs(vol_surf['Strike'] - asset_price)
    vol_surf = vol_surf.loc[vol_surf['Price Difference'].idxmin()]
    iv_latest = vol_surf['Implied Volatility']
    
    return iv_latest

def get_option_implied_vs_realized_vol(asset_ticker: str, start_date: str, 
                                      end_date: str, style: OptionStyle = OptionStyle.EURO,
                                      type: OptionType = OptionType.CALL) -> pd.DataFrame:
    """
    Compare implied volatility from options with realized volatility.
    
    Parameters:
    asset_ticker (str): ticker name
    start_date (str): start date %Y-%m-%d
    end_date (str): end date %Y-%m-%d
    window (int): window for realized volatility calculation
    style (OptionStyle): option style
    type (OptionType): option type
    
    Returns:
    pd.DataFrame: DataFrame with implied and realized volatilities
    """
    # Get option data
    options_df = get_option_data(asset_ticker, start_date, end_date, style=style, type=type)
    
    if options_df.empty:
        return pd.DataFrame()
    
    # Get asset prices
    prices_df = get_asset_prices(asset_ticker, start_date, end_date)
    prices_df['TradeDate'] = pd.to_datetime(prices_df['TradeDate'])
    prices_df.set_index('TradeDate', inplace=True)
    
    # Calculate realized volatility (EWMA)
    realized_vol = ewma_volatility(prices_df['Asset Price'], alpha=0.94).copy()
    # Prepare comparison DataFrame
    resultado = filter_recent_maturity(options_df, min_trade_qty=5)
    
    for _, row in resultado.iterrows():
        trade_date = row['TradeDate']
        # asset_price = row['Asset Price']
        if trade_date not in realized_vol.index: continue
        print(trade_date)
        # # recent_maturity = row['Maturity']
        # # vol_surf = vol_surface(asset_ticker, trade_date, type='black')
        # # vol_surf = vol_surf.loc[vol_surf['Maturity'] == recent_maturity]
        # # vol_surf['Price Difference'] = abs(vol_surf['Strike'] - asset_price)
        # # vol_surf = vol_surf.loc[vol_surf['Price Difference'].idxmin()]
        # # iv_latest = vol_surf['Implied Volatility']
        # realized_vol.loc[trade_date, 'Implied Volatility'] = iv_latest
        realized_vol.loc[trade_date, 'Implied Volatility'] = estimate_vol(asset_ticker, trade_date, option_type=type)

    return realized_vol

def plot_asset_prices(asset_ticker: str, database: str, _ndays = 10):
    data_start = nworkdays(database, -1*_ndays)
    data_end = nworkdays(database, _ndays)
    prices_df = get_asset_prices(asset_ticker, data_start, data_end)
    plt.figure(figsize=(10, 6))
    plt.plot(prices_df['TradeDate'], prices_df['Asset Price'], color='darkolivegreen')
    plt.title(f'Asset Prices for {asset_ticker} from {data_start} to {data_end}')
    plt.xlabel('Date')
    plt.ylabel('Asset Price')
    plt.grid()
    plt.show()

def test_returns():
    S0=100
    tau=0.5
    r=0.1
    sigma=0.5
    M=1
    dt=0.001
    database = "2025-01-30"
    # database = "2020-10-16"
    ticker = 'PETR4'

    # dataend = nworkdays(database, int(252/2))
    data_start =  nworkdays(database, 2)
    data_end = nworkdays(data_start, int(230))
    print(data_start, data_end)
    S = get_asset_prices(ticker, data_start, data_end)
    S = S['Asset Price'].values
    plot_returns(S)

    _, W = geometric_brownian_motion(S0=S0, tau=tau, dt=dt, r=r, sigma=sigma, M=M)
    plot_returns(W)

    heston_params = load_params('heston', database, ticker)
    print(heston_params)
    _, W_h, _ = heston_model(S0=S0, tau=tau, dt=dt, r=r, M=M, **heston_params)
    plot_returns(W_h)

    kou_params = load_params('kou', database, ticker)
    print(kou_params)
    t, W_k = kou_process(S0=S0, tau=tau, dt=dt, r=r, M=M, **kou_params)
    # plot_returns(W_k, bins = int(50 * tau * 20))
    plot_returns(W_k)

def test_smile():
    # database = "2025-01-30"
    # database = "2023-11-01"
    # database = "2021-04-20"
    ticker = "PETR4"
    for database in ['2021-04-20', '2023-11-01', '2025-01-30']:
        surface_market = vol_surface(ticker, database, 'market')
        # print(surface_black['Maturity'].value_counts().sort_values(ascending=False))
        maturity = surface_market['Maturity'].value_counts().sort_values(ascending=False).index[0]
        surface_market = surface_market[surface_market['Maturity'] == maturity].sort_values(by='Strike')
        surface_heston = vol_surface(ticker, database, 'heston')
        surface_heston = surface_heston[surface_heston['Maturity'] == maturity].sort_values(by='Strike')
        surface_kou = vol_surface(ticker, database, 'kou')
        surface_kou = surface_kou[surface_kou['Maturity'] == maturity].sort_values(by='Strike')
        vol_hist = estimate_sigma_hist(ticker, database, _ndays=7).values.flatten()[-1]
        plt.figure(figsize=(10, 6))
        plt.scatter(surface_market['Strike'], surface_market['Implied Volatility'], color='darkolivegreen', label='Market Implied Volatility', s=10)
        plt.plot(surface_heston['Strike'], surface_heston['Implied Volatility'], color='indigo', linestyle='dashed', label='Heston Implied Volatility')
        plt.plot(surface_kou['Strike'], surface_kou['Implied Volatility'], color='darkgoldenrod', linestyle='dashed', label='Kou Implied Volatility')
        plt.axhline(y=vol_hist, color='teal', linestyle='dashed', label='BS Historical Volatility')
        plt.title(f'Implied Volatility Smile Data Base {database} for {ticker} on maturity {maturity}')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid()
        plt.show()

def test_vol():
    asset_ticker = 'PETR4'
    start_date = '2021-04-12'
    end_date = '2021-04-20'
    
    vol_comparison = get_option_implied_vs_realized_vol(asset_ticker, start_date, end_date)
    print(vol_comparison)

def test_asset_prices():
    asset_ticker = 'PETR4'
    for database in ['2021-04-20', '2023-11-01', '2025-01-30']:
        plot_asset_prices(asset_ticker, database, _ndays=30)
    
if __name__ == "__main__":
    # test_returns()
    # test_vol()
    test_smile()
    # test_asset_prices()