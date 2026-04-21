
from brownian_motion import geometric_brownian_motion
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis, shapiro, anderson, normaltest
import seaborn as sns
from typing import List, Literal

from black_scholes import implied_vol
from heston_model import heston_model, heston_price
from kou_jump_diffusion import kou_option_price, kou_process
from calibration import estimate_v0
from utils import OptionType, ewma_volatility, get_option_data, get_asset_prices, get_selic, load_params, nworkdays, estimate_sigma_hist, OptionStyle
from imp_vol import vol_surface
from datetime import datetime

def plot_returns(flat_returns: np.array, bins: int = 100) -> None:
    """
    Plot the returns distribution and QQ plot.
    Args:
        W (np.ndarray): Simulated stock prices matrix.
    """
    
    # returns = np.diff(np.log(W), axis=0)
    # flat_returns = returns.flatten()

    print('ploting returns shape:', flat_returns.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(211)
    xlim = np.std(flat_returns) * 4
    ax.set_xlim([-xlim, xlim])
    sns.histplot(flat_returns, bins=bins, color='darkolivegreen', stat='density', ax=ax)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, bins*2)
    p = norm.pdf(x, np.mean(flat_returns), np.std(flat_returns))
    plt.plot(x, p, color='r', linestyle='dashed', linewidth=3, label='Dist. Normal')
    plt.xlabel(f'Retorno diário')
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

def plot_n_returns(n_flat_returns: List[np.array], titles: List[str], size: int, bins: int = 100) -> None:
    """
    Plot the returns distribution and QQ plot for multiple sets of returns.
    Args:
        n_flat_returns (List[np.array]): List of flattened returns arrays.
        titles (List[str]): List of titles for each plot.
        size (int): Number of return sets to plot.
        bins (int): Number of bins for the histogram.
    """
    
    # returns = np.diff(np.log(W), axis=0)
    # flat_returns = returns.flatten()

    print('ploting returns shape:', [n_flat_returns[i].shape for i in range(size)])
    fig = plt.figure(figsize=(14, 7))
    for n in range(1, size+1):
        flat_returns = n_flat_returns[n-1]
        # print(int(f'2{size}{n}'), int(f'2{size}{n+size}'))
        ax = fig.add_subplot(int(f'2{size}{n}'))

        xlim = np.std(flat_returns) * 4
        ax.set_xlim([-xlim, xlim])
        sns.histplot(flat_returns, bins=bins, color='darkolivegreen', stat='density', ax=ax)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, bins*2)
        p = norm.pdf(x, np.mean(flat_returns), np.std(flat_returns))
        plt.plot(x, p, color='r', linestyle='dashed', linewidth=3, label='Dist. Normal')
        plt.xlabel(f'Retorno diário')
        plt.ylabel('Frequência')
        plt.grid(linewidth=0.3)
        plt.legend()
        plt.title(titles[n-1])

        ax = fig.add_subplot(int(f'2{size}{n+size}'))
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
    sigma=0.5
    M=100
    dt=0.001
    ndays = 252
    daily_steps = int(1/dt)
    database = '2020-07-13'
    ddatabase = '2021-04-20'
    ticker = 'PETR4'
    moneyness_divergence = 0.6
    r = get_selic(database)/100
    # for database in ['2021-04-20', '2023-11-01', '2025-01-30']:
    data_start =  nworkdays(database, 2)
    data_end = nworkdays(data_start, ndays)
    # tau = ndays/252
    
    print(data_start, data_end)
    S = get_asset_prices(ticker, data_start, data_end)
    S = S['Asset Price'].values
    S = np.diff(np.log(S), axis=0)
    S = S.flatten()
    # print(f'Curtose Retorno Mercado database {ticker}', round(kurtosis(S, axis=0, bias=False, fisher=False), 2))
    # print(f'Assimetria Retorno Mercado database {ticker}', round(skew(S, axis=0, bias=False), 2))
    # stat, p = shapiro(S)
    # print(f'Shapiro-Wilk Mercado database {ticker}: Stat={stat:.4f}, p={p:.4e}')
    # plot_returns(S)

    _, W = geometric_brownian_motion(S0=S0, tau=ndays, dt=dt, r=r, sigma=sigma, M=M)
    W = W[::daily_steps, :]
    W = np.diff(np.log(W), axis=0)

    mu=np.mean(W)
    # W = W.flatten()
    # print(f'Curtose Retorno MB database {ticker}', np.mean(np.round(kurtosis(W, axis=0, bias=False, fisher=False), 2)))
    # print(f'Assimetria Retorno MB database {ticker}', np.mean(np.round(skew(W, axis=0, bias=False), 2)))
    # stat, p = shapiro(W[:,0].flatten())
    # print(f'Shapiro-Wilk MB (path 0) database {ticker}: Stat={stat:.4f}, p={p:.4e}')
    # plot_returns(W[:,0].flatten())

    # plot_n_returns([S, W.flatten()], [f'Retorno Mercado {ticker}', f'Retorno MBG {ticker}'], size=2)
    plot_n_returns([S, W[:,0].flatten()], [f'Retorno Mercado {ticker}', f'Retorno MBG {ticker}'], size=2)

    heston_params = load_params('heston', database, ticker, params_file=f'calibrated_params {moneyness_divergence*100}%.json')
    print(heston_params)
    _, W_h, _ = heston_model(S0=S0, tau=ndays, dt=dt, r=mu, M=M, **heston_params)
    W_h = W_h[::daily_steps, :]
    W_h = np.diff(np.log(W_h), axis=0)
    # W_h = W_h.flatten()
    # print(f'Curtose Retorno Heston database {ticker}', np.mean(np.round(kurtosis(W_h, axis=0, bias=False, fisher=False), 2)))
    # print(f'Assimetria Retorno Heston database {ticker}', np.mean(np.round(skew(W_h, axis=0, bias=False), 2)))
    # stat, p = shapiro(W_h[:,0].flatten())
    # print(f'Shapiro-Wilk Heston (path 0) database {ticker}: Stat={stat:.4f}, p={p:.4e}')
    # plot_returns(W_h[:,0].flatten())

    kou_params = load_params('kou', database, ticker, params_file=f'calibrated_params {moneyness_divergence*100}%.json')
    print(kou_params)
    t, W_k = kou_process(S0=S0, tau=ndays, dt=dt, r=mu, M=M, **kou_params)
    W_k = W_k[::daily_steps, :]
    W_k = np.diff(np.log(W_k), axis=0)
    # W_k = W_k.flatten()
    # plot_returns(W_k, bins = int(50 * tau * 10))
    # print(f'Curtose Retorno Kou database {ticker}', np.mean(np.round(kurtosis(W_k, axis=0, bias=False, fisher=False), 2)))
    # print(f'Assimetria Retorno Kou database {ticker}', np.mean(np.round(skew(W_k, axis=0, bias=False), 2)))
    # stat, p = shapiro(W_k[:,0].flatten())
    # print(f'Shapiro-Wilk Kou (path 0) database {ticker}: Stat={stat:.4f}, p={p:.4e}')
    # plot_returns(W_k[:,0].flatten())
    # plot_n_returns([W_h.flatten(), W_k.flatten()], ['Retorno Heston', 'Retorno Kou'], size=2)
    plot_n_returns([W_h[:,0].flatten(), W_k[:,0].flatten()], [f'Retorno Heston {ticker}', f'Retorno Kou {ticker}'], size=2)

def save_returns_metrics():
    S0=100
    M=100
    dt=0.001
    ndays = 252
    daily_steps = int(1/dt)
    moneyness_divergence = 0.6
    
    results = []
    for ticker in ['PETR4', 'VALE3']:
        for database in ['2020-07-13', '2021-04-20', '2022-04-18', '2023-11-01', '2025-01-30', '2025-06-10']:
            for moneyness_divergence in [0.15, 0.6]:
                r = get_selic(database)/100
                # for database in ['2021-04-20', '2023-11-01', '2025-01-30']:
                data_start =  nworkdays(database, 2)
                data_end = nworkdays(data_start, ndays if database != '2025-06-10' else 200)
                # tau = ndays/252
                
                print(data_start, data_end)
                S = get_asset_prices(ticker, data_start, data_end)
                S = S['Asset Price'].values
                S = np.diff(np.log(S), axis=0)
                S = S.flatten()
                mu = np.mean(S)
                sigma = np.std(S)
                market_kurt = round(kurtosis(S, axis=0, bias=False, fisher=False), 8)
                market_skew = round(skew(S, axis=0, bias=False), 8)
                print(f'Curtose Retorno Mercado database {ticker}', market_kurt)
                print(f'Assimetria Retorno Mercado database {ticker}', market_skew)
                # market_stat, market_p = shapiro(S)
                market_stat, market_p = normaltest(S)
                print(f'D’Agostino Test Mercado database {ticker}: Stat={market_stat:.4f}, p={market_p:.4e}')

                _, W = geometric_brownian_motion(S0=S0, tau=ndays, dt=dt, r=mu, sigma=sigma, M=M)
                W = W[::daily_steps, :]
                W = np.diff(np.log(W), axis=0)
                gbm_kurt = np.mean(np.round(kurtosis(W, axis=0, bias=False, fisher=False), 8))
                gbm_skew = np.mean(np.round(skew(W, axis=0, bias=False), 8))
                print(f'Curtose Retorno MB database {ticker}', gbm_kurt)
                print(f'Assimetria Retorno MB database {ticker}', gbm_skew)
                # gbm_stat, gbm_p = shapiro(W[:,0].flatten())
                gbm_stat, gbm_p = normaltest(W.flatten())
                print(f'D’Agostino Test MB (path 0) database {ticker}: Stat={gbm_stat:.4f}, p={gbm_p:.4e}')


                heston_params = load_params('heston', database, ticker, params_file=f'calibrated_params {moneyness_divergence*100}%.json')
                # print(heston_params)
                _, W_h, _ = heston_model(S0=S0, tau=ndays, dt=dt, r=mu, M=M, **heston_params)
                W_h = W_h[::daily_steps, :]
                W_h = np.diff(np.log(W_h), axis=0)
                heston_kurt= np.mean(np.round(kurtosis(W_h, axis=0, bias=False, fisher=False), 8))
                heston_skew = np.mean(np.round(skew(W_h, axis=0, bias=False), 8))
                print(f'Curtose Retorno Heston database {ticker}', heston_kurt)
                print(f'Assimetria Retorno Heston database {ticker}', heston_skew)
                # heston_stat, heston_p = shapiro(W_h[:,0].flatten())
                heston_stat, heston_p = normaltest(W_h.flatten())
                print(f'D’Agostino Test Heston (path 0) database {ticker}: Stat={heston_stat:.4f}, p={heston_p:.4e}')

                kou_params = load_params('kou', database, ticker, params_file=f'calibrated_params {moneyness_divergence*100}%.json')
                # print(kou_params)
                _, W_k = kou_process(S0=S0, tau=ndays, dt=dt, r=mu, M=M, **kou_params)
                W_k = W_k[::daily_steps, :]
                W_k = np.diff(np.log(W_k), axis=0)
                kou_kurt = np.mean(np.round(kurtosis(W_k, axis=0, bias=False, fisher=False), 8))
                kou_skew = np.mean(np.round(skew(W_k, axis=0, bias=False), 8))
                print(f'Curtose Retorno Kou database {ticker}', kou_kurt)
                print(f'Assimetria Retorno Kou database {ticker}', kou_skew)
                # kou_stat, kou_p = shapiro(W_k[:,0].flatten())
                kou_stat, kou_p = normaltest(W_k.flatten())
                print(f'D’Agostino Test Kou (path 0) database {ticker}: Stat={kou_stat:.4f}, p={kou_p:.4e}')

                for medida in ['Curtose', 'Assimetria', 'D’Agostino p-value']:
                    results.append({
                        'Ticker': ticker,
                        'Database': datetime.strptime(database, '%Y-%m-%d').strftime('%d/%m/%Y'),
                        'Medida': medida,
                        'Mercado': format(float(market_kurt) if medida == 'Curtose' else market_skew if medida == 'Assimetria' else market_p, '.4f'),
                        'Black-Scholes': format(float(gbm_kurt) if medida == 'Curtose' else gbm_skew if medida == 'Assimetria' else gbm_p, '.4f'),
                        'Heston': format(float(heston_kurt) if medida == 'Curtose' else heston_skew if medida == 'Assimetria' else heston_p, '.4f'),
                        'Kou': format(float(kou_kurt) if medida == 'Curtose' else kou_skew if medida == 'Assimetria' else kou_p, '.4f'),
                        'Moneyness Divergence': moneyness_divergence,

                        # 'Market Kurtosis': round(float(market_kurt), 4),
                        # 'Market Skewness': round(float(market_skew), 4),
                        # # 'Market D’Agostino Stat': round(float(market_stat), 8),
                        # 'Market D’Agostino p-value': round(float(market_p), 4),
                        # 'GBM Kurtosis': round(float(gbm_kurt), 4),
                        # 'GBM Skewness': round(float(gbm_skew), 4),
                        # # 'GBM D’Agostino Stat': round(float(gbm_stat), 8),
                        # 'GBM D’Agostino p-value': round(float(gbm_p), 4),
                        # 'Heston Kurtosis': round(float(heston_kurt), 4),
                        # 'Heston Skewness': round(float(heston_skew), 4),
                        # # 'Heston D’Agostino Stat': round(float(heston_stat), 8),
                        # 'Heston D’Agostino p-value': round(float(heston_p), 4),
                        # 'Kou Kurtosis': round(float(kou_kurt), 4),
                        # 'Kou Skewness': round(float(kou_skew), 4),
                        # # 'Kou D’Agostino Stat': round(float(kou_stat), 8),
                        # 'Kou D’Agostino p-value': round(float(kou_p), 4),
                        # 'Moneyness Divergence': moneyness_divergence,
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv('stylized_facts_results.csv', index=False)

def check_smile():
    # database = "2025-01-30"
    # database = "2023-11-01"
    # database = "2021-04-20"   
    ticker = "VALE3"
    # for database in ['2021-04-20', '2023-11-01', '2025-01-30']:
    # database = '2022-04-18'
    database = '2025-06-10'
    surface_market = vol_surface(ticker, database, 'market')
    # print(surface_market)
    maturity = surface_market['Maturity'].value_counts().sort_values(ascending=False).index[0]
    surface_market = surface_market[surface_market['Maturity'] == maturity].sort_values(by='Strike')
    vol_hist = estimate_sigma_hist(ticker, database, _ndays=7).values.flatten()[-1]
    plt.figure(figsize=(10, 6))
    plt.scatter(surface_market['Strike'], surface_market['Implied Volatility'], color='darkolivegreen', label='Market Implied Volatility', s=10)
    plt.axhline(y=vol_hist, color='teal', linestyle='dashed', label='BS Historical Volatility')
    plt.title(f'Implied Volatility Smile Data Base {database} for {ticker} on maturity {maturity}')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid()
    plt.show()

def test_smile():
    ticker = "PETR4"
    moneynesses = [0.15, 0.6]
    for database in ['2020-07-13', '2021-04-20', '2022-04-18', '2023-11-01', '2025-01-30', '2025-06-10']:
    # for database in ['2021-04-20', '2022-04-18']:
        n_cols = len(moneynesses)
        fig, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
        axs = axs.flatten()
        fig.suptitle(f'Sorriso de volatilidade Data Base {database} para {ticker}')
        maturity = None
        for (index, moneyness_divergence) in enumerate(moneynesses):
        # for moneyness_divergence in [0.2, 0.7]:
            surface_market = vol_surface(ticker, database, 'market', moneyness_divergence=moneyness_divergence)
            # print(surface_black['Maturity'].value_counts().sort_values(ascending=False))
            maturity = maturity or surface_market['Maturity'].value_counts().sort_values(ascending=False).index[0]
            print(f"Database: {database}, Moneyness Divergence: {moneyness_divergence}, Maturity: {maturity}")

            surface_market = surface_market[surface_market['Maturity'] == maturity].sort_values(by='Strike')
            surface_heston = vol_surface(ticker, database, 'heston', moneyness_divergence=moneyness_divergence)
            surface_heston = surface_heston[surface_heston['Maturity'] == maturity].sort_values(by='Strike')
            surface_kou = vol_surface(ticker, database, 'kou', moneyness_divergence=moneyness_divergence)
            surface_kou = surface_kou[surface_kou['Maturity'] == maturity].sort_values(by='Strike')
            vol_hist = estimate_sigma_hist(ticker, database, _ndays=10).values.flatten()[-1]
            valid_mask = ~np.isnan(surface_market['Implied Volatility']) & ~np.isnan(surface_heston['Implied Volatility']) & ~np.isnan(surface_kou['Implied Volatility'])
            surface_market = surface_market[valid_mask]
            surface_heston = surface_heston[valid_mask]
            surface_kou = surface_kou[valid_mask]
            sqr_error_heston = (1/len(surface_market['Implied Volatility'])) * np.mean((surface_market['Implied Volatility'] - surface_heston['Implied Volatility']) ** 2)
            sqr_error_kou = (1/len(surface_market['Implied Volatility'])) * np.mean((surface_market['Implied Volatility'] - surface_kou['Implied Volatility']) ** 2)
            print(f'Squared Error Heston: {format(sqr_error_heston, ".6f")}, Squared Error Kou: {format(sqr_error_kou, ".6f")}')
            axs[index].scatter(surface_market['Strike'], surface_market['Implied Volatility'], color='darkolivegreen', label='Market', s=10)
            axs[index].plot(surface_heston['Strike'], surface_heston['Implied Volatility'], color='indigo', linestyle='dashed', label='Heston')
            axs[index].plot(surface_kou['Strike'], surface_kou['Implied Volatility'], color='darkgoldenrod', linestyle='dashed', label='Kou')
            axs[index].axhline(y=vol_hist, color='teal', linestyle='dashed', label='BS Volatility')
            axs[index].set_title(f'Moneyness [{round(1-moneyness_divergence, 2)}, {round(1+moneyness_divergence, 2)}], maturity {maturity}')
            axs[index].set_xlabel('Preço de Exercício')
            if index == 0:
                axs[index].set_ylabel('Volatilidade Implícita')
            axs[index].legend()
            axs[index].grid()
        
        plt.tight_layout()
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
    test_returns()
    # save_returns_metrics()
    # test_vol()
    # test_smile()
    # check_smile()
    # test_asset_prices()