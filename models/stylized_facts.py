
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
from utils import OptionType, get_option_data, load_params

def vol_surface(ticker: str, database: str, r: float = 0.1, type: Literal['heston', 'kou', 'black'] = 'black' ):
    options_data = get_option_data(ticker, database, database)
    options_data = options_data[options_data['Asset Ticker'] == ticker]
    imp_vols = []
    for (_, row) in options_data.iterrows():
        if type == 'heston':
            params = load_params("heston", database, ticker)
            price = heston_price(**{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] }, **params)
            print(f"Heston price: {price:.2f}, Market price: {row['LastPrice']}")
        elif type == 'kou':
            params = load_params("kou", database, ticker)
            price = kou_option_price(**{ 'S0': row['Asset Price'], 'K': row['Strike'], 'r': r, 'tau': row['Days to Maturity'] }, **params)
            print(f"Kou price: {price:.2f}, Market price: {row['LastPrice']}")
        elif type == 'black':
            price = row['LastPrice']
            print(f"Black-Scholes, Market price: {row['LastPrice']}")
        imp_vol = implied_vol(price, row['Asset Price'], row['Strike'], row['Days to Maturity'], r=r, option_type=OptionType.CALL if row.Type == "CALL" else OptionType.PUT)
        imp_vols.append(imp_vol)
    vol_surface = pd.DataFrame()
    vol_surface['Strike'] = options_data['Strike']
    vol_surface['Maturity'] = options_data['Maturity']    
    vol_surface['Implied Volatility'] = imp_vols
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

def test_returns():
    S0=100
    tau=1
    r=0.1
    sigma=0.5
    M=25
    dt=0.001
    database = "2025-01-30"
    ticker = 'PETR4'

    _, W = geometric_brownian_motion(S0=S0, tau=tau, dt=dt, r=r, sigma=sigma, M=M)
    plot_returns(W)

    heston_params = load_params('heston', database, ticker)
    print(heston_params)
    _, W_h, _ = heston_model(S0=S0, tau=tau, dt=dt, r=r, M=M, **heston_params)
    plot_returns(W_h)

    kou_params = load_params('kou', database, ticker)
    print(kou_params)
    t, W_k = kou_process(S0=S0, tau=tau, dt=dt, r=r, M=M, **kou_params)
    plot_returns(W_k, bins = 1000)


def test_smile():
    database = "2025-01-30"
    # maturity = "2025-06-20"
    maturity = "2025-03-21"
    ticker = "PETR4"
    # model = 'kou'  # 'heston', 'kou', 'black'
    # df = vol_surface(ticker, database, 0.1, model)
    # df = df[df['Maturity'] == maturity]
    surface_black = vol_surface(ticker, database, 0.1, 'black')
    surface_black = surface_black[surface_black['Maturity'] == maturity]
    surface_heston = vol_surface(ticker, database, 0.1, 'heston')
    surface_heston = surface_heston[surface_heston['Maturity'] == maturity]
    surface_kou = vol_surface(ticker, database, 0.1, 'kou')
    surface_kou = surface_kou[surface_kou['Maturity'] == maturity]
    plt.figure(figsize=(10, 6))
    plt.scatter(surface_black['Strike'], surface_black['Implied Volatility'], color='darkolivegreen', label='Black Implied Volatility', s=10)
    plt.scatter(surface_heston['Strike'], surface_heston['Implied Volatility'], color='indigo', label='Heston Implied Volatility', s=10)
    plt.scatter(surface_kou['Strike'], surface_kou['Implied Volatility'], color='darkgoldenrod', label='Kou Implied Volatility', s=10)
    plt.title(f'Implied Volatility Smile for {ticker} on maturity {maturity}')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    test_returns()
    test_smile()
