
from brownian_motion import geometric_brownian_motion
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns

from black_scholes import implied_vol
from heston_model import heston_model
from kou_jump_diffusion import kou_process
from utils import OptionType, get_option_data, load_params

def vol_surface(ticker: str, database: str, r: float = 0.1):
    options_data = get_option_data(ticker, database, database)
    options_data = options_data[options_data['Asset Ticker'] == ticker]
    imp_vols = []
    for (_, row) in options_data.iterrows():
        imp_vol = implied_vol(row['LastPrice'], row['Asset Price'], row['Strike'], row['Days to Maturity'], r=r, option_type=OptionType.CALL if row.Type == "CALL" else OptionType.PUT)
        imp_vols.append(imp_vol)
    vol_surface = pd.DataFrame()
    vol_surface['Strike'] = options_data['Strike']
    vol_surface['Maturity'] = options_data['Maturity']    
    vol_surface['Implied Volatility'] = imp_vols
    return vol_surface

def plot_returns(W: np.array) -> None:
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
    sns.histplot(flat_returns, bins=50, color='darkolivegreen', stat='density', ax=ax)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(flat_returns), np.std(flat_returns))
    plt.plot(x, p, color='r', linestyle='dashed', linewidth=3, label='Gaussian')
    plt.xlabel(f'GBM retorno diário')
    plt.ylabel('Frequência')
    plt.legend()

    ax = fig.add_subplot(212)
    qqplot(flat_returns, line='s', ax=ax, markerfacecolor='darkolivegreen', fit=True)
    plt.xlabel('Quantil Teórico')
    plt.ylabel('Quantil Amostral')
    plt.xlim([-4.5,4.5])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    S0=100
    tau=1
    r=0.1
    sigma=0.5
    database = '2025-05-02'
    _, W = geometric_brownian_motion(S0=S0, tau=tau, dt=0.001, r=r, sigma=sigma, M=100)
    plot_returns(W)
    heston_params = load_params('heston', database)
    print(heston_params)
    _, W_h, _ = heston_model(S0=S0, tau=tau, dt=0.001, r=r, M=100, **heston_params)
    plot_returns(W_h)
    kou_params = load_params('kou', database)
    print(kou_params)
    t, W_k = kou_process(S0=S0, tau=tau, dt=0.001, r=r, M=100, **kou_params)
    plot_returns(W_k)
    # plt.plot(t, W_k[:, :5], '.', color='darkolivegreen', markersize=2)
    # plt.grid()
    # plt.show()
