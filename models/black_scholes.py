from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from brownian_motion import geometric_brownian_motion
from utils import get_option_data

from utils import OptionType

def black_scholes(S, K, T, r=0.07, sigma=0.2, option_type=OptionType.CALL):
    """
    Calculates the Black-Scholes-Merton option price.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (enum): 'call' for a call option, 'put' for a put option

    Returns:
    float: The calculated option price
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == OptionType.CALL:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == OptionType.PUT:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def black_scholes_monte_carlo(S, K, T, r=0.07, sigma=0.2, option_type=OptionType.CALL, num_simulations=100000, seed=None):
    """
    Calculates Black-Scholes option price using Monte Carlo simulation.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (enum): OptionType.CALL for a call option, OptionType.PUT for a put option
    num_simulations (int): Number of Monte Carlo simulations
    seed (int): Random seed for reproducibility

    Returns:
    float: The calculated option price
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random price paths using Geometric Brownian Motion
    # Z = np.random.standard_normal(num_simulations)
    # ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    _, S = geometric_brownian_motion(S0=S, T=T, dt=T, r=r, sigma=sigma, M=num_simulations)
    ST = S[-1, :]
    # Calculate payoffs
    if option_type == OptionType.CALL:
        payoffs = np.maximum(ST - K, 0)
    elif option_type == OptionType.PUT:
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")
    
    # Discount the average payoff to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

def black_scholes_vega(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def implied_vol_newton_raphson(price_mkt: float, S0: float, K: float, T: float, r: float = 0.1, sigma_0: float = 0.25, option_type: OptionType = OptionType.CALL, tol: float = 1e-6, max_iter: int = 100):
    """
    Calculate the implied volatility using the Black-Scholes-Merton model.

    Parameters:
    Price_mkt (float): Market price of the option
    S0 (float): Current stock price
    sigma_0 (float): Volatility initial guess
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)

    Returns:
    float: The implied volatility
    """

    epsilon = 1e-6
    error = 1.0
    iteration = 0
    imp_vol = sigma_0
    
    while iteration < max_iter and error > epsilon:
        g = price_mkt - black_scholes(S0, K, T, r, imp_vol, option_type)
        vega = black_scholes_vega(S0, K, r, imp_vol, T)
        if vega < 1e-5:
            break
        imp_vol_new = imp_vol - (g / vega)
        error = abs(imp_vol_new - imp_vol)
        imp_vol = max(imp_vol_new, tol)
        if error < tol:
            return imp_vol
        iteration += 1
    raise RuntimeError("Newton-Raphson não convergiu")


def implied_vol_bissection(price_mkt: float, S0: float, K: float, T: float, r: float = 0.1, vol_low: float = 1e-8, vol_high: float = 5, option_type: OptionType = OptionType.CALL, tol: float = 1e-6, max_iter: int = 100):
    """
    Calculate the implied volatility using the bisection method.
    Parameters:
        price_mkt (float): Market price of the option
        S0 (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        vol_low (float): Lower bound for volatility
        vol_high (float): Upper bound for volatility
        option_type (enum): OptionType.CALL for a call option, OptionType.PUT for a put option
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
    Returns:
        float: The implied volatility
    """
    for _ in range(max_iter):
        vol_mid = 0.5 * (vol_low + vol_high)
        price_mid = black_scholes(S0, K, T, r, vol_mid, option_type)

        if abs(price_mid - price_mkt) < tol:
            return vol_mid

        if price_mid > price_mkt:
            vol_high = vol_mid
        else:
            vol_low = vol_mid

    return np.nan

def implied_vol(price_mkt: float, S0: float, K: float, T: float, r: float = 0.1, vol_low: float = 1e-8, vol_high: float = 5, option_type: OptionType = OptionType.CALL, tol: float = 1e-6, max_iter: int = 100):
    """
    Calculate the implied volatility using the bisection method.
    Parameters:
        price_mkt (float): Market price of the option
        S0 (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        vol_low (float): Lower bound for volatility
        vol_high (float): Upper bound for volatility
        option_type (enum): OptionType.CALL for a call option, OptionType.PUT for a put option
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
    Returns:
        float: The implied volatility
    """
    try:
        return implied_vol_newton_raphson(price_mkt, S0, K, T, r, sigma_0=0.2, option_type=option_type, tol=tol, max_iter=max_iter)
    except RuntimeError:
        return implied_vol_bissection(price_mkt, S0, K, T, r, vol_low, vol_high, option_type, tol, max_iter)
    
def test_smile():
    database = "2025-05-02"
    maturity = "2025-06-20"
    ticker = "PETR4"
    df = get_option_data(ticker, database, database)
    df = df[df['Maturity'] == maturity]
    df = df[df['Asset Ticker'] == ticker]
    imp_vols = []
    for row in df.itertuples():
        imp_vol = implied_vol(row.LastPrice, row._16, row.Strike, row._17, r=0.1, option_type=OptionType.CALL if row.Type == "CALL" else OptionType.PUT)
        imp_vols.append(imp_vol)

    df['Implied Volatility'] = imp_vols
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Strike'], df['Implied Volatility'], color='blue', label='Implied Volatility')
    plt.title(f'Implied Volatility Smile for {ticker} on maturity {maturity}')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # S0 = 100
    # sigma = 0.16
    # r = 0.05
    # T = 0.5
    # K = 98

    # BSM_call = black_scholes(S0, K, T, r, sigma, OptionType.CALL)
    # BSM_put = black_scholes(S0, K, T, r, sigma, OptionType.PUT)

    # print(f'BSM Call Option Price: {BSM_call:.2f}')
    # print(f'BSM Put Option Price: {BSM_put:.2f}')
    
    # # Monte Carlo pricing
    # MC_call = black_scholes_monte_carlo(S0, K, T, r, sigma, OptionType.CALL, num_simulations=30_000, seed=42)
    # MC_put = black_scholes_monte_carlo(S0, K, T, r, sigma, OptionType.PUT, num_simulations=30_000, seed=42)
    
    # print(f'\nMonte Carlo Call Option Price: {MC_call:.2f}')
    # print(f'Monte Carlo Put Option Price: {MC_put:.2f}')
    # print(f'\nCall Price Difference: {abs(BSM_call - MC_call):.4f}')
    # print(f'Put Price Difference: {abs(BSM_put - MC_put):.4f}')
    # print(implied_vol(4.96, 30.66, 25.97, 16/252, 0.1))
    
    # print(implied_vol(5.2, 30.47, 26.72, 96/252, 0.1))
    test_smile()
