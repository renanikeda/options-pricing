from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from enum import Enum

from utils import OptionType


def black_scholes_merton(S, K, T, r=0.07, sigma=0.2, option_type=OptionType.CALL):
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


if __name__ == "__main__":
    S0 = 10
    sigma = 0.2
    r = 0.5
    T = 1
    K = 15

    BSM_call = black_scholes_merton(S0, K, T, r, sigma, OptionType.CALL)
    BSM_put = black_scholes_merton(S0, K, T, r, sigma, OptionType.PUT)

    print(f'BSM Call Option Price: {BSM_call:.2f}')
    print(f'BSM Put Option Price: {BSM_put:.2f}')


