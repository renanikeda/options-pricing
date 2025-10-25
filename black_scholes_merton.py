from brownian_motion import geometric_brownian_motion
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

def black_scholes_merton(S, K, T, r, sigma, option_type=OptionType.CALL):
    """
    Calculates the Black-Scholes-Merton option price.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (str): 'call' for a call option, 'put' for a put option

    Returns:
    float: The calculated option price
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == OptionType.CALL:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == OptionType.PUT:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

dt=0.01
T=5
M=10
r = 0.07
sigma=0.2
t, GBM = geometric_brownian_motion(r=r, sigma=sigma, M=M, T=T, dt=dt)
plt.plot(t, GBM)
plt.title(f'Geometric Brownian Motion - {M} Paths')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.grid()
plt.show()
