from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from brownian_motion import geometric_brownian_motion

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

if __name__ == "__main__":
    S0 = 100
    sigma = 0.16
    r = 0.05
    T = 0.5
    K = 98

    BSM_call = black_scholes_merton(S0, K, T, r, sigma, OptionType.CALL)
    BSM_put = black_scholes_merton(S0, K, T, r, sigma, OptionType.PUT)

    print(f'BSM Call Option Price: {BSM_call:.2f}')
    print(f'BSM Put Option Price: {BSM_put:.2f}')
    
    # Monte Carlo pricing
    MC_call = black_scholes_monte_carlo(S0, K, T, r, sigma, OptionType.CALL, num_simulations=30_000, seed=42)
    MC_put = black_scholes_monte_carlo(S0, K, T, r, sigma, OptionType.PUT, num_simulations=30_000, seed=42)
    
    print(f'\nMonte Carlo Call Option Price: {MC_call:.2f}')
    print(f'Monte Carlo Put Option Price: {MC_put:.2f}')
    print(f'\nCall Price Difference: {abs(BSM_call - MC_call):.4f}')
    print(f'Put Price Difference: {abs(BSM_put - MC_put):.4f}')


