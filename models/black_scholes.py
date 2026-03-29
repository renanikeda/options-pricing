from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from brownian_motion import geometric_brownian_motion
from utils import OptionType
import pandas as pd

def black_scholes(S0, K, tau, r=0.07, sigma=0.2, option_type=OptionType.CALL):
    """
    Calculate the Black-Scholes-Merton option price.

    Parameters:
    S0 (float): Current stock price
    K (float): Strike price
    tau (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (OptionType): OptionType.CALL for a call option, OptionType.PUT for a put option

    Returns:
    float: The calculated option price
    
    Raises:
    ValueError: If option_type is not OptionType.CALL or OptionType.PUT
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_type == OptionType.CALL:
        return S0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    elif option_type == OptionType.PUT:
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")

def black_scholes_monte_carlo(S, K, tau, r=0.07, sigma=0.2, option_type=OptionType.CALL, num_simulations=100000, seed=None):
    """
    Calculate Black-Scholes option price using Monte Carlo simulation.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    tau (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (OptionType): OptionType.CALL for a call option, OptionType.PUT for a put option
    num_simulations (int): Number of Monte Carlo simulations
    seed (int, optional): Random seed for reproducibility

    Returns:
    float: The calculated option price
    
    Raises:
    ValueError: If option_type is not OptionType.CALL or OptionType.PUT
    """
    if seed is not None:
        np.random.seed(seed)
    
    _, S = geometric_brownian_motion(S0=S, tau=tau, dt=tau, r=r, sigma=sigma, M=num_simulations)
    ST = S[-1, :]
    
    if option_type == OptionType.CALL:
        payoffs = np.maximum(ST - K, 0)
    elif option_type == OptionType.PUT:
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")
    
    option_price = np.exp(-r * tau) * np.mean(payoffs)
    
    return option_price

def black_scholes_vega(S, K, r, sigma, tau):
    """
    Calculate the vega (sensitivity to volatility) of a Black-Scholes option.
    
    Vega is the same for both call and put options.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    tau (float): Time to expiration (in years)

    Returns:
    float: The vega of the option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return S * np.sqrt(tau) * norm.pdf(d1)

def implied_vol_newton_raphson(price_mkt: float, S0: float, K: float, tau: float, 
                               r: float = 0.1, sigma_0: float = 0.25, 
                               option_type: OptionType = OptionType.CALL, 
                               tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Calculate the implied volatility using Newton-Raphson method.

    Parameters:
    price_mkt (float): Market price of the option
    S0 (float): Current stock price
    K (float): Strike price
    tau (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma_0 (float): Initial volatility guess
    option_type (OptionType): OptionType.CALL for a call option, OptionType.PUT for a put option
    tol (float): Tolerance for convergence
    max_iter (int): Maximum number of iterations

    Returns:
    float: The implied volatility
    
    Raises:
    RuntimeError: If Newton-Raphson does not converge or vega becomes too small
    """
    
    # Initialize
    sigma = sigma_0
    
    for iteration in range(max_iter):
        price_theo = black_scholes(S0, K, tau, r, sigma, option_type)
        
        diff = price_theo - price_mkt
        
        if abs(diff) < tol:
            return sigma
        
        vega = black_scholes_vega(S0, K, r, sigma, tau)
        
        if vega < 1e-8:
            raise RuntimeError(f"Vega too small ({vega:.2e}) at iteration {iteration}")
        
        sigma_new = sigma - diff / vega
        
        if sigma_new <= 0:
            sigma_new = sigma / 2  # Halve the volatility instead of going negative
        
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        
        sigma = sigma_new
    
    raise RuntimeError(f"Newton-Raphson did not converge after {max_iter} iterations. Last sigma: {sigma:.6f}")

def implied_vol_bissection(price_mkt: float, S0: float, K: float, tau: float, 
                          r: float = 0.1, vol_low: float = 1e-8, vol_high: float = 7, 
                          option_type: OptionType = OptionType.CALL, 
                          tol: float = 1e-6, max_iter: int = 200) -> float:
    """
    Calculate the implied volatility using the bisection method.
    
    Parameters:
    price_mkt (float): Market price of the option
    S0 (float): Current stock price
    K (float): Strike price
    tau (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    vol_low (float): Lower bound for volatility search
    vol_high (float): Upper bound for volatility search
    option_type (OptionType): OptionType.CALL for a call option, OptionType.PUT for a put option
    tol (float): Tolerance for convergence
    max_iter (int): Maximum number of iterations
    
    Returns:
    float: The implied volatility, or np.nan if convergence fails
    """
    for _ in range(max_iter):
        vol_mid = 0.5 * (vol_low + vol_high)
        price_mid = black_scholes(S0, K, tau, r, vol_mid, option_type)

        if abs(price_mid - price_mkt) < tol:
            return vol_mid

        if price_mid > price_mkt:
            vol_high = vol_mid
        else:
            vol_low = vol_mid

    return np.nan

def implied_vol(price_mkt: float, S0: float, K: float, tau: float, 
               r: float = 0.1, vol_low: float = 1e-8, vol_high: float = 5, 
               option_type: OptionType = OptionType.CALL, 
               tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Calculate the implied volatility using hybrid approach.
    
    First attempts Newton-Raphson method for faster convergence.
    Falls back to bisection method if Newton-Raphson fails.
    
    Parameters:
    price_mkt (float): Market price of the option
    S0 (float): Current stock price
    K (float): Strike price
    tau (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    vol_low (float): Lower bound for volatility (used in bisection fallback)
    vol_high (float): Upper bound for volatility (used in bisection fallback)
    option_type (OptionType): OptionType.CALL for a call option, OptionType.PUT for a put option
    tol (float): Tolerance for convergence
    max_iter (int): Maximum number of iterations
    
    Returns:
    float: The implied volatility, or np.nan if both methods fail
    """
    try:
        return implied_vol_newton_raphson(price_mkt, S0, K, tau, r, sigma_0=0.2, 
                                         option_type=option_type, tol=tol, max_iter=max_iter)
    except RuntimeError:
        return implied_vol_bissection(price_mkt, S0, K, tau, r, vol_low, vol_high, 
                                     option_type, tol, max_iter)

if __name__ == "__main__":
    S0 = 100
    sigma = 0.16
    r = 0.1
    tau = 1.0
    K = 100

    BSM_call = black_scholes(S0, K, tau, r, sigma, OptionType.CALL)
    BSM_put = black_scholes(S0, K, tau, r, sigma, OptionType.PUT)

    print(f'BSM Call Option Price: {BSM_call:.2f}')
    print(f'BSM Put Option Price: {BSM_put:.2f}')
