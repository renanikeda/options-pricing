import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple

def heston_model(S0: float, v0: float, rho: float, kappa: float, theta: float, 
                 sigma: float, r: float, T: float, dt: float, M: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the Heston stochastic volatility model.
    
    Parameters:
    S0 (float): initial stock price
    v0 (float): initial variance
    rho (float): correlation between stock and variance Brownian motions
    kappa (float): mean reversion speed
    theta (float): long-term variance mean
    sigma (float): volatility of variance (vol of vol)
    r (float): risk-free rate
    T (float): time horizon
    dt (float): time step
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: (time grid, stock prices, variance paths)
    """
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    
    # Generate correlated Brownian motions
    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])
    Z = np.random.multivariate_normal(mu, cov, (N, M))
    
    S = np.full(shape=(N+1, M), fill_value=float(S0))
    v = np.full(shape=(N+1, M), fill_value=float(v0))
    
    for i in range(1, N+1):
        v_pos = np.maximum(v[i-1], 0)

        S[i] = S[i-1] * np.exp((r - 0.5*v_pos)*dt + np.sqrt(v_pos * dt) * Z[i-1, :, 0])
        v[i] = np.maximum(v[i-1] + kappa*(theta - v_pos)*dt + sigma*np.sqrt(v_pos*dt)*Z[i-1, :, 1], 0)
    
    return t, S, v

def heston_option_price_mc(S0: float, K: float, v0: float, rho: float, kappa: float, 
                          theta: float, sigma: float, r: float, T: float, dt: float, 
                          M: int, option_type: str = 'call') -> float:
    """
    Price European option using Monte Carlo simulation with Heston model.
    
    Parameters:
    S0 (float): initial stock price
    K (float): strike price
    v0 (float): initial variance
    rho (float): correlation between stock and variance Brownian motions
    kappa (float): mean reversion speed
    theta (float): long-term variance mean
    sigma (float): volatility of variance
    r (float): risk-free rate
    T (float): time to maturity
    dt (float): time step
    M (int): number of Monte Carlo simulations
    option_type (str): 'call' or 'put'
    
    Returns:
    float: option price
    """
    _, S, _ = heston_model(S0, v0, rho, kappa, theta, sigma, r, T, dt, M)
    S_T = S[-1, :]  # Final stock prices
    
    # Calculate payoff
    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Discount expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def test_heston_model() -> None:
    """Test Heston model visualization."""
    S0 = 100
    v0 = 0.25**2
    rho = -0.7
    kappa = 3.0
    theta = 0.04
    sigma = 0.6
    T = 1.0
    dt = 0.001
    M = 5
    r = 0.02

    t, S, v = heston_model(S0=S0, v0=v0, rho=rho, kappa=kappa, theta=theta, 
                          sigma=sigma, r=r, T=T, dt=dt, M=M)
    risk_free_rate = np.exp(r * t) * S0

    # Plot stock price paths
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, S, '-', linewidth=1)
    plt.plot(t, risk_free_rate, 'k--', linewidth=2, label='Risk-free rate')
    plt.title('Heston Model - Stock Price Paths')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot variance paths
    plt.subplot(2, 1, 2)
    plt.plot(t, v[:, :5], '-', linewidth=1)
    plt.axhline(y=theta, color='k', linestyle='--', linewidth=2, label=f'Long-term var (θ={theta})')
    plt.title('Heston Model - Variance Paths')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_heston_option_pricing() -> None:
    """Test Heston option pricing."""
    S0 = 100
    K = 100
    v0 = 0.04
    rho = -0.7
    kappa = 3.0
    theta = 0.04
    sigma = 0.6
    r = 0.05
    T = 1.0
    dt = 0.001
    M = 100_000
    
    call_price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, T, dt, M, 'call')
    put_price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, T, dt, M, 'put')
    
    print(f"Call option price: {call_price:.4f}")
    print(f"Put option price: {put_price:.4f}")
    
    # Verify put-call parity
    pv_strike = K * np.exp(-r * T)
    parity_diff = call_price - put_price - (S0 - pv_strike)
    print(f"Put-call parity difference: {parity_diff:.4f}")

if __name__ == "__main__":
    test_heston_model()
    # test_heston_option_pricing()

