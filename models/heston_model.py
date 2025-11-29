import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
from brownian_motion import cov_brownian_motion_diff
from utils import OptionType
from scipy.integrate import quad

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
    _, dW = cov_brownian_motion_diff(T, dt, rho, 2, M)
    
    S = np.full(shape=(N+1, M), fill_value=float(S0))
    v = np.full(shape=(N+1, M), fill_value=float(v0))
    
    for i in range(1, N+1):
        v_pos = np.maximum(v[i-1], 0)

        S[i] = S[i-1] * np.exp((r - 0.5*v_pos)*dt + np.sqrt(v_pos) * dW[i-1, :, 0])
        v[i] = np.maximum(v[i-1] + kappa*(theta - v_pos)*dt + sigma*np.sqrt(v_pos)*dW[i-1, :, 1], 0)
    
    return t, S, v

def heston_option_price_mc(S0: float, K: float, v0: float, rho: float, kappa: float, 
                          theta: float, sigma: float, r: float, lambd:float, T: float, dt: float, 
                          M: int, option_type: OptionType = OptionType.CALL) -> float:
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
    option_type (OptionType): 'CALL' or 'PUT'
    
    Returns:
    float: option price
    """
    kappa2 = kappa + lambd
    theta2 = (kappa * theta) / kappa2
    _, S, _ = heston_model(S0, v0, rho, kappa2, theta2, sigma, r, T, dt, M)
    S_T = S[-1, :]  # Final stock prices
    
    # Calculate payoff
    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == OptionType.PUT:
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Discount expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


def characteristic_function(phi: complex, S0: float, v0: float, kappa: float, theta: float, 
                           sigma: float, rho: float, lambd: float, T: float, r: float) -> complex:
    """
    Calculate the characteristic function for the Heston model.
    
    The characteristic function is used in the semi-analytical pricing formula
    for European options under the Heston stochastic volatility model.
    
    Parameters:
    phi (complex): frequency parameter in the Fourier transform
    S0 (float): initial stock price
    v0 (float): initial variance
    kappa (float): mean reversion speed
    theta (float): long-term variance mean
    sigma (float): volatility of variance (vol of vol)
    rho (float): correlation between stock and variance Brownian motions
    lambd (float): risk premium parameter
    T (float): time to maturity
    r (float): risk-free rate
    
    Returns:
    complex: value of the characteristic function at phi
    """
    a = kappa * theta
    b = kappa + lambd
    rspi = rho * sigma * phi * 1j

    d = np.sqrt((rho * sigma * phi * 1j - b)**2 + (phi * 1j + phi**2) * sigma**2)
    g = (b - rspi + d)/(b - rspi - d)

    exp1 = np.exp(r * phi * 1j * T)
    term2 = S0**(phi * 1j) * ((1 - g*np.exp(d*T))/(1 - g))**(-2*a/sigma**2)
    exp2 = np.exp(a*T*(b - rspi + d)/sigma**2 + v0*(b - rspi + d)*((1 - np.exp(d*T))/(1 - g*np.exp(d*T)))/sigma**2)

    return exp1 * term2 * exp2

def integrand(phi: float, S0: float, v0: float, K: float, kappa: float, theta: float, 
             sigma: float, rho: float, lambd: float, tau: float, r: float) -> complex:
    """
    Calculate the integrand for the Heston option pricing formula.
    
    This function computes the integrand used in the Fourier inversion
    to obtain the option price from the characteristic function.
    
    Parameters:
    phi (float): integration variable (frequency)
    S0 (float): initial stock price
    v0 (float): initial variance
    K (float): strike price
    kappa (float): mean reversion speed
    theta (float): long-term variance mean
    sigma (float): volatility of variance (vol of vol)
    rho (float): correlation between stock and variance Brownian motions
    lambd (float): risk premium parameter
    tau (float): time to maturity
    r (float): risk-free rate
    
    Returns:
    complex: value of the integrand at phi
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r*tau)*characteristic_function(phi-1j, *args) - K*characteristic_function(phi, *args)
    denominator = 1j*phi*K**(1j*phi)
    return numerator/denominator

def heston_price(S0: float, K: float, v0: float, kappa: float, theta: float, 
                sigma: float, rho: float, lambd: float, tau: float, r: float) -> float:
    """
    Calculate European call option price using the Heston model semi-analytical formula.
    
    This function uses Fourier inversion to compute the option price from the
    characteristic function of the Heston model. The implementation follows the
    original Heston (1993) formulation.
    
    Parameters:
    S0 (float): initial stock price
    K (float): strike price
    v0 (float): initial variance
    kappa (float): mean reversion speed
    theta (float): long-term variance mean
    sigma (float): volatility of variance (vol of vol)
    rho (float): correlation between stock and variance Brownian motions
    lambd (float): risk premium parameter
    tau (float): time to maturity
    r (float): risk-free rate
    
    Returns:
    float: European call option price
    
    Notes:
    - This formula is valid for European call options
    - Put prices can be obtained via put-call parity
    - The integration is performed from 0 to 100 (approximating infinity)
    
    References:
    Heston, S. L. (1993). "A Closed-Form Solution for Options with 
    Stochastic Volatility with Applications to Bond and Currency Options"
    """
    args = (S0, v0, K, kappa, theta, sigma, rho, lambd, tau, r)

    # real_integral, err = np.real( quad(integrand, 0, 100, args=args))
    real_integral, err = quad(lambda phi: np.real(integrand(phi, *args)), 0, 100)

    return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

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

def test_heston_option_pricing_mc() -> None:
    """Test Heston option pricing."""
    S0 = 100
    K = 100
    v0 = 0.25
    rho = -0.5711
    kappa = 1.5768
    theta = 0.0398
    sigma = 0.3
    lambd = 0.575
    r = 0.06
    T = 1.0
    dt = 0.001
    M = 100_000
    
    call_price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, lambd, T, dt, M, OptionType.CALL)
    # put_price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, lambd, T, dt, M, OptionType.CALL)
    
    print(f"Call option price: {call_price:.4f}")
    # print(f"Put option price: {put_price:.4f}")
    
    # Verify put-call parity
    # pv_strike = K * np.exp(-r * T)
    # parity_diff = call_price - put_price - (S0 - pv_strike)
    # print(f"Put-call parity difference: {parity_diff:.4f}")

def test_heston_option_pricing() -> None:
    """Test Heston option pricing."""
    S0 = 100
    K = 100
    v0 = 0.1
    rho = -0.5711
    kappa = 1.5768
    theta = 0.0398
    sigma = 0.3
    lambd = 0.575
    r = 0.03
    T = 1.0
    
    call_price = heston_price( S0, K, v0, kappa, theta, sigma, rho, lambd, T, r )
    
    print(f"Call option price: {call_price:.4f}")


if __name__ == "__main__":
    # test_heston_model()
    # test_heston_option_pricing_mc()
    test_heston_option_pricing()

