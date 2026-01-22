import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
from brownian_motion import cov_brownian_motion_diff
from utils import OptionType, measure, load_params
from scipy.integrate import quad
import QuantLib as ql
import pandas as pd

def heston_model(S0: float, v0: float, rho: float, kappa: float, theta: float, 
                 sigma: float, r: float, tau: float, dt: float, M: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    tau (float): time horizon
    dt (float): time step
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: (time grid, stock prices, variance paths)
    """
    N = int(tau / dt)
    t = np.linspace(0, tau, N + 1)
    
    # Generate correlated Brownian motions
    _, dW = cov_brownian_motion_diff(tau, dt, rho, 2, M)
    
    S = np.full(shape=(N+1, M), fill_value=float(S0))
    v = np.full(shape=(N+1, M), fill_value=float(v0))
    
    for i in range(1, N+1):
        v_pos = np.maximum(v[i-1], 0)

        S[i] = S[i-1] * np.exp((r - 0.5*v_pos)*dt + np.sqrt(v_pos) * dW[i-1, :, 0])
        v[i] = np.maximum(v[i-1] + kappa*(theta - v_pos)*dt + sigma*np.sqrt(v_pos)*dW[i-1, :, 1], 0)
    print(np.mean(S[-1]) / S0)
    print(np.exp(r * tau))
    return t, S, v

def heston_option_price_mc(S0: float, K: float, v0: float, rho: float, kappa: float, 
                          theta: float, sigma: float, r: float, tau: float, dt: float, 
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
    tau (float): time to maturity
    dt (float): time step
    M (int): number of Monte Carlo simulations
    option_type (OptionType): 'CALL' or 'PUT'
    
    Returns:
    float: option price
    """
    kappa2 = kappa
    theta2 = (kappa * theta) / kappa2
    _, S, _ = heston_model(S0, v0, rho, kappa2, theta2, sigma, r, tau, dt, M)
    S_T = S[-1, :]  # Final stock prices
    
    # Calculate payoff
    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == OptionType.PUT:
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Discount expected payoff
    option_price = np.exp(-r * tau) * np.mean(payoffs)
    return option_price


def characteristic_function(phi: complex, S0: float, v0: float, kappa: float, theta: float, 
                           sigma: float, rho: float, tau: float, r: float, j: int) -> complex:
    """
    Calculate the characteristic function for the Heston model.
    
    The characteristic function is used in the semi-analytical pricing formula
    for European options under the Heston stochastic volatility model.
    
    This implementation uses the "Little Trap" formulation for numerical stability.
    
    Parameters:
    phi (complex): frequency parameter in the Fourier transform
    S0 (float): initial stock price
    v0 (float): initial variance
    kappa (float): mean reversion speed
    theta (float): long-term variance mean
    sigma (float): volatility of variance (vol of vol)
    rho (float): correlation between stock and variance Brownian motions
    tau (float): time to maturity
    r (float): risk-free rate
    j (int): 1 or 2, determines the form of the characteristic function

    Returns:
    complex: value of the characteristic function at phi
    """
    # Parameters
    a = kappa * theta
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    sigma_2 = sigma**2
    rspi = rho * sigma * phi * 1j
    d = np.sqrt((rspi - b)**2 - sigma_2 * (2 * u * phi * 1j - phi**2))
    if np.real(d) < 0:
        d = -d
    g = (b - rspi - d)/(b - rspi + d)
    exp_dT = np.exp(-d*tau)

    C = r * phi * 1j * tau + (a / sigma_2) * ((b - rspi - d)*tau - 2*np.log((1 - g*exp_dT)/(1 - g)))

    D = ((b - rspi - d)/sigma_2) * ((1 - exp_dT)/(1 - g*exp_dT))
    
    return np.exp(C + D*v0 + 1j * phi * np.log(S0))

def integrand(phi: float, S0: float, v0: float, K: float, kappa: float, theta: float, 
             sigma: float, rho: float, tau: float, r: float, j: int) -> float:
    """
    Calculate the integrand for the Heston option pricing formula.
    
    Parameters:
    phi (float): integration variable (frequency)
    S0 (float): initial stock price
    v0 (float): initial variance
    K (float): strike price
    kappa (float): mean reversion speed
    theta (float): long-term variance mean
    sigma (float): volatility of variance (vol of vol)
    rho (float): correlation between stock and variance Brownian motions
    tau (float): time to maturity
    r (float): risk-free rate
    j (int): 1 or 2, determines which probability measure

    Returns:
    float: real part of the integrand at phi
    """
    if abs(phi) < 1e-10:
        return 0.0 + 0j
    
    char_val = characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, tau, r, j)
    
    # Calculate integrand
    numerator = np.exp(-1j * phi * np.log(K)) * char_val
    denominator = 1j * phi
    
    result = numerator / denominator

    if not np.isfinite(result):
        return 0.0 + 0j
    
    return np.real(result)

def heston_price(S0: float, K: float, v0: float, kappa: float, theta: float, 
                sigma: float, rho: float, tau: float, r: float) -> float:
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
    args = (S0, v0, K, kappa, theta, sigma, rho, tau, r)
    # print({'S0': S0, 'v0': v0, 'K': K, 'kappa': kappa, 'theta': theta, 'sigma': sigma, 'rho': rho, 'tau': tau, 'r': r})
    # real_integral, err = np.real( quad(integrand, 0, 100, args=args))
    phi_max = 100
    integration1, _ = quad(integrand, 0, phi_max, args=(*args, 1), limit=500, epsabs=1e-8, epsrel=1e-8)
    integration2, _ = quad(integrand, 0, phi_max, args=(*args, 2), limit=500, epsabs=1e-8, epsrel=1e-8)

    P1 = 0.5 + (1/np.pi) * integration1
    P2 = 0.5 + (1/np.pi) * integration2
    price = S0 * P1 - K * np.exp(-r * tau) * P2
    if price < 0.0:
        price = 0.0
    return price

def heston_price_ql(S0: float, K: float, v0: float, kappa: float, theta: float, 
                sigma: float, rho: float, tau: float, r: float) -> float:
    """Calculate European call option price using QuantLib's Heston model implementation."""  
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity_date = today + ql.Period(int(365 * tau), ql.Days)
    exercise = ql.EuropeanExercise(maturity_date)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.EuropeanOption(payoff, exercise)
    risk_free_curve = ql.FlatForward(today, r, ql.Actual365Fixed())
    dividend_curve = ql.FlatForward(today, 0, ql.Actual365Fixed())
    risk_free_handle = ql.YieldTermStructureHandle(risk_free_curve)
    dividend_handle = ql.YieldTermStructureHandle(dividend_curve)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    heston_process = ql.HestonProcess(
        risk_free_handle,
        dividend_handle,
        spot_handle,
        v0,      # initial variance
        kappa,   # mean reversion speed
        theta,   # long-term variance
        sigma,   # vol of vol
        rho      # correlation
    )
    
    # Heston model
    heston_model = ql.HestonModel(heston_process)
    heston_engine = ql.AnalyticHestonEngine(heston_model)
    option.setPricingEngine(heston_engine)
    option_price = option.NPV()

    return option_price

def test_heston_model() -> None:
    """Test Heston model visualization."""
    S0 = 100
    v0 = 0.25**2
    rho = -0.7
    kappa = 3.0
    theta = 0.04
    sigma = 0.6
    tau = 1.0
    dt = 0.001
    M = 5
    r = 0.02

    t, S, v = heston_model(S0=S0, v0=v0, rho=rho, kappa=kappa, theta=theta, 
                          sigma=sigma, r=r, tau=tau, dt=dt, M=M)
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
    S0 = 35
    K = 25
    # v0 = 0.1
    # rho = -0.5711
    # kappa = 1.5768
    # theta = 0.0398
    # sigma = 0.3
    r = 0.03
    tau = 1.0
    dt = 0.001
    M = 100_000
    v0 = 0.0522284576620002
    kappa=0.4054180835596147
    theta=0.22112599883353512
    sigma=0.4999999999999986
    rho=-0.42944183484265186

    call_price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, tau, dt, M, OptionType.CALL)
    # put_price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, tau, dt, M, OptionType.CALL)
    
    print(f"Call option price: {call_price:.4f}")
    # print(f"Put option price: {put_price:.4f}")
    
    # Verify put-call parity
    # pv_strike = K * np.exp(-r * tau)
    # parity_diff = call_price - put_price - (S0 - pv_strike)
    # print(f"Put-call parity difference: {parity_diff:.4f}")

def test_heston_option_pricing() -> None:
    """Test Heston option pricing."""
    S0 = 35
    K = 25
    # v0 = 0.1
    # rho = -0.2
    # kappa = 0.5
    # theta = 0.01
    # sigma = 0.13
    r = 0.1
    tau = 1.0
    v0 = 0.0522284576620002
    kappa=0.4054180835596147
    theta=0.22112599883353512
    sigma=0.4999999999999986
    rho=-0.42944183484265186

    
    call_price = heston_price( S0, K, v0, kappa, theta, sigma, rho, tau, r )
    
    print(f"Call option price: {call_price:.4f}")

    call_price_ql = heston_price_ql( S0, K, v0, kappa, theta, sigma, rho, tau, r )
    print(f"Call option price ql: {call_price_ql:.4f}")

if __name__ == "__main__":
    # test_heston_model()
    measure(test_heston_option_pricing)
    measure(test_heston_option_pricing_mc)
    
