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
    Simulate the Heston stochastic volatility model using Euler discretization.
    
    The Heston model describes the evolution of stock price and variance as:
    dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW1_t
    dv_t = kappa*(theta - v_t)*dt + sigma*sqrt(v_t)*dW2_t
    where dW1_t and dW2_t are correlated Brownian motions with correlation rho.
    
    Parameters:
    S0 (float): Initial stock price
    v0 (float): Initial variance
    rho (float): Correlation between stock and variance Brownian motions (range: [-1, 1])
    kappa (float): Mean reversion speed (kappa > 0)
    theta (float): Long-term variance mean (theta > 0)
    sigma (float): Volatility of variance (vol of vol, sigma > 0)
    r (float): Risk-free interest rate (annualized)
    tau (float): Time horizon (in years)
    dt (float): Time step size
    M (int): Number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        - t: Time grid of shape (N+1,)
        - S: Stock price paths of shape (N+1, M)
        - v: Variance paths of shape (N+1, M)
    
    Notes:
    - Uses Euler discretization with full truncation scheme for variance
    - Variance is floored at zero to prevent negative values
    - Feller condition (2*kappa*theta > sigma²) ensures variance stays positive
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
    
    return t, S, v

def heston_option_price_mc(S0: float, K: float, v0: float, rho: float, kappa: float, 
                          theta: float, sigma: float, r: float, tau: float, dt: float, 
                          M: int, option_type: OptionType = OptionType.CALL) -> float:
    """
    Price European option using Monte Carlo simulation with the Heston model.
    
    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    v0 (float): Initial variance
    rho (float): Correlation between stock and variance Brownian motions
    kappa (float): Mean reversion speed
    theta (float): Long-term variance mean
    sigma (float): Volatility of variance (vol of vol)
    r (float): Risk-free interest rate (annualized)
    tau (float): Time to maturity (in years)
    dt (float): Time step size for simulation
    M (int): Number of Monte Carlo paths
    option_type (OptionType): OptionType.CALL or OptionType.PUT
    
    Returns:
    float: European option price
    
    Raises:
    ValueError: If option_type is not OptionType.CALL or OptionType.PUT
    
    Notes:
    - Uses risk-neutral valuation: E[e^(-rT) * payoff]
    - Standard error decreases as O(1/sqrt(M))
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
        raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")
    
    # Discount expected payoff
    option_price = np.exp(-r * tau) * np.mean(payoffs)
    return option_price


def characteristic_function(phi: complex, S0: float, v0: float, kappa: float, theta: float, 
                           sigma: float, rho: float, tau: float, r: float, j: int) -> complex:
    """
    Calculate the characteristic function for the Heston model.
    
    The characteristic function is the Fourier transform of the probability density
    function and is used in the semi-analytical pricing formula for European options.
    
    This implementation uses the "Little Trap" formulation for numerical stability,
    which ensures the correct branch cut in the complex logarithm.
    
    Parameters:
    phi (complex): Frequency parameter in the Fourier transform
    S0 (float): Initial stock price
    v0 (float): Initial variance
    kappa (float): Mean reversion speed
    theta (float): Long-term variance mean
    sigma (float): Volatility of variance (vol of vol)
    rho (float): Correlation between stock and variance Brownian motions
    tau (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annualized)
    j (int): Index determining the probability measure (1 or 2)
        - j=1: First probability P1 (stock measure)
        - j=2: Second probability P2 (money market measure)

    Returns:
    complex: Value of the characteristic function at frequency phi
    
    Notes:
    - The branch cut is chosen such that Re(d) >= 0 for numerical stability
    - For j=1: u=0.5, b=kappa-rho*sigma (stock measure)
    - For j=2: u=-0.5, b=kappa (money market measure)
    
    References:
    Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
    "The Little Heston Trap". Wilmott Magazine.
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
    
    # Choose branch cut: Re(d) >= 0
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
    
    This function computes the real part of the integrand used in the Fourier
    inversion formula for European call option pricing under the Heston model.
    
    Parameters:
    phi (float): Integration variable (frequency parameter)
    S0 (float): Initial stock price
    v0 (float): Initial variance
    K (float): Strike price
    kappa (float): Mean reversion speed
    theta (float): Long-term variance mean
    sigma (float): Volatility of variance (vol of vol)
    rho (float): Correlation between stock and variance Brownian motions
    tau (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annualized)
    j (int): Probability measure index (1 or 2)

    Returns:
    float: Real part of the integrand at frequency phi
    
    Notes:
    - Returns 0.0 if phi is near zero (to avoid division by zero)
    - Returns 0.0 if result is not finite (numerical stability)
    - The integrand is: Re[exp(-i*phi*ln(K)) * char_func(phi) / (i*phi)]
    """
    if abs(phi) < 1e-10:
        return 0.0
    
    char_val = characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, tau, r, j)
    
    # Calculate integrand: exp(-i*phi*ln(K)) * char_func / (i*phi)
    numerator = np.exp(-1j * phi * np.log(K)) * char_val
    denominator = 1j * phi
    
    result = numerator / denominator

    if not np.isfinite(result):
        return 0.0
    
    return np.real(result)

def heston_price(S0: float, K: float, v0: float, kappa: float, theta: float, 
                sigma: float, rho: float, tau: float, r: float, **kwargs) -> float:
    """
    Calculate European call option price using the Heston model semi-analytical formula.
    
    This function uses Fourier inversion to compute the option price from the
    characteristic function of the Heston stochastic volatility model. The 
    implementation follows the original Heston (1993) formulation:
    
    C(S, K, T) = S*P1 - K*exp(-rT)*P2
    
    where P1 and P2 are probabilities computed via Fourier inversion.
    
    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    v0 (float): Initial variance (must satisfy Feller condition with kappa, theta, sigma)
    kappa (float): Mean reversion speed (kappa > 0)
    theta (float): Long-term variance mean (theta > 0)
    sigma (float): Volatility of variance (vol of vol, sigma > 0)
    rho (float): Correlation between stock and variance Brownian motions (range: [-1, 1])
    tau (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annualized)
    **kwargs: Additional keyword arguments (ignored, for compatibility)
    
    Returns:
    float: European call option price (non-negative)
    
    Notes:
    - This formula is valid for European call options only
    - Put prices can be obtained via put-call parity: P = C - S + K*exp(-rT)
    - Integration is performed from 0 to 100 (approximating infinite upper limit)
    - Returns 0.0 if computed price is negative (numerical artifact)
    - Feller condition: 2*kappa*theta > sigma² ensures variance stays positive
    
    Raises:
    May raise integration errors if parameters lead to numerical instability
    
    References:
    Heston, S. L. (1993). "A Closed-Form Solution for Options with 
    Stochastic Volatility with Applications to Bond and Currency Options".
    The Review of Financial Studies, 6(2), 327-343.
    """
    args = (S0, v0, K, kappa, theta, sigma, rho, tau, r)
    
    phi_max = 100  # Upper integration limit (approximates infinity)
    
    # Compute two integrals for probabilities P1 and P2
    integration1, _ = quad(integrand, 0, phi_max, args=(*args, 1), 
                          limit=500, epsabs=1e-8, epsrel=1e-8)
    integration2, _ = quad(integrand, 0, phi_max, args=(*args, 2), 
                          limit=500, epsabs=1e-8, epsrel=1e-8)

    # Calculate probabilities via Fourier inversion
    P1 = 0.5 + (1/np.pi) * integration1
    P2 = 0.5 + (1/np.pi) * integration2
    
    # Heston call option formula
    price = S0 * P1 - K * np.exp(-r * tau) * P2
    
    # Ensure non-negative price
    if price < 0.0:
        price = 0.0
        
    return price

def heston_price_ql(S0: float, K: float, v0: float, kappa: float, theta: float, 
                    sigma: float, rho: float, tau: float, r: float) -> float:
    """
    Calculate European call option price using QuantLib's Heston model implementation.
    
    This function serves as a benchmark to validate the custom implementation
    against the industry-standard QuantLib library.
    
    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    v0 (float): Initial variance
    kappa (float): Mean reversion speed
    theta (float): Long-term variance mean
    sigma (float): Volatility of variance (vol of vol)
    rho (float): Correlation between stock and variance Brownian motions
    tau (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annualized)
    
    Returns:
    float: European call option price computed by QuantLib
    
    Notes:
    - Uses QuantLib's AnalyticHestonEngine for pricing
    - Assumes zero dividend yield
    - Evaluation date is set to today
    - Maturity is calculated as today + tau*365 days
    
    Requires:
    - QuantLib Python bindings installed
    """
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity_date = today + ql.Period(int(365 * tau), ql.Days)
    
    # Define European call option
    exercise = ql.EuropeanExercise(maturity_date)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.EuropeanOption(payoff, exercise)
    
    # Setup market curves
    risk_free_curve = ql.FlatForward(today, r, ql.Actual365Fixed())
    dividend_curve = ql.FlatForward(today, 0.0, ql.Actual365Fixed())
    risk_free_handle = ql.YieldTermStructureHandle(risk_free_curve)
    dividend_handle = ql.YieldTermStructureHandle(dividend_curve)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    
    # Define Heston process
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
    
    # Create Heston model and pricing engine
    heston_model = ql.HestonModel(heston_process)
    heston_engine = ql.AnalyticHestonEngine(heston_model)
    option.setPricingEngine(heston_engine)
    
    option_price = option.NPV()
    return option_price

def test_heston_model() -> None:
    """
    Test and visualize the Heston stochastic volatility model.
    
    Creates two subplots:
    1. Stock price paths with risk-free growth curve
    2. Variance paths with long-term mean level
    
    Uses sample parameters to demonstrate typical model behavior.
    """
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
    """
    Test Heston option pricing using Monte Carlo simulation.
    
    Prices a European call option and displays the result.
    Uses calibrated parameters as an example.
    """
    S0 = 35
    K = 25
    r = 0.03
    tau = 1.0
    dt = 0.001
    M = 100_000
    
    # Sample calibrated parameters
    v0 = 0.0522284576620002
    kappa = 0.4054180835596147
    theta = 0.22112599883353512
    sigma = 0.4999999999999986
    rho = -0.42944183484265186

    call_price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, 
                                       r, tau, dt, M, OptionType.CALL)
    
    print(f"Call option price (Monte Carlo): {call_price:.4f}")

def test_heston_option_pricing() -> None:
    """
    Test Heston option pricing using both semi-analytical and QuantLib methods.
    
    Compares the custom implementation against QuantLib for validation.
    """
    S0 = 35
    K = 25
    r = 0.1
    tau = 1.0
    
    # Sample calibrated parameters
    v0 = 0.0522284576620002
    kappa = 0.4054180835596147
    theta = 0.22112599883353512
    sigma = 0.4999999999999986
    rho = -0.42944183484265186
    
    call_price = heston_price(S0, K, v0, kappa, theta, sigma, rho, tau, r)
    print(f"Call option price (Semi-analytical): {call_price:.4f}")

    call_price_ql = heston_price_ql(S0, K, v0, kappa, theta, sigma, rho, tau, r)
    print(f"Call option price (QuantLib):       {call_price_ql:.4f}")
    print(f"Difference:                          {abs(call_price - call_price_ql):.6f}")

if __name__ == "__main__":
    # test_heston_model()
    measure(test_heston_option_pricing)
    measure(test_heston_option_pricing_mc)

