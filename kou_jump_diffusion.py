from utils import OptionType, colors
from brownian_motion import brownian_motion_diff, brownian_motion
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import binom
import numpy as np
import math
from typing import Tuple

def poisson_process(lambd: float, T: float, dt: float, M: int = 1) -> np.ndarray:
    """
    Poisson process.
    
    Parameters:
    lambd (float): intensity parameter
    T (float): time horizon
    dt (float): step discretization
    M (int): number of paths
    
    Returns:
    np.ndarray: array of jump counts
    """
    N = int(T / dt)
    jumps = np.random.poisson(lambd * dt, (N + 1, M))
    return jumps

def generate_jump_sizes(p: float, eta1: float, eta2: float, num_jumps: int) -> np.ndarray:
    """
    Sample n independent draws of Y ~ Kou double-exponential (log-jump).
    Return array of length n.
    Implementation:
      - with probability p: Y = +Exp(scale=1/eta1) (positive)
      - with probability 1-p: Y = -Exp(scale=1/eta2) (negative)
    
    Parameters:
    p (float): probability of positive jump
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    num_jumps (int): number of jumps to generate
    
    Returns:
    np.ndarray: array of jump sizes
    """
    u = np.random.rand(num_jumps)
    pos_mask = (u < p)
    neg_mask = ~pos_mask
    Y = np.empty(num_jumps, dtype=float)
    # positive jumps
    npos = pos_mask.sum()
    if npos > 0:
        Y[pos_mask] = np.random.exponential(scale=1.0/eta1, size=npos)
    # negative jumps
    nneg = neg_mask.sum()
    if nneg > 0:
        Y[neg_mask] = -np.random.exponential(scale=1.0/eta2, size=nneg)
    return Y

def generate_matrix_jump_sizes(p: float, eta1: float, eta2: float, jumps: np.ndarray) -> np.ndarray:
    """
    Sample n independent draws of Y ~ Kou double-exponential (log-jump).
    Return array of length n.
    Implementation:
      - with probability p: Y = +Exp(scale=1/eta1) (positive)
      - with probability 1-p: Y = -Exp(scale=1/eta2) (negative)
    
    Parameters:
    p (float): probability of positive jump
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    jumps (np.ndarray): matrix of jump counts (N x M)
    
    Returns:
    np.ndarray: matrix of jump sizes
    """
    Y = np.empty(jumps.shape, dtype=float)
    for n in range(jumps.shape[1]):
        u = np.random.rand(jumps.shape[0])
        pos_mask = (u < p)
        neg_mask = ~pos_mask
        # positive jumps
        npos = pos_mask.sum()
        if npos > 0:
            Y[pos_mask, n] = np.random.exponential(scale=1.0/eta1, size=npos)
        # negative jumps
        nneg = neg_mask.sum()
        if nneg > 0:
            Y[neg_mask, n] = -np.random.exponential(scale=1.0/eta2, size=nneg)
        Y[:, n] *= jumps[:, n]
    return Y

def jumps_pdf(T: float, dt: float, p: float, eta1: float, eta2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the PDF of jump sizes for the double exponential distribution.
    
    Parameters:
    T (float): time horizon
    dt (float): time step
    p (float): probability of positive jump
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (y values, PDF values)
    """
    y = np.linspace(-math.floor(T/2), math.floor(T/2), math.floor(1/dt))
    fdp = np.zeros_like(y, dtype=float)
    
    # Positive part (y >= 0)
    positive_mask = y >= 0
    fdp[positive_mask] = p * eta1 * np.exp(-eta1 * y[positive_mask])
    
    # Negative part (y < 0)
    negative_mask = y < 0
    fdp[negative_mask] = (1-p) * eta2 * np.exp(eta2 * y[negative_mask])
    
    return y, fdp

def kou_process(S0: float, mu: float, sigma: float, T: float, dt: float, 
                eta1: float, eta2: float, p: float, lambd: float, M: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single path of the Kou jump diffusion process.
    
    Parameters:
    S0 (float): initial stock price
    mu (float): stock drift
    sigma (float): volatility
    T (float): time to maturity
    dt (float): time step
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    p (float): probability of positive jump
    lambd (float): jump intensity
    M (int): number of paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time_grid, stock_price_path)
    """
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    
    # Generate Brownian motion
    time_matrix = np.repeat(t, M).reshape(N+1, M)

    # Generate Poisson jumps
    poisson_jumps = poisson_process(lambd, T, dt, M)
    generated_jumps_matrix = generate_matrix_jump_sizes(p, eta1, eta2, poisson_jumps)

    log_S = np.zeros((N+1, M))
    t, W = brownian_motion(T, dt, M)
    log_S = np.log(S0) + mu*time_matrix + sigma*W + np.cumsum(generated_jumps_matrix, axis=0)

    return t, np.exp(log_S)

def kou_process_steps(S0: float, mu: float, sigma: float, T: float, dt: float,
                      eta1: float, eta2: float, p: float, lambd: float, M: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single path of the Kou jump diffusion process.
    
    Parameters:
    S0 (float): initial stock price
    mu (float): stock drift
    sigma (float): volatility
    T (float): time to maturity
    dt (float): time step
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    p (float): probability of positive jump
    lambd (float): jump intensity
    M (int): number of paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time_grid, stock_price_path)
    """
    N = int(T / dt)
    
    log_S = np.zeros((N + 1, M))
    log_S = np.full((N + 1, M), np.log(S0), dtype=float)
    
    t, dW = brownian_motion_diff(T, dt, M)
    jumps = np.random.poisson(lambd * dt, size=(N+1, M))

    for step in range(1, N + 1):
        Z = np.random.randn(M)
        dW = Z * np.sqrt(dt)
        log_S[step, :] = log_S[step-1, :] + (mu * dt) + (sigma * dW)
        Nj = jumps[step, :]
        idxs_with_jumps = np.nonzero(Nj)[0]
        if idxs_with_jumps.size > 0:
            for i in idxs_with_jumps:
                nj = Nj[i]
                Ys = generate_jump_sizes(p, eta1, eta2, nj)
                log_S[step, i] += Ys.sum()

    return t[:N], np.exp(log_S)[:N, :]

def kou_option_price_mc(S0: float, K: float, r: float, sigma: float, T: float, dt: float,
                        eta1: float, eta2: float, p: float, lambd: float, M: int, 
                        option_type: OptionType = OptionType.CALL) -> float:
    """
    Price European option using Monte Carlo simulation with Kou jump diffusion.
    
    Parameters:
    S0 (float): initial stock price
    K (float): strike price
    r (float): risk-free rate
    sigma (float): volatility
    T (float): time to maturity
    dt (float): time step
    eta1 (float): parameter for positive jumps (> 1)
    eta2 (float): parameter for negative jumps (> 0)
    p (float): probability of positive jump
    lambd (float): jump intensity
    M (int): number of Monte Carlo simulations
    option_type (OptionType): CALL or PUT
    
    Returns:
    float: option price
    """
    payoffs = np.zeros(M)
   
    # Generate stock price path
    csi = p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0) - 1.0
    mu_risk_neutral = r - 0.5*sigma**2 - lambd * csi

    _, S_path = kou_process(S0, mu_risk_neutral, sigma, T, dt, eta1, eta2, p, lambd, M)
    S_T = S_path[-1, :]  # Final stock price

    # Calculate payoff
    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == OptionType.PUT:
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")
    
    # Discount expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def phi(x: float) -> float:
    return norm.cdf(x)

def P(n: int, k: int, p: float, eta_1: float, eta_2: float) -> float:
    result = 0.0
    if k < 1 or n < 1: return 0
    for i in range(k, n):
        result += binom(n - k - 1, i - k) * binom(n, i) * (eta_1 / (eta_1 + eta_2))**(i - k) * (eta_2 / (eta_1 + eta_2))**(n - i) * (p ** i) * ((1 - p) ** (n - i))
    return result

def Q(n: int, k: int, p: float, eta_1: float, eta_2: float) -> float:
    result = 0.0
    if k < 1 or n < 1: return 0
    for i in range(k, n):
        result += binom(n - k - 1, i - k) * binom(n, i) * (eta_1 / (eta_1 + eta_2))**(n - i) * (eta_2 / (eta_1 + eta_2))**(i - k) * (p ** (n - i)) * ((1 - p) ** i)
    return result
 
def pi(n: int, lambd: float, T: float) -> float:
    return (np.exp(-lambd*T)*(lambd*T)**n)/math.factorial(n)

def Hh(n: int, x: float) -> float:
    if n<-1: return 0
    elif n==-1:
        return np.exp(-x**2/2)
    elif n==0:
        return math.sqrt(2*np.pi)*norm.cdf(-x)
    else:
        return (Hh(n-2,x)-x*Hh(n-1,x))/n

def I(n: int, c: int, alpha: int, beta: int, delta: int) -> float:
    if beta>0 and alpha!=0:
        suma=0
        i=0
        while i<=n:
            suma=suma+(beta/alpha)**(n-i)*Hh(i,beta*c-delta)
            i+=1
        return -(np.exp(alpha*c)/alpha)*suma+(beta/alpha)**(n+1)*(np.sqrt(2*np.pi)/beta)*np.exp((alpha*delta/beta)+(alpha**2/(2*beta**2)))*norm.cdf(-beta*c+delta+alpha/beta)
    elif beta<0 and alpha<0:
        suma=0
        i=0
        while i<=n:
            suma=suma+(beta/alpha)**(n-i)*Hh(i,beta*c-delta)
            i+=1
        return -(np.exp(alpha*c)/alpha)*suma-(beta/alpha)**(n+1)*(np.sqrt(2*np.pi)/beta)*np.exp((alpha*delta/beta)+(alpha**2/(2*beta**2)))*norm.cdf(beta*c-delta-alpha/beta)
    else: return 0

def Upsilon(x, T, mu, sigma, lambd, eta1, eta2, p):
    bound = 10
    pi0 = np.exp(-lambd * T)
    n_vals = np.arange(0, bound)
    pin = np.exp(-lambd * T) * (lambd * T) ** n_vals / np.array([math.factorial(i) for i in n_vals])
    
    sump1 = np.zeros(bound)
    sumq1 = np.zeros(bound)

    for n in range(0, bound):
        sump2 = np.zeros(n + 1)
        sumq2 = np.zeros(n + 1)
        for k in range(1, n + 1):
            sump2[k] = (
                P(n, k, p, eta1, eta2)
                * (sigma * np.sqrt(T) * eta1) ** k
                * I(k - 1, x - mu * T, -eta1, -1/(sigma * np.sqrt(T)), -sigma * eta1 * np.sqrt(T))
            )
            sumq2[k] = (
                Q(n, k, p, eta1, eta2)
                * (sigma * np.sqrt(T) * eta2) ** k
                * I(k - 1, x - mu * T, eta2, 1 / (sigma * np.sqrt(T)), -sigma * eta2 * np.sqrt(T))
            )
        sump1[n] = pin[n] * np.sum(sump2)
        sumq1[n] = pin[n] * np.sum(sumq2)
    
    Y1 = np.exp(((sigma * eta1) ** 2) * T / 2) / (sigma * np.sqrt(2 * np.pi * T)) * np.sum(sump1)
    Y2 = np.exp(((sigma * eta2) ** 2) * T / 2) / (sigma * np.sqrt(2 * np.pi * T)) * np.sum(sumq1)
    Y3 = pi0 * norm.cdf(-(x - mu * T) / (sigma * np.sqrt(T)))
    
    Y = Y1 + Y2 + Y3
    return Y

def kou_option_price(S0: float, K: float, r: float, sigma: float, T: float, 
                     eta1: float, eta2: float, p: float, lambd: float,  
                     option_type: OptionType = OptionType.CALL) -> None:
    """
    Price option using Kou model (incomplete implementation).
    
    Parameters:
    S0 (float): initial stock price
    K (float): strike price
    r (float): risk-free rate
    sigma (float): volatility
    T (float): time to maturity
    dt (float): time step
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    p (float): probability of positive jump
    lambd (float): jump intensity
    option_type (OptionType): CALL or PUT
    """
    csi = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    lambd2 = lambd * (csi + 1)
    eta12 = eta1 - 1
    eta22 = eta2 + 1
    p2 = p / (1 + csi) * eta1 / (eta1 - 1)
    print(Upsilon(x=0, sigma=0.2, T=0.5, mu=0.12, lambd=1, eta1=10, eta2=5, p=0.4))
    print(Upsilon(x=-0.5, sigma=0.2, T=0.5, mu=0.12, lambd=1, eta1=10, eta2=5, p=0.4))
    print(Upsilon(x=0.5, sigma=0.2, T=0.5, mu=0.12, lambd=1, eta1=10, eta2=5, p=0.4))
    return S0 * Upsilon(mu=r + 1/2 * sigma**2 - lambd * csi, sigma=sigma, lambd=lambd2, p=p2, eta1=eta12, eta2=eta22, x=math.log(K/S0), T=T) - K * np.exp(-r*T) * Upsilon(mu=r - 1 /2 * sigma**2 - lambd * csi, sigma=sigma, lambd=lambd, p=p, eta1=eta1, eta2=eta2, x=math.log(K/S0), T=T)


def test_poisson_process() -> None:
    """Test the Poisson process visualization."""
    lambd = 1
    T = 5
    dt = 0.01
    poison = poisson_process(lambd, T, dt)
    plt.plot(poison)
    plt.title('Poisson Process')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Jumps')
    plt.grid()
    plt.show()

def test_jump_pdf() -> None:
    """Test the jump size PDF visualization."""
    T = 5
    dt = 0.01

    y, fdp = jumps_pdf(T, dt, 0.3, 5, 5)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y, fdp, 'b-', linewidth=2)
    plt.title('Kou Jump Diffusion - Jump Size PDF')
    plt.xlabel('Jump Size')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.show()

def test_kou_process() -> None:
    """Test the Kou process visualization."""
    S0 = 100
    r = 0.1
    t, S = kou_process(S0=S0, r=r, sigma=0.16, T=1, dt=0.001, eta1=20, eta2=20, p=0.25, lambd=3, M=5)
    risk_free_rate = np.exp(r * t) * S0

    plt.figure(figsize=(10, 6))
    
    for i in range(S.shape[1]):
        plt.plot(t, S[:, i], '.', color=colors[i], markersize=2)
    plt.plot(t, risk_free_rate, 'k-', linewidth=1)
    plt.title('Kou Jump Diffusion Process')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.show()

def test_kou_pricing() -> None:
    """Test Kou pricing visualization."""
    S0 = 100     # Initial stock price
    K = 98       # Strike price
    r = 0.05     # Risk-free rate
    sigma = 0.16  # Volatility
    T = 0.5      # Time to maturity
    eta1 = 10.0   # Positive jump parameter (> 1)
    eta2 = 5.0    # Negative jump parameter (> 0)
    p = 0.4       # Probability of positive jump
    lambd = 1.0   # Jump intensity
    call_price = kou_option_price(S0, K=K, r=r, sigma=sigma, T=T,
                            eta1=eta1, eta2=eta2, p=p, lambd=lambd,
                            option_type=OptionType.CALL)
    print(f"Call option price: {call_price:.4f}")

def test_kou_pricing_mc() -> None:
    """Test option pricing with Kou model."""
    # Parameters
    S0 = 100     # Initial stock price
    K = 98       # Strike price
    r = 0.05     # Risk-free rate
    sigma = 0.16  # Volatility
    T = 0.5      # Time to maturity
    dt = 0.0005
    eta1 = 10.0   # Positive jump parameter (> 1)
    eta2 = 5.0    # Negative jump parameter (> 0)
    p = 0.4       # Probability of positive jump
    lambd = 1.0   # Jump intensity
    M = 100_000   # Number of simulations
    
    # Price call option
    call_price = kou_option_price_mc(S0, K, r, sigma, T, dt, eta1, eta2, p, lambd, M, OptionType.CALL)
    print(f"Call option price: {call_price:.4f}")
    
    # Price put option
    put_price = kou_option_price_mc(S0, K, r, sigma, T, dt, eta1, eta2, p, lambd, M, OptionType.PUT)
    print(f"Put option price: {put_price:.4f}")
    
    # Verify put-call parity (approximately)
    pv_strike = K * np.exp(-r * T)
    parity_diff = call_price - put_price - (S0 - pv_strike)
    print(f"Put-call parity difference: {parity_diff:.4f}")

def test_kou_process_risk_neutral() -> None:
    """Test risk-neutral Kou process."""
    dt = 0.001
    S0 = 100     # Initial stock price
    r = 0.05     # Risk-free rate
    sigma = 0.16  # Volatility
    T = 0.5      # Time to maturity
    eta1 = 10.0   # Positive jump parameter (> 1)
    eta2 = 5.0    # Negative jump parameter (> 0)
    p = 0.4       # Probability of positive jump
    lambd = 1.0   # Jump intensity
    M = 10

    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    csi = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    mu_risk_free = r - 0.5*sigma**2 - lambd * csi
    sigma_risk_free = sigma * np.sqrt(t)
    sigma_risk_free = sigma_risk_free[:, np.newaxis]

    t, S = kou_process(S0, mu_risk_free, sigma_risk_free, T, dt, eta1, eta2, p, lambd, M)
    risk_free_rate = np.exp(r * t) * S0

    plt.figure(figsize=(10, 6))
    for i in range(S.shape[1]):
        plt.plot(t, S[:, i], '.', color=colors[i], markersize=2)
    plt.plot(t, risk_free_rate, 'k-', linewidth=1)
    plt.title('Kou Jump Diffusion Process')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # test_kou_pricing_mc()
    test_kou_pricing()
