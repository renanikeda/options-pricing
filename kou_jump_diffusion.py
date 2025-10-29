from utils import OptionType
from brownian_motion import brownian_motion
import matplotlib.pyplot as plt
import numpy as np
import math

def poisson_process(lambd, T, dt, M=1):
    """
    Poisson process.
    
    Parameters:
    lambd (float): intensity parameter
    T (float): time horizon
    dt (float): step discretization
    
    Returns:
    np.array: array of jump counts
    """
    N = int(T / dt)
    jumps = np.random.poisson(lambd * dt, (M,N))
    return np.cumsum(jumps, axis=1)

def generate_jump_sizes(p, eta1, eta2, num_jumps):
    """
    Generate jump sizes from the double exponential distribution.
    
    Parameters:
    p (float): probability of positive jump
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps  
    num_jumps (int): number of jumps to generate
    
    Returns:
    np.array: array of jump sizes
    """
    if num_jumps == 0:
        return np.array([[]])
    
    # Generate random numbers to determine jump direction
    directions = np.random.random(num_jumps)
    jump_sizes = np.zeros(num_jumps)
    
    # Positive jumps
    positive_mask = directions < p
    num_positive = np.sum(positive_mask)
    if num_positive > 0:
        jump_sizes[positive_mask] = np.random.exponential(1/eta1, num_positive)
    
    # Negative jumps
    negative_mask = ~positive_mask
    num_negative = np.sum(negative_mask)
    if num_negative > 0:
        jump_sizes[negative_mask] = -np.random.exponential(1/eta2, num_negative)
    
    return jump_sizes

def jumps_pdf(T, dt, p, eta1, eta2):
    """
    Calculate the PDF of jump sizes for the double exponential distribution.
    
    Parameters:
    y (float or array): values where to evaluate the density
    p (float): probability of positive jump
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    
    Returns:
    float or np.array: PDF values
    """

    y = np.linspace(-math.floor(T/2), math.floor(T/2), math.floor(1/dt))
    # y = np.asarray(y)
    fdp = np.zeros_like(y, dtype=float)
    
    # Positive part (y >= 0)
    positive_mask = y >= 0
    fdp[positive_mask] = p * eta1 * np.exp(-eta1 * y[positive_mask])
    
    # Negative part (y < 0)
    negative_mask = y < 0
    fdp[negative_mask] = (1-p) * eta2 * np.exp(eta2 * y[negative_mask])
    
    return y, fdp

def kou_process(S0, r, sigma, T, dt, eta1, eta2, p, lambd, M=5):
    """
    Generate a single path of the Kou jump diffusion process.
    
    Parameters:
    S0 (float): initial stock price
    r (float): risk-free rate
    sigma (float): volatility
    T (float): time to maturity
    dt (float): time step
    eta1 (float): parameter for positive jumps
    eta2 (float): parameter for negative jumps
    p (float): probability of positive jump
    lambd (float): jump intensity
    
    Returns:
    tuple: (time_grid, stock_price_path)
    """
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    
    # Generate Brownian motion
    t, W = brownian_motion(T, dt, M)
    time_matrix = np.repeat(t, M).reshape(N+1,M)

    # W = W.flatten()
    
    # Generate Poisson jumps
    poisson_jumps = poisson_process(lambd, T, dt, M)

    # n_jumps = poisson_jumps[-1]
    # Initialize log price process
    log_S = np.zeros((N+1, M))
    log_S[:, 0] = np.log(S0)

    csi = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    dw = np.diff(W, prepend=0)

    log_S = np.log(S0) + r*time_matrix - 0.5*sigma**2*time_matrix + sigma*dw

    for path_index in range(M):
        n_jumps = poisson_jumps[path_index,-1]
        jumps_generated= generate_jump_sizes(p, eta1, eta2, n_jumps)
        if jumps_generated.size == 0: continue
        for i in range(N):
            jump_index = poisson_jumps[path_index, i]
            jumps_sum = np.sum(jumps_generated[:jump_index], axis=0)
            if jumps_sum.size == 0: jumps_sum.resize(M)
            log_S[i, path_index] += jumps_sum

    return t, np.exp(log_S)

def kou_option_price_mc(S0, K, r, sigma, T, eta1, eta2, p, lambd, M, option_type=OptionType.CALL):
    """
    Price European option using Monte Carlo simulation with Kou jump diffusion.
    
    Parameters:
    S0 (float): initial stock price
    K (float): strike price
    r (float): risk-free rate
    sigma (float): volatility
    T (float): time to maturity
    eta1 (float): parameter for positive jumps (> 1)
    eta2 (float): parameter for negative jumps (> 0)
    p (float): probability of positive jump
    lambd (float): jump intensity
    M (int): number of Monte Carlo simulations
    option_type (OptionType): CALL or PUT
    
    Returns:
    float: option price
    """
    dt = 0.01  # Fixed time step
    payoffs = np.zeros(M)
    
    # Generate stock price path
    _, S_path = kou_process(S0, r, sigma, T, dt, eta1, eta2, p, lambd, M)
    S_T = S_path[-1, :]  # Final stock price
    
    # Calculate payoff
    if option_type == OptionType.CALL:
        payoffs = np.max(S_T - K, 0)
    elif option_type == OptionType.PUT:
        payoffs = np.max(K - S_T, 0)
    else:
        raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")
    
    # Discount expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def zeta_kou(T, dt, r, sigma, eta1, eta2, p, lambd):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    
    # Generate Brownian motion
    t, W = brownian_motion(T, dt, M=1)
    W = W.flatten()

    poisson_jumps = poisson_process(lambd, T, dt, M=1).flatten()
    n_jumps = poisson_jumps[-1] if poisson_jumps.size > 0 else 0

    zeta = np.zeros(N+1)
    zeta = r*t + sigma*W

    jumps_generated= generate_jump_sizes(p, eta1, eta2, n_jumps)
    if jumps_generated.size == 0: return t, zeta
    for i in range(N):
        jump_index = poisson_jumps[i]
        jumps_sum = np.sum(jumps_generated[:jump_index], axis=0)
        zeta[i] += jumps_sum
    
    return t, zeta

def kou_option_price(S0, K, r, sigma, T, dt, eta1, eta2, p, lambd,  option_type=OptionType.CALL):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    
    zeta = zeta_kou(T, dt, r, sigma, eta1, eta2, p, lambd)


def test_poisson_process():
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

def test_jump_pdf():
    """Test the jump size PDF visualization."""
    T = 5
    dt = 0.01
    N = int(T / dt)
    # t = np.linspace(0, T, N + 1)

    y, fdp = jumps_pdf(T, dt, 0.3, 5, 5)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y, fdp, 'b-', linewidth=2)
    plt.title('Kou Jump Diffusion - Jump Size PDF')
    plt.xlabel('Jump Size')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.show()

def test_kou_process():
    S0=100
    r=0.1
    t, S = kou_process(S0=S0, r=r, sigma=0.16, T=1, dt=0.001, eta1=20, eta2=20, p=0.25, lambd=3, M=5)
    risk_free_rate = np.exp(r * t) * S0

    plt.figure(figsize=(10, 6))
    colors = ['black', 'red', 'green', 'blue', 'olive', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for i in range(S.shape[1]):
        plt.plot(t, S[:, i], '.', color=colors[i], markersize=2)
    # plt.plot(t, S, 'b.', markersize=2)
    # plt.plot(t, S, 'r-', linewidth=0.5)
    plt.plot(t, risk_free_rate, 'k-', linewidth=1)
    plt.title('Kou Jump Diffusion Process')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.show()

def test_kou_pricing():
    S0=100
    r=0.1
    T = 1.0
    dt=0.01
    sigma = 0.2
    eta1=20
    eta2=20
    p=0.25
    lambd=3
    t, S = zeta_kou(T, dt, r, sigma, eta1, eta2, p, lambd)

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'r-', linewidth=0.5)
    plt.title('Kou Jump Diffusion Process')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.show()

def test_kou_pricing_mc():
    """Test option pricing with Kou model."""
    # Parameters
    S0 = 100     # Initial stock price
    K = 100      # Strike price
    r = 0.05     # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0      # Time to maturity
    eta1 = 5.0   # Positive jump parameter (> 1)
    eta2 = 5.0   # Negative jump parameter (> 0)
    p = 0.3      # Probability of positive jump
    lambd = 1.0  # Jump intensity
    M = 100_000    # Number of simulations
    
    # Price call option
    call_price = kou_option_price_mc(S0, K, r, sigma, T, eta1, eta2, p, lambd, M, OptionType.CALL)
    print(f"Call option price: {call_price:.4f}")
    
    # Price put option
    put_price = kou_option_price_mc(S0, K, r, sigma, T, eta1, eta2, p, lambd, M, OptionType.PUT)
    print(f"Put option price: {put_price:.4f}")
    
    # Verify put-call parity (approximately)
    pv_strike = K * np.exp(-r * T)
    parity_diff = call_price - put_price - (S0 - pv_strike)
    print(f"Put-call parity difference: {parity_diff:.4f}")

if __name__ == "__main__":
    # test_jump_pdf()
    # print(generate_jump_sizes(0.3, 5, 5, 10, 5))
    # test_kou_process()
    # test_kou_pricing()
    test_kou_pricing_mc()
    