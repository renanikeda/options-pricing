from utils import OptionType, colors, measure, load_params
from brownian_motion import brownian_motion_diff, brownian_motion
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import binom
import numpy as np
import math
from typing import Tuple
import pandas as pd

def poisson_process(lambd: float, tau: float, dt: float, M: int = 1) -> np.ndarray:
    """
    Poisson process.
    
    Parameters:
    lambd (float): intensity parameter
    tau (float): time horizon
    dt (float): step discretization
    M (int): number of paths
    
    Returns:
    np.ndarray: array of jump counts
    """
    N = int(tau / dt)
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

def kou_process_steps(S0: float, r: float, sigma: float, tau: float, dt: float,
                      eta1: float, eta2: float, p: float, lambd: float, M: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate paths of the Kou jump diffusion process.
    
    Parameters:
    S0 (float): initial stock price
    r (float): risk-free rate
    sigma (float): volatility
    tau (float): time to maturity
    dt (float): time step
    eta1 (float): parameter for positive jumps (η₁ > 1)
    eta2 (float): parameter for negative jumps (η₂ > 0)
    p (float): probability of positive jump
    lambd (float): jump intensity (λ)
    M (int): number of paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time_grid, stock_price_paths)
    """
    N = int(tau / dt)
    
    # Calculate drift correction for martingale property
    # ξ = λ * E[e^Y - 1] = λ * [p*η₁/(η₁-1) + (1-p)*η₂/(η₂+1) - 1]
    csi = lambd * (p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1)
    
    # Adjusted drift
    mu = r - 0.5 * sigma**2 - csi
    
    # Initialize log-prices
    log_S = np.full((N + 1, M), np.log(S0), dtype=float)
    
    # Generate Brownian increments
    Z = np.random.randn(N, M)
    dW = Z * np.sqrt(dt)
    
    # Generate jump counts for all time steps
    jumps = np.random.poisson(lambd * dt, size=(N, M))
    
    # Simulate paths
    for step in range(N):
        # Diffusion component
        log_S[step + 1, :] = log_S[step, :] + mu * dt + sigma * dW[step, :]
        
        # Jump component
        Nj = jumps[step, :]
        idxs_with_jumps = np.nonzero(Nj)[0]
        
        if idxs_with_jumps.size > 0:
            for i in idxs_with_jumps:
                nj = int(Nj[i])
                Ys = generate_jump_sizes(p, eta1, eta2, nj)
                log_S[step + 1, i] += Ys.sum()
    
    # Time grid
    t = np.linspace(0, tau, N + 1)
    S = np.exp(log_S)
    print(np.mean(S[-1]) / S0)
    print(np.exp(r * tau))
    return t, np.exp(log_S)


if __name__ == "__main__":
    """Test the Kou process visualization."""
    S0 = 100
    r = 0.1
    t, S = kou_process_steps(S0=S0, r=r, sigma=0.16, tau=1, dt=0.001, eta1=20, eta2=20, p=0.25, lambd=3, M=1000)
    risk_free_rate = np.exp(r * t) * S0

    plt.figure(figsize=(10, 6))
    
    for i in range(5):
        plt.plot(t, S[:, i], '.', color=colors[i], markersize=2)
    plt.plot(t, risk_free_rate, 'k-', linewidth=1)
    plt.title('Kou Jump Diffusion Process')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.show()