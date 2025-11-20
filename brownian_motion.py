import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def brownian_motion_diff(T: float = 10, dt: float = 0.01, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Brownian Motion differential (increments).
    
    Parameters:
    T (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian increments)
    """
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    
    dW = np.sqrt(dt) * np.random.normal(size=(N+1, M))
    return t, dW


def brownian_motion(T: float = 10, dt: float = 0.01, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Brownian Motion path.
    
    Parameters:
    T (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian motion paths)
    """
    N = int(T / dt)
    
    t, dW = brownian_motion_diff(T, dt, M)
    W = np.zeros((N + 1, M))
    W[1:, :] = np.cumsum(dW[:-1, :], axis=0)

    return t, W

def cov_brownian_motion_diff(T: float = 10, dt: float = 0.01, rho: float = 0, num: int = 0, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Covariate Brownian Motion differential (increments).
    
    Parameters:
    T (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian increments)
    """
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)

    mu = np.zeros(num)
    cov = np.full((num, num), rho)
    np.fill_diagonal(cov, 1)

    dW = np.sqrt(dt) * np.random.multivariate_normal(mu, cov, (N + 1, M))
    return t, dW

def cov_brownian_motion(T: float = 10, dt: float = 0.01, rho: float = 0, num: int = 2, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Covariate Brownian Motion path.
    
    Parameters:
    T (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian motion paths)
    """
    N = int(T / dt)
    
    t, dW = cov_brownian_motion_diff(T, dt, rho, num, M)
    W = np.zeros((N + 1, M, num))
    W[1:, :, :] = np.cumsum(dW[:-1, :], axis=0)

    return t, W

def geometric_brownian_motion(S0: float = 1, T: float = 10, dt: float = 0.07, 
                             r: float = 0.1, sigma: float = 0.2, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Geometric Brownian Motion (GBM) path.
    
    Parameters:
    S0 (float): initial stock price
    T (float): time horizon
    dt (float): time step size
    r (float): risk-free interest rate
    sigma (float): volatility of the stock
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, simulated stock prices)
    """
    N = int(T / dt)
    t, W = brownian_motion(T, dt, M)
    time_matrix = np.repeat(t, M).reshape(N+1, M)

    S = S0 * np.exp((r - sigma**2/2) * time_matrix + sigma * W)
    return t, S


def test_brownian_motion() -> None:
    """Test Brownian motion visualization."""
    t, BM = brownian_motion(M=5)  # Generate 5 paths

    # Plot all paths at once
    plt.figure(figsize=(10, 6))
    plt.plot(t, BM)  # This plots all columns automatically
    plt.grid()
    plt.title(f'Brownian Motion - {BM.shape[1]} Paths')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

def test_geometric_brownian_motion() -> None:
    """Test geometric Brownian motion visualization."""
    risk_free_rate = 0.05
    t, GBM = geometric_brownian_motion(r=risk_free_rate, M=10)
    risk_free_slope = np.exp(risk_free_rate * t)

    plt.figure(figsize=(10, 6))
    plt.plot(t, GBM) 
    plt.plot(t, risk_free_slope, 'k--', label='Risk-free rate')
    plt.grid()
    plt.legend()
    plt.title(f'Geometric Brownian Motion - {GBM.shape[1]} Paths')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()

def test_cov_brownian_motion() -> None:
    t, GBM = cov_brownian_motion(T=1, dt=0.01, rho=0.98, num = 2, M=1)
    plt.figure(figsize=(10, 6))
    plt.plot(t, GBM[:,:,0]) 
    plt.plot(t, GBM[:,:,1]) 
    plt.grid()
    plt.legend()
    plt.title(f'Cov Brownian Motion - {GBM.shape[1]} Paths')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()

if __name__ == "__main__":
    # test_brownian_motion()
    # test_geometric_brownian_motion()
    test_cov_brownian_motion()



