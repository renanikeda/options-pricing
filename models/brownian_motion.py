import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from utils import colors

def brownian_motion_diff(tau: float = 10, dt: float = 0.01, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Brownian Motion increments (differentials).
    
    Parameters:
    tau (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian increments)
    """
    N = int(tau / dt)
    t = np.linspace(0, tau, N + 1)
    
    dW = np.sqrt(dt) * np.random.normal(size=(N+1, M))
    return t, dW

def SimulateBM(T=1.0, N=1000, M=10000) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Função que simula o movimento Browniano
    T (float > 0) - maturidade (em anos)
    N (int) - discretização do tempo
    M (int) - número de simulações para estimação por Monte Carlo
    '''
    time = np.linspace(0, T, N+1) # vetor dos tempos
    dt = time[1] - time[0]
    dW = np.sqrt(dt) * np.random.normal(size=(N,M))
    W = np.zeros((N+1,M))
    W[1:,:] = np.cumsum(dW, axis=0)
    # print(np.mean(np.mean(W, axis=0)), np.mean(np.std(W, axis=0)))
    return time, W

def brownian_motion(tau: float = 1, dt: float = 0.001, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Brownian Motion path.
    
    Parameters:
    tau (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian motion paths)
    """
    N = int(tau / dt)
    
    t, dW = brownian_motion_diff(tau, dt, M)
    W = np.zeros((N + 1, M))
    W[1:, :] = np.cumsum(dW[:-1, :], axis=0)
    # print(np.mean(np.mean(W, axis=0)), np.mean(np.std(W, axis=0)))
    return t, W


def cov_brownian_motion_diff(tau: float = 10, dt: float = 0.01, rho: float = 0, num: int = 0, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Covariate Brownian Motion differential (increments).
    
    Parameters:
    tau (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian increments)
    """
    N = int(tau / dt)
    t = np.linspace(0, tau, N + 1)

    mu = np.zeros(num)
    cov = np.full((num, num), rho)
    np.fill_diagonal(cov, 1)

    dW = np.sqrt(dt) * np.random.multivariate_normal(mu, cov, (N + 1, M))
    return t, dW

def cov_brownian_motion(tau: float = 10, dt: float = 0.01, rho: float = 0, num: int = 2, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Covariate Brownian Motion path.
    
    Parameters:
    tau (float): time horizon
    dt (float): time step size
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, Brownian motion paths)
    """
    N = int(tau / dt)
    
    t, dW = cov_brownian_motion_diff(tau, dt, rho, num, M)
    W = np.zeros((N + 1, M, num))
    W[1:, :, :] = np.cumsum(dW[:-1, :], axis=0)

    return t, W

def geometric_brownian_motion(S0: float = 1, tau: float = 10, dt: float = 0.07, 
                             r: float = 0.1, sigma: float = 0.2, M: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Geometric Brownian Motion (GBM) path.
    
    Parameters:
    S0 (float): initial stock price
    tau (float): time horizon
    dt (float): time step size
    r (float): risk-free interest rate
    sigma (float): volatility of the stock
    M (int): number of simulation paths
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (time points, simulated stock prices)
    """
    N = int(tau / dt)
    t, W = brownian_motion(tau, dt, M)
    # W = np.sqrt(sigma)*np.random.normal(0,1,(N+1,M)) 
    time_matrix = np.repeat(t, M).reshape(N+1, M)

    S = S0 * np.exp((r - 0.5*sigma**2) * time_matrix + sigma * W)
    return t, S


def test_brownian_motion() -> None:
    """Test Brownian motion visualization."""
    t, BM = brownian_motion(M=5)  # Generate 5 paths

    # Plot all paths at once
    plt.figure(figsize=(10, 6))
    for i in range(BM.shape[1]):
        plt.plot(t, BM[:, i], color=colors[i])
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
    t, GBM = cov_brownian_motion(tau=1, dt=0.01, rho=0.98, num = 2, M=1)
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
    test_brownian_motion()
    # test_geometric_brownian_motion()
    # test_cov_brownian_motion()
    # t, BM = brownian_motion(M=100_000)
    # t, BM2 = SimulateBM(M= 100_000)



