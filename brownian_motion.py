import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(T=10, dt=0.01, M=1):
    """
    Simulate a Geometric Brownian Motion (GBM) path.
    
    Parameters:
    T : float : time horizon
    dt : float : time step size
    M : int : number of simulation paths
    
    Returns:
    t : numpy array : time points
    W: numpy array : simulated stock prices
    """

    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    
    dW = np.sqrt(dt) * np.random.normal(size=(N,M))
    W = np.zeros((N + 1,M))
    W[1:,:] = np.cumsum(dW, axis=0)

    return t, W

def geometric_brownian_motion(S0=1, T=10, dt=0.07, r=0.1, sigma=0.2, M=1):
    """
    Simulate a Geometric Brownian Motion (GBM) path.
    
    Parameters:
    S0 : float : initial stock price
    T : float : time horizon
    dt : float : time step size
    r : float : risk-free interest rate
    sigma : float : volatility of the stock
    M : int : number of simulation paths
    
    Returns:
    t : numpy array : time points
    S: numpy array : simulated stock prices
    """
    N = int(T / dt)
    t, W = brownian_motion(T, dt, M)
    time_matrix = np.repeat(t, M).reshape(N+1,M)

    S = S0 * np.exp((r - sigma**2/2) * time_matrix + sigma * W)
    return t, S


def test_brownian_motion():
    t, BM = brownian_motion(M=5)  # Generate 10 paths

    # Plot all paths at once
    plt.figure(figsize=(10, 6))
    plt.plot(t, BM)  # This plots all columns automatically
    plt.grid()
    plt.title(f'Brownian Motion - {BM.shape[1]} Paths')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

def test_geometric_brownian_motion():
    risk_free_rate = 0.07
    t, GBM = geometric_brownian_motion(r =risk_free_rate, M=10)
    risk_free_slope = np.exp(risk_free_rate * t)

    plt.figure(figsize=(10, 6))
    plt.plot(t, GBM) 
    plt.plot(t, risk_free_slope, 'k--', label='Risk-free rate')
    plt.grid()
    plt.title(f'Brownian Motion - {GBM.shape[1]} Paths')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

if __name__ == "__main__":
    test_brownian_motion()
    test_geometric_brownian_motion()



