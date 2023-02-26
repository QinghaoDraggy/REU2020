import numpy as np
from matplotlib import pyplot as plt

# Simulating geometric Brownian motion
def stock_sim_path(S, alpha, delta, sigma, T, N, n):
    """Simulates geometric Brownian motion."""
    h = T/n
    # uncomment below for deterministic trend. or, can pass it in as alpha as an array
    alpha = alpha # + np.linspace(0, 0.1, 500).reshape((n,N))
    mean = (alpha - delta - .5*sigma**2)*h
    vol = sigma * h**.5
    return S*np.exp((mean + vol*np.random.randn(n,N)).cumsum(axis = 0))

# Example
if __name__ == "__main__":
    T = 2
    days = int(250*T)
    stock_path = stock_sim_path(100, 0.05, 0, .15, T, 1, days)

    plt.plot(np.arange(days), stock_path)
    plt.show()