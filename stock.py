import numpy as np
import scipy.stats
from brownian_motion import stock_sim_path


class stock:
    def __init__(self, Ks, Kb, T, r, d, M):
        """
        Initializes the stock class with relevant parameters.

        Args:
            Ks (float): Strike price for the call option.
            Kb (float): Strike price for the barrier option.
            T (int): Total number of time steps.
            r (float): Risk-free interest rate.
            d (int): Number of stocks.
            M (int): Number of sample paths.
        """
        self.Ks = Ks
        self.Kb = Kb
        self.T = T
        self.r = r
        self.d = d
        self.M = M

    def g(self, n, m, X, *args):
        """
        Defines the function g which is used in the simulation.

        Args:
            n (int): Current time step.
            m (int): Current sample path.
            X (numpy.ndarray): Matrix containing stock prices for each sample path.
            args (float): Optional arguments.

        Returns:
            float: The value of the function g at the current time step and sample path.
        """
        return np.log(X[int(n), :, m] * (1 - self.Ks) + self.r * (self.T - n))

    def g_t(self, n, m, X, V, t):
        """
        Defines the function g_t which is used in the simulation.

        Args:
            n (int): Current time step.
            m (int): Current sample path.
            X (numpy.ndarray): Matrix containing stock prices for each sample path.
            V (numpy.ndarray): Matrix containing the value of the barrier option at each time step and sample path.
            t (int): Time step at which to evaluate the function.

        Returns:
            float: The value of the function g_t at the current time step and sample path.
        """
        return self.r * (n - t) - np.log(X[int(n), :, m] * (1 + self.Kb)) + \
               np.max(V[t, int(n):, m])

    def h(self, n, m, X, V, t):
        """
        Defines the function h which is used in the simulation.

        Args:
            n (int): Current time step.
            m (int): Current sample path.
            X (numpy.ndarray): Matrix containing stock prices for each sample path.
            V (numpy.ndarray): Matrix containing the value of the barrier option at each time step and sample path.
            t (int): Time step at which to evaluate the function.

        Returns:
            float: The value of the function h at the current time step and sample path.
        """
        return np.log(X[int(n), :, m] * (1 - self.Ks)) \
               + np.max(V[t, int(n):, m])

    # Static method for simulating geometric Brownian motion.
    stock_sim_path = staticmethod(stock_sim_path)

