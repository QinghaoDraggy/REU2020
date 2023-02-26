import torch
import numpy as np
import scipy.stats

from model import NN 


def getTauAndV(S, X, f, V, t):
    """
    Compute the optimal stopping time and value for each path at time t.
    Args:
        S (stock): stock object
        X (torch.Tensor): tensor of stock prices
        f (function): function used to compute the value of the option
        V (float or numpy.ndarray): initial value(s) for V
        t (int): current time step
    Returns:
        tau_mat (numpy.ndarray): matrix of optimal stopping times for each path at each time step
        V_mat_test (numpy.ndarray): matrix of optimal values for each path at each time step
        mods (list): list of neural network models for each time step
    """
    mods = [None] * S.T  # initialize a list of models (para) at each time step
    tau_mat = np.zeros((S.T + 1, S.M))  # initialize matrix of optimal stopping times
    tau_mat[S.T, :] = S.T  # set the stopping time for each path to T

    f_mat = np.zeros((S.T + 1, S.M))  # initialize matrix of probabilities
    f_mat[S.T, :] = 1  # set the probability for each path at time T to 1

    V_mat_test = np.zeros((S.T + 1, S.M))  # initialize matrix of optimal values

    for m in range(0, S.M):
        # set V_T value for each path
        V_mat_test[S.T, m] = f(S.T, m, X, V, t) 

    # compute optimal stopping times and values for each time step
    for n in range(S.T - 1, t - 1, -1):
        probs, mod_temp = NN(n, X, S, torch.from_numpy(tau_mat[n + 1]).float(), f, V, t)
        mods[n] = mod_temp
        np_probs = probs.detach().numpy().reshape(S.M)  # get probabilities for each asset in each path at time n
        print(n, ":", np.min(np_probs), " , ", np.max(np_probs))

        # set probabilities based on whether the probability is greater than 0.5
        f_mat[n, :] = (np_probs > 0.5) * 1.0

        # compute optimal stopping times and values
        tau_mat[n, :] = np.argmax(f_mat, axis=0)
        for m in range(0, S.M):
            V_mat_test[n, m] = f(n, m, X, V, t)

    return tau_mat, V_mat_test, mods

def getValueForAllTime(S, X, f, initValue=None):
    """
    Compute the optimal stopping time and value for each time step for all paths.
    Args:
        S (stock): stock object
        f (function): function used to compute the value of the option
        initValue (float or numpy.ndarray): initial value(s) for V
    Returns:
        tauForAllTime (numpy.ndarray): array of optimal stopping times for each path at each time step
        VForAllTime (numpy.ndarray): array of optimal values for each path at each time step
        modsForAllTime (list): list of neural network models for each time step
    """
    # pre-allocate memory for V, tau, and mods
    VForAllTime = np.zeros((S.T, S.M))
    tauForAllTime = np.zeros((S.T, S.M))
    modsForAllTime = [None] * S.T
    for time in range(S.T - 1, -1, -1):
        tau, V, mods = getTauAndV(S, X, f, V=initValue, t=time)
        # store V and tau
        VForAllTime[time, :] = V
        tauForAllTime[time, :] = tau
        modsForAllTime[time] = mods
    return tauForAllTime, VForAllTime, modsForAllTime

