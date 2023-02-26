import torch
import numpy as np
import scipy.stats
import argparse

from stock import stock
from train import getValueForAllTime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters for stock simulation')
    parser.add_argument('-T', '--time', type=int, default=2, help='Number of years to simulate stock prices')
    parser.add_argument('-d', '--num_stocks', type=int, default=1, help='Number of stocks to simulate')
    parser.add_argument('-M', '--num_samples', type=int, default=100, help='Number of sample paths')
    args = parser.parse_args()
    
    # Define parameters
    T = args.time
    days = int(10 * T) - 1
    M = args.num_samples
    d = args.num_stocks

    # Initialize stock simulation
    S = stock(0, 0.02, days, 0.05, d, M)
    stock_path = S.stock_sim_path(100, 0.05, 0, .15, T)
    X = torch.from_numpy(stock_path).float()
    
    # Define number of buying decisions
    N_tau = 1

    # Store all tau and neural networks
    allDecisions = []
    allMods = []

    # Training
    tau, V, mods = getValueForAllTime(S, X, S.g)
    allDecisions.append(tau)
    allMods.append(mods)

    for n in range(N_tau - 1):
        # Compute tau and neural networks for buying and holding decisions
        tau, V, mods = getValueForAllTime(S, X, S.g_t, V)
        allDecisions.append(tau)
        allMods.append(mods)

        tau, V, mods = getValueForAllTime(S, X, S.h, V)
        allDecisions.append(tau)
        allMods.append(mods)

    # Compute final tau and neural network for selling decision
    tau, V, mods = getValueForAllTime(S, X, S.g_t, V)
    allDecisions.append(tau)
    allMods.append(mods)

    # Print results for training data
    for m in range(S.M):
        print('Decisions for sample {}:'.format(m))
        decision = 0
        for decisions in allDecisions:
            if decision >= S.T:
                break
            decision = int(decisions[decision, decision+1, m])
            print(decision, end=' ')
        print()

    # Note: allMods contains all neural networks for each decision with initial time step t for t = 0,...,S.T
    # We could use these models to get predictions for test data, but it would require tedious processing.

