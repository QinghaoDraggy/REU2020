import torch
import numpy as np
import scipy.stats
import torch.nn as nn

class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2):
        super(NeuralNet, self).__init__()
        # define the layers of the neural network
        self.a1 = nn.Linear(d, q1)
        self.relu = nn.ReLU()
        self.a2 = nn.Linear(q1, q2)
        self.a3 = nn.Linear(q2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # define the forward pass of the neural network
        out = self.a1(x)
        out = self.relu(out)
        out = self.a2(out)
        out = self.relu(out)
        out = self.a3(out)
        out = self.sigmoid(out)

        return out


def loss(y_pred, s, x, n, tau, f, V, t):
    # define the loss function used to train the neural network
    r_n = torch.zeros(s.M)
    for m in range(0, s.M):
        # compute the residual for each point in the dataset
        r_n[m] = -f(n, m, x, V, t) * y_pred[m] - f(tau[m], m, x, V, t) * (1 - y_pred[m])

    return r_n.mean()

def NN(n, x, s, tau_n_plus_1, f, V, t):
    # define a function that trains a neural network on a given dataset
    epochs = 50
    model = NeuralNet(s.d, s.d + 10, s.d + 10) # create a new neural network
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # define the optimizer

    F = None
    for epoch in range(epochs):
        # train the neural network for a fixed number of epochs
        F = model.forward(x[n].T)
        optimizer.zero_grad()
        criterion = loss(F, s, x, n, tau_n_plus_1, f, V, t)
        criterion.backward()
        optimizer.step()

    return F, model

