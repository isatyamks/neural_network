import numpy as np
from activations import relu, softmax

'''
This function performs the forward pass of a neural network with one hidden layer.
The forward pass computes the following:

simple forward pass  Z1 = W1*X +B1
A1 = ReLU(Z1)

A2 = softmax(W2 * A1 + b2)

cache: Save intermediate values needed for backpropagation
'''

def forward_pass(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache
