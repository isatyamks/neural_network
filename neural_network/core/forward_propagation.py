import numpy as np
from activations import relu, softmax

def forward_pass(W1, b1, W2, b2, X):
    """
    Computes the forward propagation steps:
    Z1 = W1·X + b1, A1 = relu(Z1)
    Z2 = W2·A1 + b2, A2 = softmax(Z2)
    Returns:
        A2: Output probabilities.
        cache: Tuple of intermediate values.
    """
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache
