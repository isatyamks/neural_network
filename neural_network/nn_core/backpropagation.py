import numpy as np
from utils import relu_derivative

def backpropagation(W2, cache, Y):
    """
    Computes gradients using backpropagation.
    Args:
        W2: Second layer weights.
        cache: Tuple containing (X, Z1, A1, Z2, A2) from forward_pass.
        Y: True labels (one-hot).
    Returns:
        grads: Dictionary of gradients.
    """
    X, Z1, A1, Z2, A2 = cache
    m = X.shape[1]
    dZ2 = A2 - Y  # derivative of softmax-crossentropy
    dW2 = (1.0/m) * np.dot(dZ2, A1.T)
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1.0/m) * np.dot(dZ1, X.T)
    db1 = (1.0/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads
