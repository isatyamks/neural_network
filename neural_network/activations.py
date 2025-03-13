import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    shiftZ = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(shiftZ)
    return exps / np.sum(exps, axis=0, keepdims=True)
