import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    m = Y_true.shape[1]
    eps = 1e-15  # avoid log(0)
    loss = -np.sum(Y_true * np.log(Y_pred + eps)) / m
    return loss
