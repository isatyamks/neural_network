import numpy as np


'''Introduces non-linearity into the model by outputting  x if x>0 and 0 if x<=0...'''
def relu(Z):
    return np.maximum(0, Z)


''' Compute the derivative (gradient) of the ReLU activation. d/dz(ReLU(z)) = 1 if z>0 and if z<=0 then 0
The code uses a boolean condition (Z > 0), converting the result to float (1.0 or 0.0)'''

def relu_derivative(Z):
    return (Z > 0).astype(float)


'''i explained it in the notebook clearly so in short Convert raw output scores (logits) 
into probabilities that sum to 1 for each sample.'''

def softmax(Z):
    max_val = np.max(Z, axis=0, keepdims=True)
    Z_stable = Z - max_val
    exp_vals = np.exp(Z_stable)
    total = np.sum(exp_vals, axis=0, keepdims=True)
    return exp_vals / total
