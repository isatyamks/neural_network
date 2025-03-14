def params_update(W1, b1, W2, b2, grads, learning_rate):
    W1_new = W1 - learning_rate * grads["dW1"]
    b1_new = b1 - learning_rate * grads["db1"]
    W2_new = W2 - learning_rate * grads["dW2"]
    b2_new = b2 - learning_rate * grads["db2"]
    return W1_new, b1_new, W2_new, b2_new
