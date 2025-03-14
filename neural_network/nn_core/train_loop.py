import numpy as np
from utils import cross_entropy_loss

def train(network, X, Y, epochs=10, batch_size=64, X_val=None, Y_val=None):
    m = X.shape[1]  # Number of samples
    history = {"loss": [], "val_loss": []}  # To store loss history

    for epoch in range(epochs):
        # Shuffle the data
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        for i in range(0, m, batch_size):
            # Get the current batch
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]

            # Forward pass
            A2, cache = network.forward(X_batch)
            loss = cross_entropy_loss(A2, Y_batch)

            # Backward pass and update parameters
            grads = network.backward(cache, Y_batch)
            network.update_parameters(grads)

        # Calculate and log training loss
        A2_full, _ = network.forward(X)
        train_loss = cross_entropy_loss(A2_full, Y)
        history["loss"].append(train_loss)

        # Calculate and log validation loss if validation data is provided
        if X_val is not None and Y_val is not None:
            A2_val, _ = network.forward(X_val)
            val_loss = cross_entropy_loss(A2_val, Y_val)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")

    return history