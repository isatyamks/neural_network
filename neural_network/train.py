import numpy as np
import pickle
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_preprocessing import load_data





X_train, Y_train, X_test, Y_test = load_data("data/mnist_train.csv", "data/mnist_test.csv")

input_size = 784      # 28x28 images flattened
hidden_size = 64      # Tunable parameter
output_size = 10      # 10 digit classes (0-9)
learning_rate = 0.1
epochs = 10




nn_mnist = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)


history = nn_mnist.train(X_train, Y_train, epochs=epochs, batch_size=64,X_val=X_test, Y_val=Y_test)



with open(f"model/nn_mnist_model.pkl", "wb") as f:
    pickle.dump(nn_mnist, f)


predictions = nn_mnist.predict(X_test)
accuracy = np.mean(predictions == np.argmax(Y_test, axis=0)) * 100
print(f"Test Accuracy: {accuracy}%")

# plt.plot(history["loss"], label="Train Loss")
# if history["val_loss"]:
#     plt.plot(history["val_loss"], label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.legend()
# plt.show()
