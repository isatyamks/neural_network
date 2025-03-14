import numpy as np
from utils import relu, relu_derivative, softmax,cross_entropy_loss
from nn_core.forward_propagation import forward_propagation
from nn_core.backpropagation import backpropagation
from nn_core.train_loop import train
from nn_core.params_update import params_update

class Neural_Network:
    '''
np.random.randn(hidden_size, input_size)  because for each hidden neuron we need x weights and here x no of input 
so for y neuron we need xy weights
so here using np i create a 2d matrix of x row and y cols 
the scalling factor is selected only becuase its perfect for relu and you can change it .....     

'''

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
    def forward(self, X):
        A2, cache = forward_propagation(self.W1, self.b1, self.W2, self.b2, X)
        return A2, cache
    
    def backward(self, cache, Y):
        grads = backpropagation(self.W2, cache, Y)
        return grads
    
    def update_parameters(self, grads):
        self.W1, self.b1, self.W2, self.b2 = params_update(
            self.W1, self.b1, self.W2, self.b2, grads, self.learning_rate
        )
    
    def train(self, X, Y, epochs=10, batch_size=64, X_val=None, Y_val=None):
        history = train(self, X, Y, epochs=epochs, batch_size=batch_size, X_val=X_val, Y_val=Y_val)
        return history  
    
    def predict(self, X):
        A2, _ = self.forward(X)
        predictions = np.argmax(A2, axis=0)
        return predictions

if __name__ == "__main__":
    pass
