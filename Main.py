import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
class Layer_Dense:
    # Initialize Weight and Biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons) #randomly initialize weights
        self.biases =np.zeros((1, n_neurons)) #initialize biases to zero
        pass #using pass parameter as a placeholder
    #forward pass
    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases
        #calculate output values from inputs, weights and biases
        pass
nnfs.init()
X, y = spiral_data(samples=100, classes=3) 
plt.scatter(X[:, 0], X[:, 1],c=y, cmap='brg')
dense1=Layer_Dense(2,3) #create layer with 2 inputs and 3 neurons
dense1.forward(X) #forward pass through layer
print(dense1.output[:100]) #print first 100 output values of the layer