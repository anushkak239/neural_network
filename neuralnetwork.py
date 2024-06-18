import numpy as np


inputs = [[1, 2, 3, 2.5],
          [2.0, 3.0, 4.1, 5.0],
          [1.2, 2.7, 2.4, 4.5]]


class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = layer_dense(4,6)
layer2 = layer_dense(6,2)
 
layer1.forward(inputs)
print(layer1.output)
layer2.forward(layer1.output)
print (layer2.output)
 
