import numpy as np
import nnfs


from nnfs.datasets import spiral_data

np.random.seed(0)

#lalala
nnfs.init()

input = [[1, 2, 3, 4,],
          [2, 4.5, 6, 4.6]]
input, y = spiral_data(100, 3)
class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases 
        
class activation_class:
    def forward(self, input):
        self.output = np.maximum(0, input)
        
layer1 = layer_dense(2,3)

layer1.forward (input)
print(layer1.output)

act1 = activation_class();
act1.forward (layer1.output)
print(act1.output)
        