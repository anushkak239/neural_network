import numpy as np 
import nnfs

from nnfs.datasets import spiral_data 
np.random.seed(0)
nnfs.init()

class layer_dense():
    def __init__(self, n_input, n_neuron):
        self.weights =  0.10* np.random.rand(n_input, n_neuron)
        self.biases = np.zeros ((1, n_neuron))
        
    def forward(self, input):
        self.output= np.dot( input, self.weights) + self.biases
        
    
    
class activation_class:
    def forward(self, input):
        self.output = np.maximum(0, input)
        
class softmax:
    def forword(self, input):
        e_value = np.exp(input - np.max(input, axis= 1, keepdims= True))
        probab = e_value / np.sum(e_value, axis= 1, keepdims = True)
        self.output = probab
        
'''calculate loss'''       
class loss:
    def calculate  (self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss
'''calculate who wrong is neural network loss categorial cross entropy'''
class lcce(loss):
    def forward(self,  y_predict, y_true):
        sample = len(y_predict)
        y_predict_clipped = np.clip(y_predict, 1e-7, 1-1e-7)
        
        if len(y_true.shape) ==1:
            correct_confidence = y_predict_clipped[range(sample), y_true]
            
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_predict_clipped*y_true, axis= 1)
            
        negative_log = -np.log(correct_confidence)
        return negative_log
        
      
                
input, y = spiral_data(100, 3)

dense1= layer_dense(2,3) 
act1 = activation_class()

dense2= layer_dense(3,3)
act2 = softmax()

dense1.forward (input)
act1.forward(dense1.output)

dense2.forward(act1.output)
act2.forword(dense2.output)

loss_function = lcce()
loss = loss_function.calculate(act2.output, y )

print(act2.output[:5],
      loss)    