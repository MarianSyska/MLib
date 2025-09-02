
from itertools import zip_longest

import numpy as np

import nn

class MultiLayerPerceptron(nn.base.Module):
    
    def __init__(self, layers_sizes, bias=True, init_weights=None, activation_fn=nn.activation.Sigmoid):
        super().__init__()
    
        activation_n = len(layers_sizes) - 2
        activations = [activation_fn() for _ in range(activation_n)]
        
        layer_n = len(layers_sizes) - 1
        layers = [nn.layers.LinearModule(layers_sizes[i], layers_sizes[i+1], bias=bias, init_weights=init_weights) for i in range(layer_n)]
        
        self.layers = list(zip_longest(activations, layers))
        

    def forward(self, x: np.ndarray):
        for activation, layer in self.layers:
            if activation is None:
                x = layer(x)
            else:
                x = activation(layer(x))
        return x
    
    
    def backward(self, grad_output: np.ndarray):
        for activation, layer in reversed(self.layers):
            if activation is None:
                grad_output = layer.backward(grad_output)
            else:
                grad_output = layer.backward(activation.backward(grad_output))
                
        return grad_output
    
    
    def zero_grad(self):
        for activation, layer in self.layers:
            layer.zero_grad()
            if activation is not None:
                activation.zero_grad()


    def step(self, lr):
        for activation, layer in self.layers:
            layer.step(lr)