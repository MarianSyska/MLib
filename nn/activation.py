import numpy as np

from .base import Module

# Activation Functions

class ReLU(Module):
    
    def forward(self, x: np.ndarray):
        self.last_input = x
        return np.maximum(0, x)
    
    
    def backward(self, grad_output: np.ndarray):
        relu_grad = (self.last_input > 0)
        return grad_output * relu_grad
    
    
    def zero_grad(self):
        super().zero_grad()
        self.last_input = None


class Sigmoid(Module):
    
    def forward(self, x: np.ndarray):
        self.last_input = x
        return 1 / (1 + np.exp(-x))
    
    
    def backward(self, grad_output: np.ndarray):
        s = 1 / (1 + np.exp(-self.last_input))
        sigmoid_grad = s * (1 - s)
        return grad_output * sigmoid_grad
    
    
    def zero_grad(self):
        super().zero_grad()
        self.last_input = None

        
class Tanh(Module):
    
    def forward(self, x: np.ndarray):
        self.last_input = x
        return np.tanh(x)
    
    
    def backward(self, grad_output: np.ndarray):
        t = np.tanh(self.last_input)
        tanh_grad = 1 - t ** 2
        return grad_output * tanh_grad
    
    
    def zero_grad(self):
        super().zero_grad()
        self.last_input = None