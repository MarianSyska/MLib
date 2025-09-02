import numpy as np

from .base import Module

class LinearModule(Module):
    
    def __init__(self, in_number, out_number, bias=True, init_weights=None):
        self.bias = bias
        
        weight_shape = (in_number, out_number) if bias else (in_number, out_number)
        
        if init_weights is not None:
            self.weights = init_weights(weight_shape)
        else:
            self.weights = np.random.rand(*weight_shape)
        
        if bias:
            self.bias_value = 0.0
            self.bias_grad = 0.0
        
        self.gradients = np.zeros_like(self.weights)
      
            
    def forward(self, x : np.ndarray):
        if x.ndim > 2:
            raise ValueError("Input must be a 1D or, for batched input, a 2D array.")
        
        if x.shape[-1] != self.weights.shape[0]:
            raise ValueError(f"Input needs to have {self.weights.shape[0]} elements, but got {x.shape[-1]}.")
            
        self.last_input = x
        
        return x.dot(self.weights) + self.bias_value if self.bias else x.dot(self.weights)


    def backward(self, grad_output: np.ndarray):
        if grad_output.shape[-1] != self.weights.shape[1]:
            raise ValueError(f"Gradient Output needs to have {self.weights.shape[1]} elements, but got {grad_output.shape[-1]}.")
        
        np.einsum('ni,nj->ij', self.last_input, grad_output, out=self.gradients)
        np.divide(self.gradients, grad_output.shape[0], self.gradients)
        
        if self.bias:
            self.bias_grad = np.sum(grad_output) / grad_output.shape[0]
    
        return grad_output.dot(self.weights.T)
    
    
    def zero_grad(self):
        super().zero_grad()
        self.gradients = np.zeros_like(self.weights)
        if self.bias:
            self.bias_grad = 0.0
        self.last_input = None


    def step(self, lr: float):
        self.weights -= lr * self.gradients
        if self.bias:
            self.bias_value -= lr * self.bias_grad

