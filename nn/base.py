import numpy as np

class Module:
    def forward(self, x: np.ndarray):
        raise NotImplementedError("Forward method not implemented.")
    
    def backward(self, grad_output: np.ndarray):
        raise NotImplementedError("Backward method not implemented.")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args)
    
    def zero_grad(self):
        pass
    
    def step(self, lr: float):
        pass