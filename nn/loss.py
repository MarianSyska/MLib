import numpy as np

from .base import Module

# Loss Functions

class MSE(Module):
    
    def forward(self, prediction: np.ndarray, target: np.ndarray):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must have the same shape.")
        self.last_prediction = prediction
        self.last_target = target
        return np.mean((prediction - target) ** 2)
    
    
    def backward(self):
        return self.last_prediction - self.last_target


    def zero_grad(self):
        super().zero_grad()
        self.last_prediction = None
        self.last_target = None

class CrossEntropy(Module):
    
    def forward(self, prediction: np.ndarray, target):
        if isinstance(target, np.ndarray) and target.ndim > 1:
            if prediction.shape[0] != target.shape[0]:
                raise ValueError("Prediction and target must have the same batch size.")
            elif target.shape[1] != 1:
                raise ValueError("Target must have one element.")
        
        exp_logits = np.exp(prediction - np.max(prediction, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        correct_class_probabilities = None
        if isinstance(target, np.ndarray) and target.ndim > 1:
            batch_size = prediction.shape[0]
            
            correct_class_probabilities = np.zeros_like(target, dtype=prediction.dtype)
            for i in range(batch_size):
                correct_class_probabilities[i] = probabilities[i, int(target[i])]
        else:
                correct_class_probabilities = probabilities[int(target)]
        
        log_probabilities = -np.log(correct_class_probabilities + 1e-10)
        avg_loss = np.mean(log_probabilities)

        self.last_prediction = prediction
        self.last_target = target
        return avg_loss
    
    
    def backward(self):
        exp_logits = np.exp(self.last_prediction - np.max(self.last_prediction, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        one_hot_labels = np.zeros_like(probabilities)
        if self.last_target.ndim > 1:
            batch_size = self.last_prediction.shape[0]
            
            for i in range(batch_size):
                one_hot_labels[i, int(self.last_target[i])] = 1
        else:
            one_hot_labels[int(self.last_target)] = 1

        gradient = probabilities - one_hot_labels

        return gradient


    def zero_grad(self):
        super().zero_grad()
        self.last_prediction = None
        self.last_target = None