import argparse

import numpy as np
import matplotlib.pyplot as plt

import nn
from model import MultiLayerPerceptron
import dataset


def consume_args():
    parser = argparse.ArgumentParser(
                    prog='MLib Train',
                    description='Train a sin curve or the MNIST dataset with simple network')
    
    parser.add_argument('-l', '--layers', type=int, nargs='+', action='store', default=[128, 256, 128], dest='layers')
    parser.add_argument('-a', '--activation', type=str, action='store', default='relu', choices=['relu', 'sigmoid', 'tanh'], dest='activation')
    parser.add_argument('-d', '--dataset', type=str, action='store', default='sin', choices=['sin', 'mnist'], dest='dataset')
    parser.add_argument('-lr', '--learning_rate', type=float, action='store', default=1e-3, dest='lr')
    parser.add_argument('-e', '--epochs', type=int, action='store', default=100000, dest='epochs')
    parser.add_argument('-ep', '--epochs_print', type=int, action='store', default=100, dest='epochs_print')
    
    return parser.parse_args()


def visualize_sin(model):
    vis_data = np.linspace(-2*np.pi, 2*np.pi, 1000, dtype=np.float32).reshape(-1, 1)
    
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    ax.plot(vis_data, np.sin(vis_data), label='sin(x)')
    ax.plot(vis_data, model(vis_data), label='MLP Approximation')
    plt.show()


def print_accuracy(model):
        num_correct = 0
        for x, y in zip(x_test, y_test):
            preds = model(x)
            preds = np.argmax(preds, axis=-1)
            for class_pred, class_y in zip(preds, y):
                if int(class_y) == class_pred:
                    num_correct += 1
        accuracy = num_correct / (x_test.shape[0] * x_test.shape[1])
        print(f'Accuracy: {accuracy}')
        

if __name__ == "__main__":
    
    args = consume_args()
    
    
    # Load Dataset
    x_train, y_train, x_test, y_test, batch_size = dataset.load_sin() if args.dataset == 'sin' else dataset.load_mnist()
    

    # Define Network architecture
    layers = [x_train.shape[-1]]
    layers.extend(args.layers)
    layers.extend([1 if args.dataset == 'sin' else 10])
    
    activation_fn = {'relu': nn.activation.ReLU, 'sigmoid': nn.activation.Sigmoid, 'tanh': nn.activation.Tanh}[args.activation]
    
    def init_weights_xavier(shape):
        weights = (np.random.rand(*shape) * 2 - 1)  * np.sqrt( 6 / (shape[0] + shape[1]))
        return weights
    
    model = MultiLayerPerceptron(layers, bias=True, activation_fn=activation_fn, init_weights=init_weights_xavier)
    
    loss_fn = loss_fn = nn.loss.MSE() if args.dataset == 'sin' else  nn.loss.CrossEntropy()
    
    
    # Train the model
    for epoch in range(args.epochs):
        total_train_loss = 0
        total_test_loss = 0
        
        
        # One epoch 
        for x, y in zip(x_train, y_train):
            
            # Model's prediciton 
            preds = model(x)
            loss = loss_fn(preds, y)
            total_train_loss += loss.mean()
            
            
            # Backpropagtion and Optimization step
            model.backward(loss_fn.backward())
            model.step(args.lr)
            model.zero_grad()
            loss_fn.zero_grad()
        
        
        # Calculate total train loss
        total_train_loss /= x_train.shape[0]
        
        
        # Infer test data
        for x, y in zip(x_test, y_test):
            preds = model(x)
            total_test_loss += loss_fn(preds, y).mean()
        
        
        # Calculate total test loss
        total_test_loss /= x_test.shape[0]
        
        
        # Print Train and Test Result
        if epoch % args.epochs_print == 0:
            print(f"Epoch {epoch}, Train Loss: {total_train_loss}, Test Loss: {total_test_loss}")
    
    
    # Visualize Result
    
    if args.dataset == 'sin':
        visualize_sin(model)
    else:
        print_accuracy(model)
