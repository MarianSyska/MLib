# MLib

## Description

This is a small ML library created for educational purposes. It serves as a fundamental example to demonstrate machine learning concepts and reinforce personal understanding of the underlying mathematics, functions, algorithms, and objects.

Implemented is a Multi-Layer Perceptron with adjustable activation functions (Sigmoid, ReLU, or Tanh). The depth and number of neurons per layer are adjustable. The model can be trained on either a sin-function or the MNIST-dataset. The sin-function training outputs a plot for visualization, and the MNIST-dataset training prints the final accuracy of the model.

## Installation

Create a new virtual environment:
```
python3 -m venv venv
```

Install dependencies:
```
pip install -r requirements.txt
```

## How to use:

First, activate your virtual environment:
For Windows:
```
.\venv\Scripts\venv
```
For Linux:
```
source venv/bin/activate
```

Then, run the training script:
```
python train.py
```

Arguments:

-d: Specifies the dataset. Options are 'sin' (for the sin-function) or 'mnist' (for the MNIST-dataset). Default: 'sin'.

-e: Sets the number of epochs. Use a low number for MNIST. Default: 100000.

-a: Defines the layer structure. It's a list of integers representing the number of neurons in the hidden layers. For example, 20 20 20 creates a network with three hidden layers, each with 20 neurons. The number of neurons in the input and output layers is determined by the dataset. Default: 128 258 128.

-lr: Sets the learning rate. Default: 1e-3.




