# Simple-Autodiff-Engine

A minimal automatic differentiation engine inspired by educational material from advanced machine learning lectures.  
This repository contains both the core implementation provided during the course and several extensions and additional activation functions implemented by me as part of personal exercises and experimentation.

The project demonstrates how computational graphs, backpropagation, and basic neural networks can be built from scratch, without relying on external ML libraries.

## Features

### Autodiff Engine (Value class)
- Computational graph construction  
- Reverse-mode automatic differentiation (backpropagation)  
- Topological sorting for gradient propagation  
- Support for scalar operations:  
  - addition, subtraction, multiplication, division  
  - power operator  
  - exponential and logarithm  
- Built-in activation functions:  
  - tanh  
  - relu  
  - Added by me: sigmoid, leaky_relu, softplus  
- Extendable with more custom functions

### Neural Network Components
- Neuron class with trainable weights and bias  
- Layer class (fully connected)  
- MLP class supporting arbitrary architectures  
- Forward pass using the autodiff engine  
- Training via gradient descent

## Experiments & Notebooks

The included Jupyter notebook demonstrates:
- Training a small MLP on the XOR dataset  
- Manual gradient verification  
- Numerical gradient checking (finite differences)  
- Comparison against PyTorch autograd to validate correctness  

These experiments show that the autodiff engine produces gradients consistent with established frameworks.

## Repository Structure
```
Simple-Autodiff-Engine/
│
├── Value.py                # Automatic differentiation engine
├── MLP.py                  # Minimal neural network implementation
├── xor_training.ipynb      # Notebook with training & gradient checks
└── README.md               # Project documentation
```
