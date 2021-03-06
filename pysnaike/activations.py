"""Activation functions used in Deep Learning."""

import numpy as np


def softmax(x, derivative=False):
    """Softmax activation function.
    
    Activation function often used in the final layer. The probabilities will add up to 1.

    args:
        x (list): Vector with input values.
        derivative (bool): Whether the derivative should be returned instead. Defaults to False.
    """

    exps = np.exp(x - x.max())
    temp = exps / np.sum(exps, axis=0)
    if derivative:
        return temp * (1 - temp)
    return temp


def identity(x, derivative=False):
    """Identity activation function.

    Args:
        x (list): Vector with input values.
        derivative (bool, optional): Whether the derivative should be returned instead. Defaults to False.
    """

    if derivative: return np.full(x.shape, 1)
    else: return x

def sigmoid(x, derivative=False):
    """Non vectorized sigmoid activation function.
    
    The values are treated separately and will have a value between 0 and 1.

    Args:
        x (list): Vector with input values.
        derivative (bool, optional): Whether the derivative should be returned instead. Defaults to False.
    """
    if derivative:
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1+np.exp(-x))

# _relu_not_vect
def _relu_not_vect(x, derivative=False):
    """Non vectorized ReLU activation function.

    Args:
        x (list): Vector with input values.
        derivative (bool, optional): Whether the derivative should be returned instead. Defaults to False.
    """

    if x > 0:
        if derivative:
            return 1
        else: return x
    else:
        return 0

def _leaky_relu_not_vect(x, derivative=False):
    """Non vectorized LeakyReLU activation function.

    Args:
        x (list): Vector with input values.
        derivative (bool, optional): Whether the derivative should be returned instead. Defaults to False.
    """
    a = 0.02
    if x > 0:
        if derivative:
            return 1
        else: return x
    else:
        if derivative:
            return a
        else: return a*x


# Vectorize activation functions
__relu = np.vectorize(_relu_not_vect)
__leaky_relu = np.vectorize(_leaky_relu_not_vect)
# __sigmoid = np.vectorize(_sigmoid_not_vect)


def relu(x, derivative=False):
    """Vectorized ReLU activation function.

    Args:
        x (list): Vector with input values.
        derivative (bool, optional): Whether the derivative should be returned instead. Defaults to False.
    """

    return __relu(x, derivative=derivative)

def leaky_relu(x, derivative=False):
    """Vectorized ReLU activation function.

    Args:
        x (list): Vector with input values.
        derivative (bool, optional): Whether the derivative should be returned instead. Defaults to False.
    """

    return __leaky_relu(x, derivative=derivative)