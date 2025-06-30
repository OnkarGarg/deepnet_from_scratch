import numpy as np


def relu(x):
    x = np.array(x)
    return np.maximum(x, 0)


def relu_derivative(x):
    x = np.array(x)
    return (x > 0).astype(float)