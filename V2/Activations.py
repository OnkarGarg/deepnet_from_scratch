import numpy as np


def relu(values):
    return np.maximum(0, values)


def relu_derivative(values):
    return (values > 0).astype(float)


def linear(values):
    return values


def linear_derivative(values):
    return 1.0
