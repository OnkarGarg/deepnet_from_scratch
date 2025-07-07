import numpy as np


def mse(pred, real):
    if np.shape(pred) != np.shape(real):
        print("shape mismatch in loss", np.shape(pred), np.shape(real))
        return None
    return (1 / np.shape(pred)[0]) * (real - pred) ** 2


def mse_derivative(pred, real):
    if np.shape(pred) != np.shape(real):
        print("shape mismatch in loss_derivative", np.shape(pred), np.shape(real))
        return None
    return (2 / np.shape(pred)[0]) * (pred - real)
