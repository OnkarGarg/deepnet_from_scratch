import numpy as np


def relu(values):
    return np.maximum(0, values)


def relu_derivative(values):
    return (values > 0).astype(float)


def linear(values):
    return values


def linear_derivative(values):
    return 1.0


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


class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self._input_size = input_size  # n
        self._output_size = output_size  # m
        self._input = None  # B x n
        self._bias = np.zeros(output_size)  # m
        # self._weights = np.random.rand(output_size, input_size)  # m x n
        self._weights = np.random.normal(0, np.sqrt(2/input_size), (output_size, input_size))  # m x n

        if activation is None:
            self._activation = linear
            self._activation_derivative = linear
        elif activation.lower() == "relu":
            self._activation = relu
            self._activation_derivative = relu_derivative
        elif activation.lower() == "linear":
            self._activation = linear
            self._activation_derivative = linear

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    def train_layer(self, learning_rate, real=None, dy=None):
        y = self.output()
        da = self._activation_derivative(y)

        if dy is None:
            dy = mse_derivative(y, real)

        dw = (da * dy).transpose() @ self._input + 2 * 0.001 * self._weights
        db = np.sum(da * dy, axis=0)
        dx = (da * dy) @ self._weights

        max_grad = 1.0
        dw = np.clip(dw, -max_grad, max_grad)
        db = np.clip(db, -max_grad, max_grad)

        self._weights -= dw * learning_rate
        self._bias -= db * learning_rate

        return dx

    def output(self):
        return self._input @ self._weights.transpose() + self._bias
