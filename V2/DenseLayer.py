import numpy as np

from Activations import relu, relu_derivative, linear_derivative, linear
from Layer import Layer
from Losses import mse_derivative


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation=None, weights_initialization=None):
        super().__init__(input_size, output_size)
        self._bias = np.zeros(output_size)  # m
        self._weights = np.random.normal(0, np.sqrt(2/input_size), (output_size, input_size))  # m x n
        self._connection = "full"

        
        if weights_initialization == "he_normal":
            self._weights = np.random.normal(0, np.sqrt(2/self._input_size), (self._output_size, self._input_size))
        elif weights_initialization == "he_uniform":
            self._weights = np.random.uniform(np.sqrt(-6/self._input_size), np.sqrt(6/self._input_size), ((self._output_size, self._input_size)))
        elif weights_initialization == "xavier" or weights_initialization == "glorot":
            self._weights = np.random.normal(0, np.sqrt(2/(self._input_size + self._output_size)), (self._output_size, self._input_size))
        else:
            self._weights = np.ones((self._output_size, self._input_size))

        if activation is None:
            self._activation = linear
            self._activation_derivative = linear_derivative
        elif activation.lower() == "relu":
            self._activation = relu
            self._activation_derivative = relu_derivative
        elif activation.lower() == "linear":
            self._activation = linear
            self._activation_derivative = linear_derivative

    def __str__(self):
        return (f"Dense Layer\n\tInput: {self._input_size}\n"
                f"\tOutput: {self._output_size}\n"
                f"\tActivation: {self._activation}")

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

    def train_layer(self, learning_rate, y_real=None, dy=None):
        y_pred = self.output()
        da = self._activation_derivative(y_pred)

        if dy is None:
            dy = mse_derivative(y_pred, y_real)

        dw = (da * dy).transpose() @ self._input + 2 * 0.001 * self._weights
        db = np.sum(da * dy, axis=0)
        dx = (da * dy) @ self._weights

        max_grad = 1.0
        dw = np.clip(dw, -max_grad, max_grad)
        db = np.clip(db, -max_grad, max_grad)

        self._weights -= dw * learning_rate
        self._bias -= db * learning_rate

        return dx

    def output(self, training=False):
        return self._input @ self._weights.transpose() + self._bias
