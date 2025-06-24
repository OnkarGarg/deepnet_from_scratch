import numpy as np
from Activations.Activations import relu, relu_derivative

class DenseLayer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._input = np.random.rand(self._input_size)
        self._bias = np.random.rand(self._output_size)
        self._weights = np.random.rand(self._output_size, self._input_size)

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def bias(self):
        return self._bias

    @property
    def weights(self):
        return self._weights

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def output(self, activation):
        if activation == "RELU":
            return relu(np.dot(self._weights, self._input) + self._bias)

    def __str__(self):
        return "Bias: " + np.array_str(self._bias) + "\nWeights: " + np.array_str(self._weights)
