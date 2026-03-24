import numpy as np

from Layer import Layer


class DropoutLayer(Layer):
    def __init__(self, input_size, output_size, dropout_prob):
        super().__init__(input_size, output_size)
        self._prob = dropout_prob
        self._mask = np.random.binomial(1, 1 - dropout_prob, output_size)
        self._connection = "direct"
        # print(self._mask)

    def __str__(self):
        return (f"Dropout Layer\n\tInput: {self._input_size}\n"
                f"\tOutput: {self._output_size}")

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def train_layer(self, learning_rate, y_real=None, optimizer=None, velocity_decay=None, momentum_decay=None, dy=None):
        return (dy/(1 - self._prob)) * self._mask

    def output(self, training=False):
        if training:
            return (self._input * self._mask)/(1 - self._prob)
        return self._input
