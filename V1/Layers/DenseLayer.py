import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._input = np.ones(self._input_size)
        self._bias = np.ones(self._output_size)
        self._weights = np.ones((self._output_size, self._input_size))

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

    def output(self):
            return np.dot(self._weights, self._input) + self._bias


    def outputs(self, inputs):
        if np.size(self._input) != np.size(inputs[0]):
            return None
        outputs = []
        for vec in inputs:
            self._input = vec
            outputs.append(self.output())
        return np.vstack(outputs)

    def __str__(self):
        return "Bias: " + np.array_str(self._bias) + "\nWeights: " + np.array_str(self._weights)
