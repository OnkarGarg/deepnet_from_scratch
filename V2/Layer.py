class Layer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size  # n
        self._output_size = output_size  # m
        self._input = None  # B x n

    def __str__(self):
        pass

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    def train_layer(self, learning_rate, y_real=None, dy=None):
        pass

    def output(self, training=False):
        pass
