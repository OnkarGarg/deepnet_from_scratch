import numpy as np

class Model:
    def __init__(self, layers=None):
        if layers is None:
            layers = list()
        self._layers = layers

    def add(self, layer):
        self._layers.append(layer)

    def _connect_layers(self):
        for i in range(1, len(self._layers)):
            print(i)
            self._layers[i].input = self._layers[i-1].output()


