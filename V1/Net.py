from Layers.DenseLayer import  DenseLayer

class Net:
    def __init__(self, layers):
        self._layers = layers
        self._net = []

        # for layer in layers:
        #     if layer == "dense":
        #         self._net.append(DenseLayer(self._net[len(self._net) - 1]))