import numpy as np

from Layers.DenseLayer import DenseLayer
from Activations.Activations import relu
from Training import train_layer

layer = DenseLayer(10, 1)
# print(layer)

inputs = [np.array([17, -2,73, 4, 5, 76, 7,78, 9, 10]),
          np.array([1, 2, 3,-84, 5, 6, 7, -8, 8, 18]),
          np.array([1, 2, 1, 4, 5, 6, 1, -1, 1, -10]),
          np.array([1, 2, 3, -4, 5, -3, 3, -3, 9, 10])]

real_vals = [27.7, 4.2, 1.0, 2.3]

# print("Output:", layer.outputs(inputs))
# print("After ReLu", relu(layer.outputs(inputs)))

for _ in range(100):
    print(real_vals)
    train_layer(layer, inputs, real_vals, 0.01)
    print("weights", layer.weights)
    print("bias", layer.bias)