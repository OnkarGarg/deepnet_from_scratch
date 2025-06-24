import numpy as np

from Layers.DenseLayer import DenseLayer

a = DenseLayer(10, 4)
print(a)
print("Output:", a.output("RELU"))