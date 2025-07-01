import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

from DenseLayer import DenseLayer, mse
from Model import Model

model = Model([DenseLayer(8, 16, "relu"), DenseLayer(16, 4, "relu"), DenseLayer(4, 1, "linear")])

model._connect_layers()
