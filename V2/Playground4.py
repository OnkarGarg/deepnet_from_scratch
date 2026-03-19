import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from DenseLayer import DenseLayer
from DropoutLayer import DropoutLayer
from Model import Model

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'Concrete_Data.xls')

data = pd.read_excel(file_path)

y = data.iloc[:, -1].to_numpy().reshape(-1, 1)
x = data.iloc[:, :-1].to_numpy()

scaler = StandardScaler()
y = scaler.fit_transform(y)
x = scaler.fit_transform(x)

split_idx = int(0.8 * len(x))
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

model = Model([DenseLayer(8, 64, "relu", "he_normal"),
               DenseLayer(64, 32, "relu", "he_normal"),
               DenseLayer(32, 64, "relu", "he_normal"),
               DenseLayer(64, 8, "relu", "he_normal"),
               DenseLayer(8, 4, "relu", "he_normal"),
               DenseLayer(4, 1, "linear")])

learning_rates = [0.00001, 0.00001, 0.0001, 0.0001, 0.00001, 0.00001]

model.train_model(x_train, y_train, learning_rates, 10000, x_val, y_val, graphing=True)
