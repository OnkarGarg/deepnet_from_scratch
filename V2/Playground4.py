import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

from DenseLayer import DenseLayer
from DropoutLayer import DropoutLayer
from Model import Model
from Activations import *
from Optimizer import *
from LRScheduler import *

# base_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(base_dir, '..', 'data', 'Concrete_Data.xls')

# data = pd.read_excel(file_path)

# y = data.iloc[:, -1].to_numpy().reshape(-1, 1)
# X = data.iloc[:, :-1].to_numpy()

X, y = make_regression(n_samples=5000, n_features=8, noise=0.9, random_state=42)
X = StandardScaler().fit_transform(X)

y = y.reshape(-1, 1)

scaler = StandardScaler()
y = scaler.fit_transform(y)
X = scaler.fit_transform(X)

split_idx = int(0.8 * len(X))
x_train, x_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(x_train[0], y_train[0])
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

model = Model([DenseLayer(8, 64, Relu(), "he_normal", L2_strength=0.001),
               DenseLayer(64, 32, Relu(), "he_normal", L2_strength=0.001),
               DropoutLayer(32, 32, 0.2),
               DenseLayer(32, 64, Relu(), "he_normal", L2_strength=0.001),
               DenseLayer(64, 8, Relu(), "he_normal", L2_strength=0.001),
               DenseLayer(8, 4, Relu(), "he_normal", L2_strength=0.001),
               DenseLayer(4, 1, Linear(), "he_normal", L2_strength=0.001, L1_strength=0.001)])

# model.draw_model()

learning_rates = [0.00001, 0.00001, 0, 0.00001, 0.00001, 0.00001, 0.00001]

model.train_model(x_train=x_train, y_train=y_train, learning_rate=learning_rates, epochs=1000, x_val=x_val, y_val=y_val, batch_size=40, drop_last=False, optimizer=Adam(), graphing=True)

test_input = np.array((3.47791782, 0.85688631, -0.85713204, -0.91866329, -0.62922529, 0.86916012, -1.91765845, -0.29973311))
expected_output = 2.64519215

print(model.predict(test_input))