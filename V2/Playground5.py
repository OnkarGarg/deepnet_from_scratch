import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib.pyplot as plt

from DenseLayer import DenseLayer
from DropoutLayer import DropoutLayer
from Model import Model
from Activations import *
from Optimizer import *
from LRScheduler import *

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

print(x_train[0], y_train[0])

learning_rates = [0.00001, 0.00001, 0, 0.00001, 0.00001, 0.00001, 0.00001]

base_model = Model([DenseLayer(8, 64, Relu(), "he_normal"),
               DenseLayer(64, 32, Relu(), "he_normal"),
               DropoutLayer(32, 32, 0.2),
               DenseLayer(32, 64, Relu(), "he_normal"),
               DenseLayer(64, 8, Relu(), "he_normal"),
               DenseLayer(8, 4, Relu(), "he_normal"),
               DenseLayer(4, 1, Linear(), "he_normal")])

experiments = [
    ("GD", None),
    ("Adagrad", AdaGrad()),
    ("RMSprop", RMSProp()),
    ("Adam", Adam()),
]

models = [copy.deepcopy(base_model) for _ in experiments]
histories = {}

for (name, optimizer), model in zip(experiments, models):

    train_errors, val_errors = model.train_model(
        x_train, y_train,
        learning_rates,
        25000,
        x_val, y_val,
        optimizer=optimizer,
        scheduler=None,
        graphing=False
    )
    histories[name] = (train_errors, val_errors)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for ax, (name, (train_errors, val_errors)) in zip(axes.flat, histories.items()):
    ax.plot(train_errors, label="Train", linewidth=2)
    ax.plot(val_errors, label="Validation", linewidth=2)
    ax.set_title(name)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle("Optimizer Comparison", fontsize=14)
plt.tight_layout()
plt.show()