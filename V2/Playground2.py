import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

from DenseLayer import DenseLayer, mse

file_path = os.path.join("..", "data", "Concrete_Data.xls")

data = pd.read_excel(file_path)

y = data.iloc[:, -1].to_numpy().reshape(-1, 1)
x = data.iloc[:, :-1].to_numpy()

scaler = StandardScaler()
y = scaler.fit_transform(y)
x = scaler.fit_transform(x)

print(data)

layer1 = DenseLayer(8, 64, "relu")
layer2 = DenseLayer(64, 32, "relu")
layer3 = DenseLayer(32, 8, "relu")
layer4 = DenseLayer(8, 4, "relu")
layer5 = DenseLayer(4, 1, "linear")

split_idx = int(0.8 * len(x))
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

epochs = 20000

iteration = []
train_errors = []
validation_errors = []

plt.ion()
fig, ax = plt.subplots()
train_line, = ax.plot([], [], 'bo-', label='Train')  # blue
val_line, = ax.plot([], [], 'ro-', label='Validation')  # red

for i in range(epochs):
    layer1.input = x_train
    layer2.input = layer1.output()
    layer3.input = layer2.output()
    layer4.input = layer3.output()
    layer5.input = layer4.output()

    layer1.train_layer(0.00001,
                       layer2.train_layer(0.00001,
                                          layer3.train_layer(0.0001,
                                                             layer4.train_layer(0.0001,
                                                                                layer5.train_layer(0.0001,
                                                                                                   y_train)))))
    train_error = np.sum(mse(layer5.output(), y_train))

    layer1.input = x_val
    layer2.input = layer1.output()
    layer3.input = layer2.output()
    layer4.input = layer3.output()
    layer5.input = layer4.output()
    validation_error = np.sum(mse(layer5.output(), y_val))

    iteration.append(i)
    train_errors.append(train_error)
    validation_errors.append(validation_error)

    print(train_error, validation_error)

    train_line.set_xdata(iteration)
    train_line.set_ydata(train_errors)

    val_line.set_xdata(iteration)
    val_line.set_ydata(validation_errors)

    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Autoscale the axes

    plt.draw()
    plt.pause(0.0000005)  # Short pause to allow update (animation effect)

plt.ioff()  # Turn off interactive mode
plt.show(block=False)
