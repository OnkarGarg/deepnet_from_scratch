import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from DenseLayer import DenseLayer, mse

file_path = os.path.join("..", "data", "Concrete_Data.xls")

data = pd.read_excel(file_path)

print(data)

layer = DenseLayer(2, 3)

data = np.array([[1, 2], [3, 4], [5, 6], [7, -8]])
real = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [-4, -4, -4]])

# real = data.iloc[:, -1].to_numpy().reshape(-1, 1)
# data = data.iloc[:, :-1].to_numpy()

print(real, data)

layer.input = data

print(layer.output())
print()
print(layer.bias)
print()
print(layer.weights)

epochs = 10000

x_vals = []
y_vals = []
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-')

for i in range(epochs):
    layer.train_layer(0.001, real)
    out = np.sum(mse(layer.output(), real))

    x_vals.append(i)
    y_vals.append(out)

    line.set_xdata(x_vals)
    line.set_ydata(y_vals)

    ax.relim()           # Recalculate limits
    ax.autoscale_view()  # Autoscale the axes

    plt.draw()
    plt.pause(0.01)      # Short pause to allow update (animation effect)

plt.ioff()  # Turn off interactive mode
plt.show()

print(layer.output())
