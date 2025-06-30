import numpy as np

from Activations.Activations import relu_derivative, relu
from Layers.DenseLayer import DenseLayer


def loss(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    if np.size(y_real) != np.size(y_pred):
        return None
    return (1 / np.size(y_real)) * np.sum((y_real - y_pred) ** 2)


def loss_derivative(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    # print(np.size(y_real), np.size(y_pred))

    if np.size(y_real) != np.size(y_pred):
        return None
    return (2 / np.size(y_real)) * np.sum(y_pred - y_real)


def train_layer(layer: DenseLayer, inputs, true_values, learning_rate):

    layer_outputs = layer.outputs(inputs)
    layer_outputs_after_activation = relu(layer_outputs)

    print("LAYER OUTPUTS", layer_outputs_after_activation)

    dy = loss_derivative(true_values, layer_outputs_after_activation) * np.array(relu_derivative(layer_outputs))

    dw = dy * layer.input
    dw = np.mean(dw, axis=0, keepdims=True)
    dx = dy * layer.weights
    dx = np.mean(dx, axis=0, keepdims=True)
    db = dy
    db = np.mean(db, axis=0, keepdims=True)

    layer.bias = layer.bias - (db * learning_rate)
    layer.weights = layer.weights - (dw * learning_rate)

    layer_outputs = layer.outputs(inputs)
    layer_outputs_after_activation = relu(layer_outputs)

    print("LAYER OUTPUTS", layer_outputs_after_activation)

    return dx