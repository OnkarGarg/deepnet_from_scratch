import numpy as np
import matplotlib.pyplot as plt

from Losses import mse

class Model:
    def __init__(self, layers=None):
        if layers is None:
            self._layers = list()
        else:
            self._layers = layers
        # self._inputs = None

    # @property
    # def inputs(self):
    #     return self._inputs
    #
    # @inputs.setter
    # def inputs(self, inputs):
    #     self._inputs = inputs

    def add_layer(self, layer):
        self._layers.append(layer)

    def _connect_layers(self, training=False):
        for i in range(1, len(self._layers)):
            self._layers[i].input = self._layers[i-1].output(training)

    def _train_layers(self, y_train, learning_rates, index=0):
        if index == len(self._layers) - 1:
            # print(f"self._layers[{index}].train_layer(learning_rates[{index}], y_train)")
            return self._layers[index].train_layer(learning_rates[index], y_train)
        # print(f"self._layers[{index}].train_layer(learning_rates[{index}], self.train_model(y_train, learning_rates, {index + 1}))")
        return self._layers[index].train_layer(learning_rates[index], dy=self._train_layers(y_train, learning_rates, index + 1))

    def train_model(self, x_train, y_train, learning_rates, epochs, x_val=None, y_val=None, graphing=False):

        iteration = []
        train_errors = []
        validation_errors = []

        if graphing:
            plt.ion()
            fig, ax = plt.subplots()
            train_line, = ax.plot([], [], 'bo-', label='Train')  # blue
            val_line, = ax.plot([], [], 'ro-', label='Validation')  # red

        for i in range(epochs):
            self._layers[0].input = x_train
            self._connect_layers(True)

            self._train_layers(y_train, learning_rates)
            train_error = np.sum(mse(self._layers[-1].output(), y_train))

            self._layers[0].input = x_val
            self._connect_layers()
            validation_error = np.sum(mse(self._layers[-1].output(), y_val))

            iteration.append(i)
            train_errors.append(train_error)
            validation_errors.append(validation_error)

            print(train_error, validation_error)

            if graphing:
                train_line.set_xdata(iteration)
                train_line.set_ydata(train_errors)

                val_line.set_xdata(iteration)
                val_line.set_ydata(validation_errors)

                ax.relim()  # Recalculate limits
                ax.autoscale_view()  # Autoscale the axes

                plt.draw()
                plt.pause(0.0000005)  # Short pause to allow update (animation effect)

        if graphing:
            plt.ioff()  # Turn off interactive mode
            plt.show()

        plt.loglog(iteration, train_errors)
        plt.loglog(iteration, validation_errors)
        plt.show()
