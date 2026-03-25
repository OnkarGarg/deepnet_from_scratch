import numpy as np
import matplotlib.pyplot as plt

from Losses import mse

class Model:
    def __init__(self, layers=None):
        if layers is None:
            self._layers = list()
        else:
            self._layers = layers

    def add_layer(self, layer):
        self._layers.append(layer)

    def _connect_layers(self, training=False):
        for i in range(1, len(self._layers)):
            self._layers[i].input = self._layers[i-1].output(training)

    def _train_layers(self, y_train, learning_rates, optimizer, index=0):
        if index == len(self._layers) - 1:
            return self._layers[index].train_layer(learning_rates[index], y_train, optimizer)
        return self._layers[index].train_layer(learning_rates[index], optimizer=optimizer, dy=self._train_layers(y_train, learning_rates, optimizer, index + 1))

    def train_model(self, x_train, y_train, learning_rate, epochs, x_val=None, y_val=None, optimizer=None, graphing=False):

        if type(learning_rate) is type([]):
            pass
        elif type(learning_rate) is type(0.0) or type(learning_rate) is type(0):
            learning_rate = [learning_rate] * len(self._layers)
        else:
            print(type(learning_rate), type([]), type(0.0), type(0))
            raise TypeError("Learning rate can only be a list of learning rates for the layers or can be a float/int if you want the same learning rate for all the layers")

        iteration = []
        train_errors = []
        validation_errors = []

        if graphing:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.set_ylabel("Error")
            ax.set_xlabel("Epochs")
            train_line, = ax.plot([], [], 'bo-', label='Train')
            val_line, = ax.plot([], [], 'ro-', label='Validation')
            ax.set_title(f"{optimizer}")
            ax.legend()

        for i in range(epochs):
            self._layers[0].input = x_train
            self._connect_layers(training=True)

            self._train_layers(y_train, learning_rate, optimizer)
            train_error = np.sum(mse(self._layers[-1].output(), y_train))

            self._layers[0].input = x_val
            self._connect_layers(training=False)
            validation_error = np.sum(mse(self._layers[-1].output(), y_val))

            iteration.append(i)
            train_errors.append(train_error)
            validation_errors.append(validation_error)

            print(f"{i}:", train_error, validation_error)

            if graphing:
                train_line.set_xdata(iteration)
                train_line.set_ydata(train_errors)
                val_line.set_xdata(iteration)
                val_line.set_ydata(validation_errors)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)

        if graphing:
            plt.ioff()

            fig2, ax2 = plt.subplots()
            ax2.loglog(iteration, train_errors, label='Train')
            ax2.loglog(iteration, validation_errors, label='Validation')
            ax2.legend()
            ax2.set_title(f"{optimizer}")
            plt.show(block=False)

        return (train_errors, validation_errors)

    def predict(self, x):
            self._layers[0].input = x
            self._connect_layers(training=False)
            y_pred = self._layers[-1].output()
            return y_pred

    def draw_model(self):
        fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
        
        layer_sizes = [layer._output_size for layer in self._layers]
        layer_labels = [layer.__layer_name__() for layer in self._layers]
        connection_modes = [layer._connection for layer in self._layers]
        connection_modes = connection_modes[1:]
        
        v_spacing = 1.0
        h_spacing = 4 * len(self._layers)
        node_radius = 0.15
        
        n_layers = len(layer_sizes)
        x_positions = np.linspace(0, h_spacing * (n_layers - 1), n_layers)
        max_layer_size = max(layer_sizes)
        
        neuron_coords = []
        colors = plt.cm.Set3(np.linspace(0, 1, n_layers))
        
        # Build all neuron coordinates first
        for i, (x, layer_size) in enumerate(zip(x_positions, layer_sizes)):
            y_positions = np.linspace(-v_spacing * (layer_size - 1) / 2,
                                    v_spacing * (layer_size - 1) / 2,
                                    layer_size)
            neuron_coords.append([(x, y) for y in y_positions])
        
        # Draw connections
        for i in range(1, n_layers):
            prev_layer = neuron_coords[i - 1]
            curr_layer = neuron_coords[i]
            mode = connection_modes[i - 1] if i - 1 < len(connection_modes) else "full"
            
            if mode == "full":
                for (x1, y1) in prev_layer:
                    for (x2, y2) in curr_layer:
                        ax.plot([x1 + node_radius, x2 - node_radius], [y1, y2], 
                            color='gray', lw=0.5, alpha=0.3, zorder=1)
            elif mode == "direct":
                for j in range(min(len(prev_layer), len(curr_layer))):
                    x1, y1 = prev_layer[j]
                    x2, y2 = curr_layer[j]
                    ax.plot([x1 + node_radius, x2 - node_radius], [y1, y2], 
                        color='gray', lw=1.5, alpha=0.5, zorder=1)
        
        # Draw neurons and labels
        for i, (x, layer_size) in enumerate(zip(x_positions, layer_sizes)):
            for y in neuron_coords[i]:
                circle = plt.Circle(y, node_radius, fill=True, color=colors[i], 
                                ec='black', linewidth=2, zorder=3, alpha=0.9)
                ax.add_patch(circle)
            
            # Add layer labels with background
            if layer_labels:
                bbox_props = dict(boxstyle='round,pad=0.5', facecolor=colors[i], 
                                alpha=0.7, edgecolor='black', linewidth=2)
                ax.text(x_positions[i], max_layer_size / 2 + 1.2, layer_labels[i],
                    fontsize=14, ha='center', fontweight='bold', bbox=bbox_props)
                # Add layer size info
                ax.text(x_positions[i], -max_layer_size / 2 - 1.0, f'({layer_size})',
                    fontsize=11, ha='center', style='italic', color='gray')
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-2, x_positions[-1] + 2)
        ax.set_ylim(-max_layer_size / 2 - 2, max_layer_size / 2 + 2)
        ax.margins(0.1)
        
        plt.tight_layout()
        plt.show(block=False)

    # def draw_model(self):

    #     fig, ax = plt.subplots(figsize=(25, 6))
    #     print(self._layers)
    #     layer_sizes = [layer._output_size for layer in self._layers]
    #     layer_labels = [layer.__layer_name__() for layer in self._layers]
    #     connection_modes = [layer._connection for layer in self._layers]
    #     connection_modes = connection_modes[1:]

    #     print(layer_sizes, layer_labels, connection_modes)

    #     v_spacing = 0.5
    #     h_spacing = 3*len(self._layers)
    #     node_radius = 0.1

    #     n_layers = len(layer_sizes)
    #     x_positions = np.linspace(0, h_spacing * (n_layers - 1), n_layers)
    #     max_layer_size = max(layer_sizes)

    #     neuron_coords = []

    #     for i, (x, layer_size) in enumerate(zip(x_positions, layer_sizes)):
    #         y_positions = np.linspace(-v_spacing * (layer_size - 1) / 2,
    #                                   v_spacing * (layer_size - 1) / 2,
    #                                   layer_size)
    #         neuron_coords.append([(x, y) for y in y_positions])

    #         for y in y_positions:
    #             circle = plt.Circle((x, y), node_radius, fill=True, color='skyblue', ec='black', zorder=3)
    #             ax.add_patch(circle)

    #         if layer_labels:
    #             ax.text(x, max_layer_size / 2 + 0.5, layer_labels[i],
    #                     fontsize=12, ha='center', fontweight='bold')

    #     for i in range(1, n_layers):
    #         prev_layer = neuron_coords[i - 1]
    #         curr_layer = neuron_coords[i]

    #         mode = connection_modes[i - 1] if connection_modes else "full"

    #         if mode == "full":
    #             for (x1, y1) in prev_layer:
    #                 for (x2, y2) in curr_layer:
    #                     ax.plot([x1 + node_radius, x2 - node_radius], [y1, y2], 'gray', lw=1)
    #         elif mode == "direct":
    #             for j in range(min(len(prev_layer), len(curr_layer))):
    #                 x1, y1 = prev_layer[j]
    #                 x2, y2 = curr_layer[j]
    #                 ax.plot([x1 + node_radius, x2 - node_radius], [y1, y2], 'gray', lw=1)

    #     ax.set_aspect('equal')
    #     ax.axis('off')
    #     ax.set_xlim(-1, x_positions[-1] + 1)
    #     ax.set_ylim(-max_layer_size / 2 - 1, max_layer_size / 2 + 1)

    #     plt.tight_layout()
    #     plt.show()