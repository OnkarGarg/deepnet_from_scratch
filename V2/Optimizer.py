import numpy as np

class Optimizer:
    def __init__(self):
        pass

    def __str__(self):
        return "Optimizer"

    def denseLayerOptimizer(self, dw, db):
        pass

class AdaGrad(Optimizer):
    def __init__(self):
        super().__init__()

        # self._accumulated_squared_weights_grads = 0
        # self._accumulated_squared_bias_grads = 0

        self._accumulated_squared_grads = {}


    def __str__(self):
        return "AdaGrad"

    def denseLayerOptimizer(self, dw, db, learning_rate, layer_id):

        if layer_id not in self._accumulated_squared_grads:
            self._accumulated_squared_grads[layer_id] = {
                'w': np.zeros_like(dw),
                'b': np.zeros_like(db)
            }
        # print(dw)
        # assert not np.any(np.isnan(dw)), "NaN in dw"
        # assert not np.any(np.isnan(db)), "NaN in db"

        # self._accumulated_squared_weights_grads += dw * dw
        # weights = learning_rate * (dw/(np.sqrt(self._accumulated_squared_weights_grads) + 1e-6))

        # self._accumulated_squared_bias_grads += db * db
        # bias = learning_rate * (db/(np.sqrt(self._accumulated_squared_bias_grads) + 1e-6))

        self._accumulated_squared_grads[layer_id]['w'] += dw * dw
        weights = learning_rate * (dw/(np.sqrt(self._accumulated_squared_grads[layer_id]['w']) + 1e-6))

        self._accumulated_squared_grads[layer_id]['b'] += db * db
        bias = learning_rate * (db/(np.sqrt(self._accumulated_squared_grads[layer_id]['b']) + 1e-6))

        return weights, bias


class RMSProp(Optimizer):
    def __init__(self, velocity_decay=0.999):
        super().__init__()

        self._velocity_decay = velocity_decay
        # self._weights_velocity = 0
        # self._bias_velocity = 0

        self._velocities = {}

    def __str__(self):
        return "RMSProp"

    def denseLayerOptimizer(self, dw, db, learning_rate, layer_id):

        if layer_id not in self._velocities:
            self._velocities[layer_id] = {
                'w': np.zeros_like(dw),
                'b': np.zeros_like(db)
            }
        
        self._velocities[layer_id]['w'] = self._velocity_decay * self._velocities[layer_id]['w'] + (1-self._velocity_decay) * dw * dw
        weights = learning_rate * (dw/(np.sqrt(self._velocities[layer_id]['w']) + 1e-6))

        self._velocities[layer_id]['b'] = self._velocity_decay * self._velocities[layer_id]['b'] + (1-self._velocity_decay) * db * db
        bias = learning_rate * (db/(np.sqrt(self._velocities[layer_id]['b']) + 1e-6))

        return weights, bias
    
    
class Adam(Optimizer):
    def __init__(self, velocity_decay=0.999, momentum_decay=0.9):
        super().__init__()

        self._step = 0
        self._velocity_decay = velocity_decay
        self._momentum_decay = momentum_decay

        # self._weights_velocity = 0
        # self._bias_velocity = 0

        self._velocities = {}
        self._momentums = {}

        # self._weights_momentum = 0
        # self._bias_momentum = 0

    def __str__(self):
        return "Adam"
    
    def denseLayerOptimizer(self, dw, db, learning_rate, layer_id):
        
        if layer_id not in self._velocities:
            self._velocities[layer_id] = {
                'w': np.zeros_like(dw),
                'b': np.zeros_like(db)
            }
        
        if layer_id not in self._momentums:
            self._momentums[layer_id] = {
                'w': np.zeros_like(dw),
                'b': np.zeros_like(db)
            }

        self._step += 1
        learning_rate *= np.sqrt(1 - self._velocity_decay**self._step) / (1 - self._momentum_decay**self._step)

        self._momentums[layer_id]['w'] = self._momentum_decay * self._momentums[layer_id]['w'] + (1 - self._momentum_decay) * dw
        self._velocities[layer_id]['w'] = self._velocity_decay * self._velocities[layer_id]['w'] + (1-self._velocity_decay) * dw * dw
        weights = learning_rate * (self._momentums[layer_id]['w']/(np.sqrt(self._velocities[layer_id]['w']) + 1e-6))

        self._momentums[layer_id]['b'] = self._momentum_decay * self._momentums[layer_id]['b'] + (1 - self._momentum_decay) * db
        self._velocities[layer_id]['b'] = self._velocity_decay*self._velocities[layer_id]['b'] + (1-self._velocity_decay) * db * db
        bias = learning_rate * (self._momentums[layer_id]['b']/(np.sqrt(self._velocities[layer_id]['b']) + 1e-6))

        return weights, bias