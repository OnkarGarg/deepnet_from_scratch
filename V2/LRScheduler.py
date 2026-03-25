import numpy as np

class LRScheduler:
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    def update(self):
        return self._learning_rate


class StepLRScheduler(LRScheduler):
    def __init__(self, learning_rates, decay, steps):
        super().__init__(learning_rates)

        self._step = 0
        self._initial_learning_rates = learning_rates
        self._learning_rates = learning_rates
        self._decays = decay
        self._layers = 0
        self._steps = steps
    
    def check(self, layers):
        self._layers = layers
        if type(self._decays) is type([]):
            pass
        elif type(self._decays) is type(0.0) or type(self._decays) is type(0):
            self._decays = [self._decays] * self._layers
        else:
            print(type(self._decays), type([]), type(0.0), type(0))
            raise TypeError("Decay can only be a list of decays for the layers or can be a float/int if you want the same decay for all the layers")

    def update(self, val_loss):
        self._learning_rates = self._initial_learning_rates * np.power(self._decays, np.floor(self._step/self._steps))
        self._step += 1
        return self._learning_rates


class ExponentialLRScheduler(LRScheduler):
    def __init__(self, learning_rates, decays):
        super().__init__(learning_rates)
        
        self._step = 0
        self._learning_rates = learning_rates
        self._initial_learning_rates = learning_rates
        self._decays = decays

    def check(self, layers):
        self._layers = layers
        if type(self._decays) is type([]):
            pass
        elif type(self._decays) is type(0.0) or type(self._decays) is type(0):
            self._decays = [self._decays] * self._layers
        else:
            print(type(self._decays), type([]), type(0.0), type(0))
            raise TypeError("Decay can only be a list of decays for the layers or can be a float/int if you want the same decay for all the layers")
    
    def update(self, val_loss):
        self._learning_rates = self._initial_learning_rates * np.power(self._decays, self._step)
        self._step += 1
        return self._learning_rates
    

class CosineAnnealingLRScheduler(LRScheduler):
    def __init__(self, learning_rates, min_learning_rates, max_step):
        super().__init__(learning_rates)
        
        self._step = 0
        self._initial_learning_rates = learning_rates
        self._learning_rates = learning_rates
        self._min_learning_rates = min_learning_rates
        self._max_step = max_step

    def check(self, layers):
        self._layers = layers
        if type(self._min_learning_rates) is type([]):
            pass
        elif type(self._min_learning_rates) is type(0.0) or type(self._min_learning_rates) is type(0):
            self._min_learning_rates = [self._min_learning_rates] * self._layers
        else:
            print(type(self._min_learning_rates), type([]), type(0.0), type(0))
            raise TypeError("Minimum Learning Rates can only be a list of minimum learning rates for the layers or can be a float/int if you want the same minimum learning rate for all the layers")
        
        self._initial_learning_rates = np.array(self._initial_learning_rates)
        self._min_learning_rates = np.array(self._min_learning_rates)

    def update(self, val_loss):
        self._step = min(self._step, self._max_step)
        self._learning_rates = self._min_learning_rates + 0.5 * (self._initial_learning_rates - self._min_learning_rates) * (1 + np.cos((np.pi*self._step)/self._max_step))
        self._step += 1
        return self._learning_rates


class CosineAnnealingWarmRestartsLRScheduler(LRScheduler):
    def __init__(self, learning_rates, min_learning_rates, max_step):
        super().__init__(learning_rates)
        
        self._step = 0
        self._initial_learning_rates = learning_rates
        self._learning_rates = learning_rates
        self._min_learning_rates = min_learning_rates
        self._max_step = max_step

    def check(self, layers):
        self._layers = layers
        if type(self._min_learning_rates) is type([]):
            pass
        elif type(self._min_learning_rates) is type(0.0) or type(self._min_learning_rates) is type(0):
            self._min_learning_rates = [self._min_learning_rates] * self._layers
        else:
            print(type(self._min_learning_rates), type([]), type(0.0), type(0))
            raise TypeError("Minimum Learning Rates can only be a list of minimum learning rates for the layers or can be a float/int if you want the same minimum learning rate for all the layers")
        
        self._initial_learning_rates = np.array(self._initial_learning_rates)
        self._min_learning_rates = np.array(self._min_learning_rates)

    def update(self, val_loss):
        self._learning_rates = self._min_learning_rates + 0.5 * (self._initial_learning_rates - self._min_learning_rates) * (1 + np.cos((np.pi*self._step)/self._max_step))
        self._step += 1
        return self._learning_rates


class ReduceOnPlateauLRScheduler(LRScheduler):
    def __init__(self, learning_rates, decay, min_delta, max_patience):
        super().__init__(learning_rates)
        
        self._learning_rates = learning_rates
        self._decays = decay
        self._min_delta = min_delta
        self._best_loss = float('inf')
        self._max_patience = max_patience
        self._patience_counter = 0

    def check(self, layers):
        self._layers = layers
        if type(self._decays) is type([]):
            pass
        elif type(self._decays) is type(0.0) or type(self._decays) is type(0):
            self._decays = [self._decays] * self._layers
        else:
            print(type(self._decays), type([]), type(0.0), type(0))
            raise TypeError("Decay can only be a list of decays for the layers or can be a float/int if you want the same decay for all the layers")
        
        self._decays = np.array(self._decays)

    def update(self, val_loss):
        if val_loss < self._best_loss - self._min_delta:
            self._best_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            if self._patience_counter >= self._max_patience:
                self._learning_rates = self._learning_rates * self._decays
                self._patience_counter = 0

        return self._learning_rates