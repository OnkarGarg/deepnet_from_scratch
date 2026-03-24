import numpy as np
import scipy as sp

class Activation:
    def __init__(self):
        pass

    def function(self, values):
        pass

    def function_derivative(self, values):
        pass


class Relu(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, values):
        return np.maximum(0, values)
    
    def function_derivative(self, values):
        return (values > 0).astype(float)


class Linear(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, values):
        return values

    def function_derivative(self, values):
        return 1.0


class Step(Activation):
    def __init__(self):
        super().__init__()

    def function(self, values):
        return np.where(values < 0, 0, 1)

    def function_derivative(self, values):
        return np.zeros_like(values)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def function(self, values):
        return sp.special.expit(values)

    def function_derivative(self, values):
        sig_vals = self.function(values)
        return sig_vals * (1 - sig_vals)


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def function(self, values):
        return np.tanh(values)

    def function_derivative(self, values):
        return 1 - np.power(self.function(values), 2)


class Softsign(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, values):
        return values/(1 + np.abs(values))

    def function_derivative(self, values):
        return np.power(1/(1 + np.abs(values)), 2)


class Gelu(Activation):
    def __init__(self):
        super().__init__()

    def function(self, values):
        return 0.5 * values * (1 + sp.special.erf(values/np.sqrt(2)))

    def function_derivative(self, values):
        return 0.5 * (1 + sp.special.erf(values/np.sqrt(2))) + values * sp.stats.norm.pdf(values)


class Softplus(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, values):
        return np.log(1 + np.exp(values))
    
    def function_derivative(self, values):
        return sp.special.expit(values)


class Elu(Activation):
    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    def function(self, values):
        return np.where(values > 0, values, self._alpha * (np.exp(values) - 1))
    
    def function_derivative(self, values):
        return np.where(values > 0, 1, self._alpha * np.exp(values))

 
class Selu(Activation):
    def __init__(self, alpha=1.67326, lmbda=1.0507):
        super().__init__()
        self._alpha = alpha
        self._lmbda = lmbda

    def function(self, values):
        return self._lmbda * np.where(values >= 0, values, self._alpha * (np.exp(values) - 1))
    
    def function_derivative(self, values):
        return self._lmbda * np.where(values >= 0, 1, self._alpha * np.exp(values))


class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, values):
        return np.where(values > 0, values, 0.01 * values)
    
    def function_derivative(self, values):
        return np.where(values > 0, 1, 0.01)


class PReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self._alpha = alpha
    
    def function(self, values):
        return np.where(values > 0, values, self._alpha * values)
    
    def function_derivative(self, values):
        return np.where(values > 0, 1, self._alpha)


class SiLU(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, values):
        return values * sp.special.expit(values)
    
    def function_derivative(self, values):
        return sp.special.expit(values) * (1 + values * (1 - sp.special.expit(values)))

 
class Gaussian(Activation):
    def __init__(self):
        super().__init__()

    def function(self, values):
        return np.exp(-np.power(values, 2))
    
    def function_derivative(self, values):
        return -2 * values * np.exp(-np.power(values, 2))


class Sinusoid(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, values):
        return np.sin(values)
    
    def function_derivative(self, values):
        return np.cos(values)


class ELiSH(Activation):
    def __init__(self):
        super().__init__()

    def function(self, values):
        return np.where(values < 0, np.exp(values) - 1, values) * sp.special.expit(values)
    
    def function_derivative(self, values):
        return np.where(values < 0, sp.special.expit(values) * (np.exp(values) + (np.exp(values) - 1) * (1 - sp.special.expit(values))), sp.special.expit(values) * (1 + values * (1 - sp.special.expit(values))))