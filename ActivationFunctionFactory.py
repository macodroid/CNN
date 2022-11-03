from torch import nn


class ActivationFunction:
    def __init__(self, activation_function, leaky_relu_param=None):
        self.activation_function = activation_function
        self.leaky_relu_param = leaky_relu_param

    def get_activation_function(self):
        if self.activation_function == "ReLU":
            return nn.ReLU()
        elif self.activation_function == "Sigmoid":
            return nn.Sigmoid()
        elif self.activation_function == "LeakyReLU":
            if self.leaky_relu_param is None:
                raise AttributeError("LeakyReLU negative_slope is not defined")
            return nn.LeakyReLU(negative_slope=self.leaky_relu_param)
        elif self.activation_function == "GELU":
            return nn.GELU()
