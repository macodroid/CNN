from torch import nn


class ActivationFunctionFactory:
    def __init__(self, activation_function: dict):
        self.activation_function = activation_function

    def get_activation_function(self):
        if self.activation_function["name"] == "relu":
            return nn.ReLU()
        elif self.activation_function["name"] == "sigmoid":
            return nn.Sigmoid()
        elif self.activation_function["name"] == "leaky_relu":
            if "negative_slope" in self.activation_function["name"]:
                return nn.LeakyReLU(
                    negative_slope=self.activation_function["negative_slope"]
                )
            return nn.LeakyReLU()
        elif self.activation_function["name"] == "gelu":
            return nn.GELU()
        elif self.activation_function["name"] == "silu":
            return nn.SiLU()
