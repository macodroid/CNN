from torch import nn


def get_activation_function(activation_function: dict):
    if activation_function["name"] == "relu":
        return nn.ReLU()
    elif activation_function["name"] == "sigmoid":
        return nn.Sigmoid()
    elif activation_function["name"] == "leaky_relu":
        if "negative_slope" in activation_function["name"]:
            return nn.LeakyReLU(negative_slope=activation_function["negative_slope"])
        return nn.LeakyReLU()
    elif activation_function["name"] == "gelu":
        return nn.GELU()
    elif activation_function["name"] == "silu":
        return nn.SiLU()
