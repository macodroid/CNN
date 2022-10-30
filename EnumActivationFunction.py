from enum import Enum

from torch import nn


class ActivationFunction(Enum):
    ReLU = nn.ReLU()
    LeakyReLU = nn.LeakyReLU()
    GELU = nn.GELU()
    Sigmoid = nn.Sigmoid()
