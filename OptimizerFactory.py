from torch import nn
import torch


class OptimizerFactory:
    def __init__(self, optimizer: dict, model):
        self.optimizer = optimizer
        self.model = model

    def get_optimizer(self):
        if self.optimizer["name"] == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=self.optimizer["learning_rate"]
            )
