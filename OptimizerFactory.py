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
        elif self.optimizer["name"] == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.optimizer["learning_rate"],
                nesterov=self.optimizer["nesterov"],
                momentum=self.optimizer["momentum"],
            )
        elif self.optimizer["rmsprop"] == "rmsprop":
            return torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.optimizer["learning_rate"],
                momentum=self.optimizer["momentum"],
                alpha=self.optimizer["alpha"],
            )
