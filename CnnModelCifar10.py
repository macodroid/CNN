import torch
from torch import nn

from ActivationFunctionFactory import ActivationFunctionFactory


class CnnModelCifar10(nn.Module):
    def __init__(
        self,
        activation_function,
        input_shape=(64, 3, 32, 32),
        num_classes=10,
    ):
        super(CnnModelCifar10, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.MaxPool2d(2),
            activation_function,
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.MaxPool2d(2),
            activation_function,
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.MaxPool2d(2),
            activation_function,
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=10),
        )

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.classifier.apply(init_weights)
        self.features.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
