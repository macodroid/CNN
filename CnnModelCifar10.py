import torch
from torch import nn


class CnnModelCifar10(nn.Module):
    def __init__(
        self,
        convolution_layers,
        fully_connected_layers,
        input_shape=(64, 3, 32, 32),
        num_classes=10,
    ):
        super(CnnModelCifar10, self).__init__()
        self.convolution_architecture = convolution_layers
        self.fully_connected_architecture = fully_connected_layers

        # Build layers
        convolution_layers = self.build_convolution_layers()
        fc_layer = self.build_fully_connected_layer()
        # Assign layers to model
        self.features = nn.Sequential(*convolution_layers)
        self.classifier = nn.Sequential(*fc_layer)

        # Initialize weights
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        # Apply initialization
        self.classifier.apply(init_weights)
        self.features.apply(init_weights)

    def build_convolution_layers(self):
        layers = []
        for layer in self.convolution_architecture:
            if self.convolution_architecture[layer]["type"] == "Conv2d":
                component = getattr(nn, self.convolution_architecture[layer]["type"])(
                    self.convolution_architecture[layer]["in_channels"],
                    self.convolution_architecture[layer]["out_channels"],
                    eval(self.convolution_architecture[layer]["kernel_size"]),
                    eval(self.convolution_architecture[layer]["stride"]),
                    eval(self.convolution_architecture[layer]["padding"]),
                )
            elif (
                self.convolution_architecture[layer]["type"] == "ReLU"
                or self.convolution_architecture[layer]["type"] == "Sigmoid"
                or self.convolution_architecture[layer]["type"] == "Tanh"
            ):
                component = getattr(nn, self.convolution_architecture[layer]["type"])()
            elif self.convolution_architecture[layer]["type"] == "LeakyReLu":
                component = getattr(nn, self.convolution_architecture[layer]["type"])(
                    layer["negative_slope"]
                )
            elif self.convolution_architecture[layer]["type"] == "MaxPool2d":
                component = getattr(nn, self.convolution_architecture[layer]["type"])(
                    eval(self.convolution_architecture[layer]["kernel_size"]),
                )
            elif self.convolution_architecture[layer]["type"] == "Dropout2d":
                component = getattr(nn, self.convolution_architecture[layer]["type"])(
                    self.convolution_architecture[layer]["p"]
                )
            elif self.convolution_architecture[layer]["type"] == "BatchNorm2d":
                component = getattr(nn, self.convolution_architecture[layer]["type"])(
                    self.convolution_architecture[layer]["num_features"]
                )
            else:
                continue
            layers.append(component)
        return layers

    def build_fully_connected_layer(self):
        layers = []
        for layer in self.fully_connected_architecture:
            if self.fully_connected_architecture[layer]["type"] == "Linear":
                component = getattr(
                    nn, self.fully_connected_architecture[layer]["type"]
                )(
                    self.fully_connected_architecture[layer]["in_features"],
                    self.fully_connected_architecture[layer]["out_features"],
                )
            elif (
                self.fully_connected_architecture[layer]["type"] == "ReLU"
                or self.fully_connected_architecture[layer]["type"] == "Sigmoid"
                or self.fully_connected_architecture[layer]["type"] == "Tanh"
            ):
                component = getattr(
                    nn, self.fully_connected_architecture[layer]["type"]
                )()
            elif self.fully_connected_architecture[layer]["type"] == "LeakyReLu":
                component = getattr(
                    nn, self.fully_connected_architecture[layer]["type"]
                )(self.convolution_architecture[layer]["negative_slope"])
            elif self.fully_connected_architecture[layer]["type"] == "Dropout":
                component = getattr(
                    nn, self.fully_connected_architecture[layer]["type"]
                )(self.fully_connected_architecture[layer]["p"])
            elif self.fully_connected_architecture[layer]["type"] == "BatchNorm1d":
                component = getattr(
                    nn, self.fully_connected_architecture[layer]["type"]
                )(self.fully_connected_architecture[layer]["num_features"])
            else:
                continue
            layers.append(component)
        return layers

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
