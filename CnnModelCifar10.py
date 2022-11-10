import torch
from torch import nn


class CnnModelCifar10(nn.Module):
    def __init__(
        self,
        activation_function,
        convolution_layer_config,
        dense_layer_config,
        input_shape=(64, 3, 32, 32),
        num_classes=10,
    ):
        super(CnnModelCifar10, self).__init__()
        # Set all properties from config
        self.activation_function = activation_function
        self.convolution_layer_config = convolution_layer_config
        self.dense_layer_config = dense_layer_config
        # Build layers
        convolution_layers = self.build_convolution_layers()
        dense_layer = self.build_dense_layer()
        # Assign layers to model
        self.features = nn.Sequential(*convolution_layers)
        self.classifier = nn.Sequential(*dense_layer)

        # Initialize weights
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        # Apply initialization
        self.classifier.apply(init_weights)
        self.features.apply(init_weights)

    def build_convolution_layers(self):
        convolution_layers = []
        convolution_layers.append(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        )
        convolution_layers.append(nn.MaxPool2d(2))
        convolution_layers.append(self.activation_function)
        if (
            self.convolution_layer_config is not None
            and "dropout" in self.convolution_layer_config
        ):
            convolution_layers.append(
                nn.Dropout2d(self.convolution_layer_config["dropout"])
            )

        convolution_layers.append(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        )
        convolution_layers.append(nn.MaxPool2d(2))
        convolution_layers.append(self.activation_function)
        if (
            self.convolution_layer_config is not None
            and "dropout" in self.convolution_layer_config
        ):
            convolution_layers.append(
                nn.Dropout2d(self.convolution_layer_config["dropout"])
            )

        convolution_layers.append(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        )
        convolution_layers.append(nn.MaxPool2d(2))
        convolution_layers.append(self.activation_function)
        if (
            self.convolution_layer_config is not None
            and "dropout" in self.convolution_layer_config
        ):
            convolution_layers.append(
                nn.Dropout2d(self.convolution_layer_config["dropout"])
            )
        return convolution_layers

    def build_dense_layer(self):
        fully_connected_layers = []
        if self.dense_layer_config is not None and "dropout" in self.dense_layer_config:
            fully_connected_layers.append(
                nn.Dropout(self.dense_layer_config["dropout"])
            )
        fully_connected_layers.append(nn.Linear(2048, 10))
        return fully_connected_layers

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
