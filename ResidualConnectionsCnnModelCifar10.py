import torch
from torch import nn


class ResidualConnectionsCnnModelCifar10(nn.Module):
    def __init__(
        self,
        input_shape=(64, 3, 32, 32),
        num_classes=10,
    ):
        super(ResidualConnectionsCnnModelCifar10, self).__init__()
        self.start = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.block32 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block64 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block128 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.block256 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.block512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.end = nn.Sequential(nn.AvgPool2d(7, 2))
        # self.block5 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        #     nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(2048),
        #     nn.MaxPool2d(kernel_size=(2, 2)),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.3),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(6400, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

        # Initialize weights
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        # Apply initialization
        self.start.apply(init_weights)
        self.block2.apply(init_weights)
        self.block3.apply(init_weights)
        self.block4.apply(init_weights)
        self.block5.apply(init_weights)
        self.block32.apply(init_weights)
        self.block64.apply(init_weights)
        self.block128.apply(init_weights)
        self.block256.apply(init_weights)
        self.block512.apply(init_weights)
        self.end.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.start(x)  # 3x32x32 -> 32x16x16
        x = self.block32(x)
        rc1 = x
        x = self.block32(x)
        x = x + rc1
        x = self.block2(x)
        x = self.block64(x)
        rc2 = x
        x = self.block64(x)
        x = x + rc2
        x = self.block3(x)
        x = self.block128(x)
        rc3 = x
        x = self.block128(x)
        x = x + rc3
        x = self.block4(x)
        x = self.block256(x)
        rc4 = x
        x = self.block256(x)
        x = x + rc4
        x = self.end(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
