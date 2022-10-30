import torch
import torchvision


class DataProvider:
    def __init__(self, batch_size, transform):
        self.batch_size = batch_size
        self.transform = transform

    def get_data(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        train_set, val_set = torch.utils.data.random_split(
            dataset, [40000, 10000], generator=torch.Generator().manual_seed(42)
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_set, batch_size=self.batch_size, shuffle=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False
        )

        return train_dataloader, val_dataloader, test_dataloader
