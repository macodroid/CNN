import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import utils
from CnnModelCifar10 import CnnModelCifar10
from EnumActivationFunction import ActivationFunction
from dataset_utils import DataProvider
from dojo import Dojo

if __name__ == "__main__":
    args = utils.parse_args()
    name_of_test = args.name_of_test
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    data_provide = DataProvider(batch_size=64, transform=transform)
    train_dl, val_dl, test_dl = data_provide.get_data()

    for X, y in val_dl:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        inp_shape = X.shape
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CnnModelCifar10(
        input_shape=inp_shape,
        activation_function=ActivationFunction[args.activation_function],
    )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.scheduler_gamma != 0.0:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.scheduler_gamma)

    trainer = Dojo(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        device=device,
    )

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    for e in range(args.epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loss, val_loss, val_acc = trainer.train()
        print("Val loss at epoch {}: {}".format(e, val_loss))
        print("Val acc at epoch {}: {}".format(e, val_acc))
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_acc)
        if args.scheduler_gamma != 0.0:
            scheduler.step()
    print(f"Test Accuracy of the model: {trainer.test() * 100:.2f}")
    # TODO running some experiments need to change name of the snapshot of model
    # torch.save(model, 'baseline_cifar10.pt')
    plt.plot(epoch_train_losses, c="r")
    plt.plot(epoch_val_losses, c="b")
    plt.legend(["Train_loss", "Val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation loss")
    # plt.show()
    # TODO same here change naming of plot
    # plt.savefig("baseline.png")
    print("Done!")
