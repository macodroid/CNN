import optimizer_factory
import torch
import utils
from CnnModelCifar10 import CnnModelCifar10
from CustomCnnModelCifar10 import CustomCnnModelCifar10
from ResidualConnectionsCnnModelCifar10 import ResidualConnectionsCnnModelCifar10
from dataset_utils import DataProvider
from dojo import Dojo
from torch import nn
from torchvision import transforms

if __name__ == "__main__":
    args = utils.parse_args()
    config = utils.model_configuration(args.config)
    experiment_directory = utils.create_experiment_dir(
        config["defaults"]["experiment_name"], args.config
    )

    if "augmentation" in config["defaults"] and config["defaults"]["augmentation"]:
        transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.ToTensor()

    data_provide = DataProvider(
        batch_size=config["defaults"]["batch_size"], transform=transform
    )
    train_dl, val_dl, test_dl = data_provide.get_data()

    for X, y in val_dl:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        inp_shape = X.shape
        break
    device = utils.get_device()
    print(device)
    print(f"Using {device} device")

    # define model
    if (
        "class" in config["architecture"]
        and config["architecture"]["class"] == "CustomCnnModelCifar10"
    ):
        model = CustomCnnModelCifar10(
            input_shape=inp_shape,
        )
    elif (
        "class" in config["architecture"]
        and config["architecture"]["class"] == "ResidualConnectionsCnnModelCifar10"
    ):
        model = ResidualConnectionsCnnModelCifar10(
            input_shape=inp_shape,
        )
    else:
        model = CnnModelCifar10(
            convolution_layers=config["architecture"]["convolution_layers"],
            fully_connected_layers=config["architecture"]["fully_connected_layers"],
            input_shape=inp_shape,
        )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    # Get optimization function
    optimizer = optimizer_factory.get_optimizer(config["optimizer"], model=model)

    # if args.scheduler_gamma != 0.0:
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.scheduler_gamma)

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
    with open(
        f"{experiment_directory}/{config['defaults']['experiment_name']}.txt", "w"
    ) as file_write:
        for e in range(config["defaults"]["epochs"]):
            file_write.write(f"\nEpoch {e + 1}\n-------------------------------")
            print(f"Epoch {e + 1}\n-------------------------------")

            train_loss, val_loss, val_acc = trainer.train()

            file_write.write("\nTrain loss at epoch {}: {}".format(e, train_loss))
            print("\nTrain loss at epoch {}: {}".format(e, train_loss))

            file_write.write("\nVal loss at epoch {}: {}".format(e, val_loss))
            print("\nVal loss at epoch {}: {}".format(e, val_loss))

            file_write.write("\nVal acc at epoch {}: {}".format(e, val_acc))
            print("\nVal acc at epoch {}: {}".format(e, val_acc))

            epoch_train_losses.append(train_loss)
            epoch_val_losses.append(val_loss)
            epoch_val_accs.append(val_acc)

            # if args.scheduler_gamma != 0.0:
            #     scheduler.step()

        acc_test = trainer.test()

        file_write.write(
            f"\n-------------------------------\nTest Accuracy of the model: {acc_test * 100:.2f}"
        )
        print(
            f"\n-------------------------------Test Accuracy of the model: {acc_test * 100:.2f}"
        )
        torch.save(
            model, f"{experiment_directory}/{config['defaults']['experiment_name']}.pt"
        )
        utils.create_plot(
            epoch_train_losses,
            epoch_val_losses,
            epoch_val_accs,
            experiment_directory,
            config["defaults"]["experiment_name"],
        )

    print("Done!")
