import os

import torch
from torch import nn
from torchvision import transforms

import activation_function_factory
import optimizer_factory

import utils
from CnnModelCifar10 import CnnModelCifar10
from dataset_utils import DataProvider
from dojo import Dojo
import shutil

if __name__ == "__main__":
    args = utils.parse_args()
    config = utils.model_configuration(args.config)
    os.mkdir(f"tests/{config['name_of_test']}")
    main_test_dir = f"tests/{config['name_of_test']}"
    shutil.copyfile(args.config, f"{main_test_dir}/{args.config}")
    transform = transforms.ToTensor()
    data_provide = DataProvider(batch_size=config["batch_size"], transform=transform)
    train_dl, val_dl, test_dl = data_provide.get_data()

    for X, y in val_dl:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        inp_shape = X.shape
        break
    device = utils.get_device()
    print(device)
    print(f"Using {device} device")

    # get activation function
    activation_function = activation_function_factory.get_activation_function(
        config["activation_function"]
    )

    # Layers config
    convolution_layer_config = (
        config["convolution_layer"] if "convolution_layer" in config else None
    )
    dense_layer_config = config["dense_layer"] if "dense_layer" in config else None

    # define model
    model = CnnModelCifar10(
        input_shape=inp_shape,
        activation_function=activation_function,
        convolution_layer_config=convolution_layer_config,
        dense_layer_config=dense_layer_config,
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
    with open(f"{main_test_dir}/{config['name_of_test']}.txt", "w") as file_write:
        for e in range(config["epochs"]):
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
        torch.save(model, f"{main_test_dir}/{config['name_of_test']}.pt")
        utils.create_plot(
            epoch_train_losses,
            epoch_val_losses,
            epoch_val_accs,
            main_test_dir,
            config["name_of_test"],
        )

    print("Done!")
