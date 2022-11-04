import argparse

from matplotlib import pyplot as plt
import toml
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="Path to the config file",
    )
    # parser.add_argument(
    #     "-a",
    #     "--activation_function",
    #     type=str,
    #     default="ReLU",
    #     help="Which activation function to use? Choose from: ReLU,LeakyReLU,GELU,Sigmoid",
    # )
    # parser.add_argument(
    #     "-lr",
    #     "--learning_rate",
    #     type=float,
    #     default=0.001,
    #     help="Size of learning rate",
    # )
    # parser.add_argument(
    #     "-n",
    #     "--name_of_test",
    #     type=str,
    #     default="baseline",
    #     help="Name of test. This argument is used for naming PyTorch snapshot and graphs.",
    # )
    # parser.add_argument(
    #     "-bs",
    #     "--batch_size",
    #     type=int,
    #     default=64,
    #     help="Batch size",
    # )
    # parser.add_argument(
    #     "-sg",
    #     "--scheduler_gamma",
    #     type=float,
    #     default=0.0,
    #     help="DISCLAIMER if this argument is more then more then 0.0, scheduler is activated otherwise it is not used.",
    # )
    # parser.add_argument(
    #     "-lrns",
    #     "--leaky_relu_ns",
    #     type=float,
    #     default=0.01,
    #     help="Leaky relu negative slope",
    # )

    return parser.parse_args()


def create_plot(epoch_train_losses, epoch_val_losses, directory, test_name):
    plt.plot(epoch_train_losses, c="r")
    plt.plot(epoch_val_losses, c="b")
    plt.legend(["Train_loss", "Val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation loss")
    plt.savefig(f"{directory}/{test_name}.png")


def model_configuration(path_to_config_file: str) -> dict:
    config = toml.load(path_to_config_file)
    return config


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.has_mps:
        return "mps"
    else:
        return "cpu"
