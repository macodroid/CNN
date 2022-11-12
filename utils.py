import argparse
import os
import shutil

from matplotlib import pyplot as plt
import toml
import torch
import yaml


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


def create_plot(
    epoch_train_losses,
    epoch_val_losses,
    epoch_val_acc,
    directory,
    test_name,
    save_plot=True,
    display_plot=False,
):
    figure, axis = plt.subplots(2, 1)
    axis[0].plot(epoch_train_losses, c="r")
    axis[0].plot(epoch_val_losses, c="b")
    axis[0].legend(["Train_loss", "Val_loss"])
    axis[0].set_title("Train vs. Validation loss")
    axis[0].set(xlabel="Epoch", ylabel="Loss")
    axis[1].plot(epoch_val_acc)
    axis[1].legend(["Val_acc"])
    axis[1].set_title("Validation accuracy")
    axis[1].set(xlabel="Epoch", ylabel="Accuracy")
    plt.tight_layout()
    if save_plot:
        plt.savefig(f"{directory}/{test_name}.png")
    if display_plot:
        plt.show()


# def model_configuration(path_to_config_file: str) -> dict:
#     config = toml.load(path_to_config_file)
#     return config


def model_configuration(path_to_config_file: str) -> dict:
    return yaml.load(
        open(f"experiments/configs/{path_to_config_file}", "r"), Loader=yaml.FullLoader
    )


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.has_mps:
        return "mps"
    else:
        return "cpu"


def create_experiment_dir(experiment_name: dict, config_name: str) -> str:
    experiments_results_dir = (
        f"{os.path.dirname(os.path.realpath(__file__))}/experiments/results"
    )
    os.mkdir(f"{experiments_results_dir}/{experiment_name}")
    shutil.copyfile(
        f"experiments/configs/{config_name}",
        f"{experiments_results_dir}/{experiment_name}/{config_name}",
    )
    return f"{experiments_results_dir}/{experiment_name}"
