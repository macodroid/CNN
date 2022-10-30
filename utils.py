import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="On how many epochs to train/validate model?",
    )
    parser.add_argument(
        "-a",
        "--activation_function",
        type=str,
        default="ReLU",
        help="Which activation function to use? Choose from: ReLU,LeakyReLU,GELU,Sigmoid",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Size of learning rate",
    )
    parser.add_argument(
        "-n",
        "--name_of_test",
        type=str,
        default="baseline",
        help="Name of test. This argument is used for naming PyTorch snapshot and graphs.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )

    return parser.parse_args()
