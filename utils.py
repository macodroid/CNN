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
    parser.add_argument(
        "-sg",
        "--scheduler_gamma",
        type=float,
        default=0.0,
        help="DISCLAIMER if this argument is more then more then 0.0, scheduler is activated otherwise it is not used.",
    )

    return parser.parse_args()
