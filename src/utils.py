import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_argparser(description=""):
    """Gets the default argument parser.

    :param:
        description: str, a name of the experiment / argument parser.

    :returns:
        parser: ArgumentParser object, contains the default settings for experiments.
    """

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training (default: False)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="enables printing messages (default: True)",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        type=str,
        help="dataset = [mnist, fashion] (default: mnist)",
    )
    parser.add_argument(
        "--loss",
        default="crossentropy",
        type=str,
        help="loss function = [crossentropy, mse] (default: crossentropy)",
    )
    parser.add_argument(
        "--optim",
        default="adam",
        type=str,
        help="optimizer = [adam, sgd] (default: adam)",
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number, useful on restarts (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        metavar="BS",
        help="mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="WD",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        metavar="LOG",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=500,
        type=int,
        metavar="PF",
        help="print frequency (default: 500)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="only evaluate model on validation set (default: False)",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        metavar="S",
        help="seed for random number generators (default: 42)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="applies normalization (default: False)",
    )

    return parser


def set_seed(seed, verbose=True):
    """Make experiments reproducible.

    :param seed: int, seed to use for random number generation.
    """

    if verbose: print(f"Setting random seed to {seed}.\n")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(dataset, batch_size, cuda, root="../data", verbose=True):
    """Load datasets from disk.

    :arg
        dataset: str, name of the dataset to load. currently supported are 'mnist' and 'fashion'.
        batch_size: int, size of each minibatch for the training set.
        cuda: bool, describes if training on GPU is enabled.

    :returns
        train_loader: DataLoader object with the training data.
        test_loader: DataLoader object with the test data.
        width: int, size of individual examples along the x-axis.
        height: int, size of individual examples along the y-axis.
        channels: int, number of channels of the data (1 for greyscale, 3 for RGB).
    """

    if verbose: print(f"Loading {dataset} dataset...")
    if dataset == "fashion":
        Dataset = datasets.FashionMNIST
        dataset_path = Path.joinpath(Path(root), "fashion-mnist")
        width, height, channels = 28, 28, 1
    elif dataset == "mnist":
        Dataset = datasets.MNIST
        dataset_path = Path.joinpath(Path(root), "mnist")
        width, height, channels = 28, 28, 1
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    train_loader = DataLoader(
        Dataset(
            dataset_path, train=True, download=False, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = DataLoader(
        Dataset(
            dataset_path, train=False, download=False, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    if verbose:
        print("Train set:")
        print(train_loader.dataset)
        print("Train Data shape:")
        print(train_loader.dataset.data.shape)
        print("Test set:")
        print(test_loader.dataset)
        print("Test Data shape:")
        print(test_loader.dataset.data.shape)
        print("Dataset loaded successfully!\n")

    return train_loader, test_loader, (width, height, channels)


def get_backend(args):
    """Checks for available GPU"""

    if args.verbose:
        print("Initializing hardware devices...")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:0")
        args.cuda = True
    else:
        device = torch.device("cpu")
        args.cuda = False
    args.device = device
    if args.verbose:
        print(f"Found GPUs: {torch.cuda.device_count()} ")
        print(f"Running on {device}\n")

    return device


def get_loss_function(loss, verbose=True):
    """Sets the requested loss function if available"""

    if loss.lower() == "crossentropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss.lower() == "mse":
        loss_function = torch.nn.MSELoss()
    else:
        raise NotImplementedError(
            f"The loss function {loss} is not implemented.\n"
            f"Valid options are: 'crossentropy', 'mse'."
        )
    if verbose:
        print(f"Initialized loss function:\n{loss_function}\n")

    return loss_function


def get_optimizer(optim, model, lr, wd, verbose=True):
    """Sets the requested optimizer if available"""

    if optim.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd
        )
    elif optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd
        )
    else:
        raise NotImplementedError(
            f"The optimizer {optim} is not implemented.\n"
            f"Valid options are: 'adam', 'sgd'."
        )
    if verbose:
        print(f"Initialized optimizer:\n{optimizer}\n")

    return optimizer


if __name__ == "__main__":
    """Test for the implementations above."""

    print("Testing argparser...")
    parser = get_argparser("Test")
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\nArgparser test compelte!\n")

    print("Testing seed setter...")
    set_seed(args.seed)


    sets = ["mnist", "fashion"]
    for dataset in sets:
        print(f"Testing dataset imports for {dataset}...")
        train_loader, test_loader, (width, height, channels) = get_datasets(
            dataset, 32, False, root="../data"
        )
        print("Dataset import successful!\n")
    print("Dataset tests complete!\n")

    print("Testing device detection...")
    device = get_backend(args)
    print("Device detection test complete!\n")

    print("Testing loss function initialization...")
    loss_function = get_loss_function(args.loss)
    print("Loss function initialization test complete!\n")

    print("Testing optimizer initialization...")
    model = torch.nn.Module()
    model.fc1 = torch.nn.Linear(2, 2)
    optimizer = get_optimizer(args.optim, model, args.lr, args.wd)
    print("optimizer initialization test complete!\n")