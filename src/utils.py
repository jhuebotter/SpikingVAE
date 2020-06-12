import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.spiking_layers import LIF_sNeuron, Pooling_sNeuron, LF_Unit


def get_argparser(description="", verbose=True):
    """Gets the default argument parser.

    :param description: str, a name of the experiment / argument parser.
    :param verbose: bool, flag to print statements.

    :return parser: object, ArgumentParser containing the default settings for experiments.
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
        "--model",
        default="scnn_autoencoder",
        type=str,
        help="model used for training or evaluation [fcn_classifier, cnn_classifier] (default: cnn_classifier)",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        type=str,
        help="dataset used for training or evaluation [mnist, fashion] (default: mnist)",
    )
    parser.add_argument(
        "--loss",
        default="mse",
        type=str,
        help="loss function used for training [crossentropy, mse] (default: crossentropy)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[], #["accuracy"],
        type=str,
        help="metrics to calculate on the output of the network (default: [accuracy])",
    )
    parser.add_argument(
        "--key_metric",
        default="validation loss",
        type=str,
        help="metric to monitor for comparing model performance (default: validation_loss)",
    )
    parser.add_argument(
        "--goal",
        default="minimize",
        type=str,
        help="decides if the metric monitored should increase or decrease [maximize, minimize] (default: minimize)",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="optimizer = [adam, sgd] (default: adam)",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        metavar="BS",
        help="mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--epoch_batches",
        default=0,
        type=int,
        metavar="EB",
        help="limits number of batches per epoch; 0 = no limit (default: 0)",
    )
    parser.add_argument(
        "--activation",
        default="lif",
        type=str,
        help="activation function = [softmax, logsoftmax, sigmoid, relu, lif] (default: lif)",
    )
    parser.add_argument(
        "--activation_out",
        default="lif",
        type=str,
        help="activation function for the output layer [softmax, logsoftmax, sigmoid, relu, lif] (default: logsoftmax)",
    )
    parser.add_argument(
        "--pooling",
        default="max",
        type=str,
        help="pooling function = [max, avg] (default: max)",
    )
    parser.add_argument(
        "--conv_channels",
        default="16, 32",
        type=str,
        metavar="CC",
        help="comma separated string with numbers of channels in convolutional layers (default: [16, 32, 64])",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.99,
        help="temporal decay variable for LIF neurons (default: 0.99)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="firing threshold for LIF neurons (default: 1.0)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="timesteps per example for spiking networks (default: 50)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=1,
        help="padding to use in convolutional layers (default: 1)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="stride to use in convolutional layers (default: 1)"
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="kernel size to use in convolutional layers (default: 5)"
    )



    # parser.add_argument(
    #    "--conv_channels",
    #    nargs="+",
    #    default=[16, 32, 64],
    #    type=int,
    #    metavar="CC",
    #    help="number of channels in convolutional layers (default: [16, 32, 64])",
    # )
    parser.add_argument(
        "--hidden_sizes",
        nargs="+",
        default=[100],
        type=int,
        metavar="HS",
        help="number of nodes in fully connected layers (default: [100])",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=0.0,
        type=float,
        metavar="WD",
        help="weight decay (default: 0.0)",
    )
    parser.add_argument(
        "--eval_first",
        action="store_true",
        default=True,
        help="evaluates a model before training starts (default: True)",
    )
    parser.add_argument(
        "--load",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
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
        help="seed for random number generators, set to 0 for not setting a state (default: 42)",
    )
    parser.add_argument(
        "--sample_freq",
        default=150,
        type=int,
        metavar="SF",
        help="step between batches to sample results of; 0 is no sampling (default: 0)",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["plot_reconstruction"],
        type=str,
        help="functions to call on sampled network output during evaluations (default: [plot_reconstruction])",
    )

    if verbose:
        print(f"Initialized argument parser {description} with settings:")
        args = parser.parse_args()
        for arg in vars(args):
            print(arg, getattr(args, arg))

    return parser


def get_fc_layers(input_size, hidden_sizes, output_size, activations, bias=True):
    """Creates a list of dicts with parameters for fully connected layers with the specified dimensions.

    :param input_size: int, size of the flattend input.
    :param hidden_sizes: list, number of nodes in hidden layers.
    :param output_size: int, number of output neurons.
    :param activations: list, flags for the activation fuctions to use.

    :return fc_parameters: list containing dicts with layer parameters.
    """

    n_nodes = [input_size] + hidden_sizes + [output_size]

    fc_parameters = []

    for i in range(0, len(n_nodes) - 1):
        fc_parameters.append(
            dict(
                name=f"fc{i+1}",
                type="fc",
                parameters=dict(
                    in_features=n_nodes[i], out_features=n_nodes[i + 1], bias=bias,
                ),
            )
        )
        fc_parameters.append(
            dict(
                name=f"activation_fc{i+1}", type=f"{activations[i]}", parameters=dict(),
            ),
        )

    return fc_parameters


def get_sfc_layers(input_size, hidden_sizes, output_size, activations, thresholds, decays, bias=False):
    """Creates a list of dicts with parameters for fully connected layers with the specified dimensions.

    :param input_size: int, size of the flattend input.
    :param hidden_sizes: list, number of nodes in hidden layers.
    :param output_size: int, number of output neurons.
    :param activations: list, flags for the activation fuctions to use.

    :return fc_parameters: list containing dicts with layer parameters.
    """

    n_nodes = [input_size] + hidden_sizes + [output_size]

    fc_parameters = []

    for i in range(0, len(n_nodes) - 1):
        fc_parameters.append(
            dict(
                name=f"fc{i+1}",
                type="fc",
                parameters=dict(
                    in_features=n_nodes[i], out_features=n_nodes[i + 1], bias=bias,
                ),
            )
        )
        fc_parameters.append(
            dict(
                name=f"activation_fc{i+1}", type=f"{activations[i]}", parameters=dict(
                    threshold=thresholds[i], decay=decays[i],
                ),
            ),
        )

    return fc_parameters


def get_conv2d_layers(
    input_channels,
    channels,
    kernel_sizes,
    strides,
    paddings,
    activations,
    pooling_funcs,
    pooling_kernels,
    pooling_strides,
    bias=False,
):
    """Creates a list of dicts with parameters for convolutional layers with the specified dimensions.

    :param input_channels: int, number of input channels.
    :param channels: list, number of output channels per conv layer.
    :param kernel_sizes: list, size of kernels to use per conv layer.
    :param strides: list, stride to use per conv layer.
    :param activations: list, flag for activation function to use per conv layer.
    :param pooling_funcs: list, flag for pooling function to use per pooling layer.
    :param pooling_kernels: list, size of kernels to use per pooling layer.
    :param pooling_strides: list, stride to use per pooling layer.

    :return conv_parameters: list, containing dicts with layer parameters.
    """

    n_channels = [input_channels] + channels

    conv_parameters = []

    for i in range(0, len(n_channels) - 1):
        conv_parameters.append(
            dict(
                name=f"conv2d{i+1}",
                type="conv2d",
                parameters=dict(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    bias=bias,
                ),
            )
        )
        conv_parameters.append(
            dict(name=f"activation_conv{i+1}", type=activations[i], parameters=dict(),)
        )
        if pooling_kernels[i] > 1:
            conv_parameters.append(
                dict(
                    name=f"pool{i+1}",
                    type=f"{pooling_funcs[i]}pool2d",
                    parameters=dict(
                        kernel_size=pooling_kernels[i], stride=pooling_strides[i],
                    ),
                )
            )

    return conv_parameters


def get_sconv2d_layers(
    input_channels,
    channels,
    kernel_sizes,
    strides,
    paddings,
    activations,
    pooling_funcs,
    pooling_kernels,
    pooling_strides,
    thresholds,
    decays,
    pool_thresholds,
):
    """Creates a list of dicts with parameters for convolutional layers with the specified dimensions.

    :param input_channels: int, number of input channels.
    :param channels: list, number of output channels per conv layer.
    :param kernel_sizes: list, size of kernels to use per conv layer.
    :param strides: list, stride to use per conv layer.
    :param activations: list, flag for activation function to use per conv layer.
    :param pooling_funcs: list, flag for pooling function to use per pooling layer.
    :param pooling_kernels: list, size of kernels to use per pooling layer.
    :param pooling_strides: list, stride to use per pooling layer.

    :return conv_parameters: list, containing dicts with layer parameters.
    """

    n_channels = [input_channels] + channels

    conv_parameters = []

    for i in range(0, len(n_channels) - 1):
        conv_parameters.append(
            dict(
                name=f"conv2d{i+1}",
                type="conv2d",
                parameters=dict(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    bias=False,
                ),
            )
        )
        conv_parameters.append(
            dict(
                name=f"activation_conv{i+1}",
                type=activations[i],
                parameters=dict(threshold=thresholds[i], decay=decays[i],),
            )
        )
        if pooling_kernels[i] > 1:
            conv_parameters.append(
                dict(
                    name=f"pool{i+1}",
                    type=f"avgpool2d",
                    parameters=dict(
                        kernel_size=pooling_kernels[i], stride=pooling_strides[i],
                    ),
                )
            )
            conv_parameters.append(
                dict(
                    name=f"activation_pool{i + 1}",
                    type="spool",
                    parameters=dict(threshold=pool_thresholds[i]),
                )
            )

    return conv_parameters


def get_convtranspose2d_layers(
    input_channels,
    channels,
    kernel_sizes,
    strides,
    paddings,
    activations,
    unpooling_funcs,
    unpooling_kernels,
    unpooling_strides,
):
    """Creates a list of dicts with parameters for convolutional layers with the specified dimensions.

    :param input_channels: int, number of input channels.
    :param channels: list, number of output channels per conv layer.
    :param kernel_sizes: list, size of kernels to use per conv layer.
    :param strides: list, stride to use per conv layer.
    :param activations: list, flag for activation function to use per conv layer.
    :param pooling_funcs: list, flag for pooling function to use per pooling layer.
    :param pooling_kernels: list, size of kernels to use per pooling layer.
    :param pooling_strides: list, stride to use per pooling layer.

    :return conv_parameters: list, containing dicts with layer parameters.
    """

    n_channels = [input_channels] + channels

    conv_parameters = []

    for i in range(0, len(n_channels) - 1):
        conv_parameters.append(
            dict(
                name=f"convT2d{i+1}",
                type="convt2d",
                parameters=dict(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    bias=False,
                ),
            )
        )
        conv_parameters.append(
            dict(name=f"activation_conv{i+1}", type=activations[i], parameters=dict(),)
        )
        if unpooling_kernels[i] > 1:
            conv_parameters.append(
                dict(
                    name=f"unpool{i+1}",
                    type=f"maxunpool2d",
                    parameters=dict(
                        kernel_size=unpooling_kernels[i], stride=unpooling_strides[i],
                    ),
                )
            )

    return conv_parameters


def get_sconvtranspose2d_layers(
    input_channels,
    channels,
    kernel_sizes,
    strides,
    paddings,
    activations,
    unpooling_funcs,
    unpooling_kernels,
    unpooling_strides,
    thresholds,
    decays,
    pool_thresholds,
):
    """Creates a list of dicts with parameters for convolutional layers with the specified dimensions.

    :param input_channels: int, number of input channels.
    :param channels: list, number of output channels per conv layer.
    :param kernel_sizes: list, size of kernels to use per conv layer.
    :param strides: list, stride to use per conv layer.
    :param activations: list, flag for activation function to use per conv layer.
    :param pooling_funcs: list, flag for pooling function to use per pooling layer.
    :param pooling_kernels: list, size of kernels to use per pooling layer.
    :param pooling_strides: list, stride to use per pooling layer.

    :return conv_parameters: list, containing dicts with layer parameters.
    """

    n_channels = [input_channels] + channels

    conv_parameters = []

    for i in range(0, len(n_channels) - 1):
        conv_parameters.append(
            dict(
                name=f"convT2d{i+1}",
                type="convt2d",
                parameters=dict(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    bias=False,
                ),
            )
        )
        conv_parameters.append(
            dict(
                name=f"activation_conv{i+1}",
                type=activations[i],
                parameters=dict(threshold=thresholds[i], decay=decays[i],),
            )
        )
        if unpooling_kernels[i] > 1:
            conv_parameters.append(
                dict(
                    name=f"unpool{i+1}",
                    type=f"maxunpool2d",
                    parameters=dict(
                        kernel_size=unpooling_kernels[i], stride=unpooling_strides[i],
                    ),
                )
            )
            conv_parameters.append(
                dict(
                    name=f"activation_pool{i + 1}",
                    type="spool",
                    parameters=dict(threshold=pool_thresholds[i]),
                )
            )

    return conv_parameters


def build_layers(parameter_list):
    """Creates a ModuleDict network layers with the specified dimensions.

    :param parameter_list: list, includes dicts with parameters passed to layer initialization.

    :return block: ModuleDict containing torch.nn layer objects.
    """

    block = nn.ModuleDict()

    if parameter_list:
        for layer_parameters in parameter_list:
            type_ = layer_parameters["type"].lower()
            name = layer_parameters["name"]
            params = layer_parameters["parameters"]
            if type_ == "relu":
                layer = nn.ReLU(**params)
            elif type_ == "sigmoid":
                layer = nn.Sigmoid(**params)
            elif type_ == "softmax":
                layer = nn.Softmax(dim=1, **params)
            elif type_ == "logsoftmax":
                layer = nn.LogSoftmax(dim=1, **params)
            elif type_ == "lif":
                layer = LIF_sNeuron(**params)
            elif type_ == "spool":
                layer = Pooling_sNeuron(**params)
            elif type_ == "fc":
                layer = nn.Linear(**params)
            elif type_ == "conv2d":
                layer = nn.Conv2d(**params)
            elif type_ == "convt2d":
                layer = nn.ConvTranspose2d(**params)
            elif type_ == "avgpool2d":
                layer = nn.AvgPool2d(**params)
            elif type_ == "maxpool2d":
                layer = nn.MaxPool2d(**params)
            elif type_ == "maxunpool2d":
                layer = nn.MaxUnpool2d(**params)
            else:
                raise NotImplementedError(
                    f"The layer {layer_parameters['type'].lower()} is not implemented."
                )
            block.update({name: layer})

    return block


def set_seed(seed, verbose=True):
    """Make experiments reproducible.

    :param seed: int, seed to use for random number generation.
    :param verbose: bool, flag to print statements.
    """

    if verbose:
        print(f"Setting random seed to {seed}.\n")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(dataset, batch_size, cuda, root="../data", verbose=True):
    """Load datasets from disk.

    :arg
        dataset: str, name of the dataset to load. currently supported are 'mnist' and 'fashion'.
        batch_size: int, size of each minibatch for the training set.
        cuda: bool, describes if training on GPU is enabled.
        root: str, directory where data is loaded from.
        verbose: bool, flag to print statements.

    :returns
        train_loader: DataLoader object with the training data.
        test_loader: DataLoader object with the test data.
        width: int, size of individual examples along the x-axis.
        height: int, size of individual examples along the y-axis.
        channels: int, number of channels of the data (1 for greyscale, 3 for RGB).
    """

    if verbose:
        print(f"Loading {dataset} dataset...")
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

    print(dataset_path)
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
    """Checks for available GPU and chooses the hardware device.

    :param args: object, arugments parsed by ArgumentParser.

    :return device: object, torch.device to use.

    """

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


def get_loss_function(loss, verbose=True, spiking=False):
    """Sets the requested loss function if available

    :param loss: str, name of the loss function to use during training.
    :param verbose: bool, flag to print statements.

    :return loss_function: func, loss function to use during training.
    """

    if loss.lower() == "crossentropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss.lower() == "mse":
        if spiking:
            loss_function = torch.nn.MSELoss(size_average=False)
        else:
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
    """Sets the requested optimizer if available.

    :param optim: str, name of optimizer to use.
    :param model: object, model to optimize.
    :param lr: float, learning rate to use for training.
    :param wd: float, weight decay to use for training.
    :param verbose: bool, flag to print statements.

    :return optimizer: object, torch.optim object.
    """

    if optim.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError(
            f"The optimizer {optim} is not implemented.\n"
            f"Valid options are: 'adam', 'sgd'."
        )
    if verbose:
        print(f"Initialized optimizer:\n{optimizer}\n")

    return optimizer


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


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

    model = torch.nn.Module()

    print("Testing initialization of convolutional layers...")
    conv_args = dict(
        input_channels=1,
        channels=[16, 23],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        activations=["relu", "relu"],
        pooling_funcs=["max", "avg"],
        pooling_kernels=[2, 2],
        pooling_strides=[2, 2],
    )
    conv_parameters = get_conv2d_layers(**conv_args)
    model.conv_layers = build_layers(conv_parameters)
    print("Convolutional layer initialization test complete!\n")

    print("Testing initialization of fully connected layers...")
    print(args.hidden_sizes)
    fc_args = dict(
        input_size=1000,
        hidden_sizes=args.hidden_sizes,
        output_size=10,
        activations=[args.activation for i in range(len(args.hidden_sizes))]
        + ["softmax"],
    )
    fc_parameters = get_fc_layers(**fc_args)  # args.hidden_sizes, 10)
    model.fc_layers = build_layers(conv_parameters)
    print("Fully connected layer initialization test complete!\n")

    print(model)

    print("Testing optimizer initialization...")
    optimizer = get_optimizer(args.optim, model, args.lr, args.wd)
    print("Optimizer initialization test complete!\n")
