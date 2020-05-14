from models.conv_classifier import ConvolutionalNeuralNetwork
import utils as u


def main():

    # set important parameters
    parser = u.get_argparser()
    args = parser.parse_args()
    args.lr = 1e-7

    # train the model
    for dataset in ["fashion", "mnist"]:
        args.dataset = dataset
        train_cnn(args)


def train_cnn(args):
    """Creates and trains a convolutional network classifier.

    :param args: object, parser ArgParser arguments.
    """

    # choose the devices for computation (GPU if available)
    device = u.get_backend(args)

    # make experiments reproducible
    if args.seed:
        u.set_seed(args.seed)

    # load dataset
    train_loader, test_loader, (width, height, channels) = u.get_datasets(
        dataset=args.dataset,
        batch_size=args.batch_size,
        cuda=args.cuda,
        verbose=args.verbose
    )

    # initialize model
    cnn = ConvolutionalNeuralNetwork(
        input_width=width,
        input_height=height,
        input_channels=channels,
        hidden_sizes=args.hidden_sizes,
        conv2d_channels=args.conv_channels,
        dataset=args.dataset,
        loss=args.loss,
        optimizer=args.optim,
        learning_rate=args.lr,
        weight_decay=args.wd,
        device=device,
        log_interval=args.log_interval,
        print_freq=args.print_freq,
        kernel_size=3,
        stride=1,
        pooling_kernel=2,
        pooling_stride=1,
        activation=args.activation,
        pooling=args.pooling,
        n_out=len(train_loader.dataset.targets.unique()),
        verbose=args.verbose,
    )

    # run training
    cnn.run_training(
        train_loader,
        test_loader,
        args.epochs,
        args.log_interval,
        reload_model=args.resume,
        start_epoch=args.start_epoch,
    )


if __name__ == "__main__":
    main()
