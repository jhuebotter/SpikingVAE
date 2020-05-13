from src.models.conv_classifier import ConvolutionalNeuralNetwork
import src.utils as u


def main():
    # set important parameters
    parser = u.get_argparser()
    args = parser.parse_args()
    args.dataset = "fashion"
    args.lr = 1e-7
    args.log_interval = 100

    # choose the devices for computation (GPU if available)
    device = u.get_backend(args)

    # make experiments reproducible
    u.set_seed(args.seed)

    # load dataset
    train_loader, test_loader, (width, height, channels) = u.get_datasets(
        args.dataset, args.batch_size, args.cuda, verbose=args.verbose
    )

    # initialize model
    cnn = ConvolutionalNeuralNetwork(
        width,
        height,
        channels,
        args.dataset,
        args.loss,
        args.optim,
        args.lr,
        args.wd,
        device,
        args.log_interval,
        args.verbose,
        args.print_freq,
        n_out=len(train_loader.dataset.targets.unique()),
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
