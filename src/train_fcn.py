from src.models.fc_classifier import FullyConnectedNeuralNetwork
import src.utils as u


def main():

    # set important parameters
    parser = u.get_argparser()
    args = parser.parse_args()
    args.lr = 1e-7
    args.log_interval = 100
    args.seed = 2

    # train the model
    for dataset in ["fashion", "mnist"]:
        args.dataset = dataset
        train_fcn(args)


def train_fcn(args):

    # choose the devices for computation (GPU if available)
    device = u.get_backend(args)

    # make experiments reproducible
    if args.seed:
        u.set_seed(args.seed)

    # load dataset
    train_loader, test_loader, (width, height, channels) = u.get_datasets(
        args.dataset, args.batch_size, args.cuda, verbose=args.verbose
    )

    print(args.verbose)

    # initialize model
    fcn = FullyConnectedNeuralNetwork(
        input_width=width,
        input_height=height,
        input_channels=channels,
        dataset=args.dataset,
        loss=args.loss,
        optimizer=args.optim,
        learning_rate=args.lr,
        weight_decay=args.wd,
        device=device,
        log_interval=args.log_interval,
        print_freq=args.print_freq,
        activation="relu",
        n_out=len(train_loader.dataset.targets.unique()),
        verbose=args.verbose
    )

    # run training
    fcn.run_training(
        train_loader,
        test_loader,
        args.epochs,
        args.log_interval,
        reload_model=args.resume,
        start_epoch=args.start_epoch,
    )


if __name__ == "__main__":
    main()
