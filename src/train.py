from logger import WandBLogger
import utils as u

def main():

    # get parameters
    parser = u.get_argparser()
    args = parser.parse_args()

    args.samplers = ["plot_output_potential", "plot_cummulative_potential", "plot_output_spikes", "plot_reconstruction"]
    args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
    args.metrics = []

    # choose the devices for computation (GPU if available)
    device = u.get_backend(args)

    # initialize logger
    logger = WandBLogger(
        args=args,
        name=args.model,
    )

    # make experiments reproducible
    if args.seed:
        u.set_seed(args.seed)

    # load dataset
    train_loader, val_loader, (width, height, channels) = u.get_datasets(
        dataset=args.dataset,
        batch_size=args.batch_size,
        cuda=args.cuda,
        verbose=args.verbose
    )

    # get model
    if args.model.lower() == "fcn_classifier":
        from models.fcn_classifier import FullyConnectedClassifier
        net = FullyConnectedClassifier(
            input_width=width,
            input_height=height,
            input_channels=channels,
            hidden_sizes=args.hidden_sizes,
            dataset=args.dataset,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            activation=args.activation,
            activation_out=args.activation_out,
            n_out=len(train_loader.dataset.targets.unique()),
            verbose=args.verbose,
        )
    
    elif args.model.lower() == "cnn_classifier":
        from models.cnn_classifier import ConvolutionalClassifier
        net = ConvolutionalClassifier(
            input_width=width,
            input_height=height,
            input_channels=channels,
            hidden_sizes=args.hidden_sizes,
            conv2d_channels=args.conv_channels,
            dataset=args.dataset,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pooling_kernel=2,
            pooling_stride=1,
            activation=args.activation,
            activation_out=args.activation_out,
            pooling=args.pooling,
            n_out=len(train_loader.dataset.targets.unique()),
            verbose=args.verbose,
        )

    elif args.model.lower() == "scnn_classifier":
        from models.scnn_classifier import SpikingConvolutionalClassifier
        net = SpikingConvolutionalClassifier(
            input_width=width,
            input_height=height,
            input_channels=channels,
            conv2d_channels=args.conv_channels,
            hidden_sizes=args.hidden_sizes,
            dataset=args.dataset,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pooling_kernel=2,
            pooling_stride=2,
            activation=args.activation,
            activation_out=args.activation_out,
            pooling="avg",
            steps=100,
            threshold=1,
            decay=0.99,
            pool_threshold=0.75,
            n_out=len(train_loader.dataset.targets.unique()),
            verbose=True,
        )

    elif args.model.lower() == "fcn_autoencoder":
        from models.fcn_autoencoder import FullyConnectedAutoencoder
        net = FullyConnectedAutoencoder(
            input_width=width,
            input_height=height,
            input_channels=channels,
            hidden_sizes=args.hidden_sizes,
            dataset=args.dataset,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            activation=args.activation,
            activation_out=args.activation_out,
            verbose=args.verbose,
        )

    elif args.model.lower() == "cnn_autoencoder":
        from models.cnn_autoencoder import ConvolutionalAutoencoder
        net = ConvolutionalAutoencoder(
            input_width=width,
            input_height=height,
            input_channels=channels,
            conv2d_channels=args.conv_channels,
            hidden_sizes=args.hidden_sizes,
            dataset=args.dataset,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pooling_kernel=1,
            pooling_stride=1,
            activation=args.activation,
            activation_out=args.activation_out,
            pooling=args.pooling,
            verbose=args.verbose,
        )

    elif args.model.lower() == "scnn_autoencoder":
        from models.scnn_autoencoder import SpikingConvolutionalAutoencoder
        net = SpikingConvolutionalAutoencoder(
            input_width=width,
            input_height=height,
            input_channels=channels,
            conv2d_channels=args.conv_channels,
            hidden_sizes=args.hidden_sizes,
            dataset=args.dataset,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pooling_kernel=1,     # not really supported due to lack of avgunpool
            pooling_stride=1,     # same
            pooling="avg",        # same
            pool_threshold=0.75,  # same
            activation=args.activation,
            activation_out=args.activation_out,
            steps=args.steps,
            threshold=args.threshold,
            decay=args.decay,
            verbose=args.verbose,
        )
        
    else:
        raise NotImplementedError(f"The model {args.model} is not implemented")

    # tell logger to watch model
    logger.watch(net.model)

    # run training and evaluation
    net.train_and_evaluate(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        epoch_batches=args.epoch_batches,
        load=args.load,
        model_name=args.model,
        metrics=args.metrics,
        key_metric=args.key_metric,
        goal=args.goal,
        eval_first=args.eval_first,
        logger=logger,
        checkpoints_dir=f"{logger.run.dir}/checkpoints",
        samplers=args.samplers,
        sample_freq=args.sample_freq,
    )
    

if __name__ == "__main__":
    main()
