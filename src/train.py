from models.cnn_classifier import ConvolutionalNeuralNetwork
from models.fcn_classifier import FullyConnectedNeuralNetwork
from models.fcn_autoencoder import FullyConnectedAutoencoder
from models.cnn_autoencoder import ConvolutionalAutoencoder
from logger import WandBLogger
import utils as u

def main():

    # set important parameters
    parser = u.get_argparser()
    args = parser.parse_args()

    #args.model = "cnn_autoencoder"
    args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
    args.metrics = []
    #args.hidden_sizes = [500]
    #args.activation_out = "relu"
    #args.loss = "mse"
    #args.dataset = "mnist"
    #args.eval_first = True

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
    
    if args.model.lower() == "fcn_classifier":
        net = FullyConnectedNeuralNetwork(
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
        net = ConvolutionalNeuralNetwork(
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
            kernel_size=3,
            stride=1,
            pooling_kernel=2,
            pooling_stride=1,
            activation=args.activation,
            activation_out=args.activation_out,
            pooling=args.pooling,
            n_out=len(train_loader.dataset.targets.unique()),
            verbose=args.verbose,
        )

    elif args.model.lower() == "fcn_autoencoder":
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
            kernel_size=3,
            stride=1,
            pooling_kernel=1,
            pooling_stride=1,
            activation=args.activation,
            activation_out=args.activation_out,
            pooling=args.pooling,
            verbose=args.verbose,
        )
        
    else:
        raise NotImplementedError(f"The model {args.model} is not implemented")

    # tell logger to watch model
    logger.watch(net.model)

    # run training
    net.train_and_evaluate(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
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
