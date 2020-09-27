from models.cnn_autoencoder import ConvolutionalAutoencoder
import utils as u
from logger import WandBLogger
import losses

# set important parameters
parser = u.get_argparser()
args = parser.parse_args()
args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
args.loss = "mse"
args.encoder = "noisy"
args.scale = 1.0
args.wd = 0.0001
args.epochs = 10
args.noise = 0.8
args.model = "cnn_autoencoder"
args.metrics = []
args.samplers = ["plot_filters",
                 "plot_reconstruction",
                 ]

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
    test_batch_size=args.test_batch_size,
    cuda=args.cuda,
    verbose=args.verbose
)

encoder_params = dict(encoder=args.encoder,
                      device=device,
                      noise=args.noise,
                      std=1.0,
                      scaling=args.scale,
                      leak=args.decay)

loss_fn = losses.get_loss_function(loss=args.loss,
                                   verbose=args.verbose,
                                   spiking=False,
                                   params=dict()
                                   )

net = ConvolutionalAutoencoder(
    input_width=width,
    input_height=height,
    input_channels=channels,
    conv2d_channels=args.conv_channels,
    hidden_sizes=args.hidden_sizes,
    loss=loss_fn,
    optimizer=args.optimizer,
    learning_rate=args.lr,
    weight_decay=args.wd,
    device=device,
    kernel_size=args.kernel_size,
    stride=1,
    padding=0,
    pooling_kernel=1,
    pooling_stride=1,
    activation="relu",
    activation_out="relu",
    encoder_params=encoder_params,
    verbose=True,
    log_func=print,
)

# tell logger to watch model
logger.watch(net.model)

net.train_and_evaluate(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=args.epochs,
    model_name="cnn_experiment",
    metrics=args.metrics,
    key_metric="validation loss",
    goal=args.goal,
    max_epoch_batches=args.epoch_batches,
    samplers=args.samplers,
    sample_freq=10000,
    logger=logger,
    checkpoints_dir=f"{logger.run.dir}/checkpoints",
    eval_first=args.eval_first,
)
