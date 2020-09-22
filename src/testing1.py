from models.scnn_autoencoder import SpikingConvolutionalAutoencoder
import utils as u
from logger import WandBLogger


# set important parameters
parser = u.get_argparser()
args = parser.parse_args()
args.loss = "mse"
args.dataset = "mnist"
args.batch_size = 32
#args.hidden_sizes = [200]
#args.conv_channels = "16, 32"
args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
args.epoch_batches = 100  #0 #10  # 00
args.epochs = 5
args.grad_clip = 5.0
args.wd = 0.001  # 5.0
args.loss = "custom"
#args.lr = 0.0005
#args.batch_size = 20
#args.steps = 100
#args.decay = 0.99
args.noise = 0.2
#args.seed = 3
args.encoder = "potential"
args.scale = 0.2
args.decoder = "max"
args.model = "scnn_autoencoder"
args.samplers = ["plot_filters",
                 # "plot_activity_matrix",
                 "plot_output_spikes",
                 "plot_reconstruction",
                 "plot_output_potential",
                 "plot_cummulative_potential"]

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

encoder_params = dict(encoder=args.encoder,
                      device=device,
                      noise=args.noise,
                      std=1.0,
                      scaling=args.scale,
                      leak=args.decay)

decoder_params = dict(decoder=args.decoder,
                      device=device,
                      scaling=args.steps * args.scale)

net = SpikingConvolutionalAutoencoder(
    input_width=width,
    input_height=height,
    input_channels=channels,
    conv2d_channels=args.conv_channels,
    hidden_sizes=args.hidden_sizes,
    loss=args.loss,
    optimizer=args.optimizer,
    learning_rate=args.lr,
    weight_decay=args.wd,
    device=device,
    kernel_size=args.kernel_size,
    stride=1,
    padding=0, #2 if args.kernel_size >= 5 else 1,
    pooling_kernel=1,
    pooling_stride=1,
    activation="lif",
    activation_out="lif",
    pooling="avg",
    steps=args.steps,
    threshold=1,
    decay=args.decay,
    pool_threshold=0.75,
    encoder_params=encoder_params,
    decoder_params=decoder_params,
    grad_clip=args.grad_clip,
    verbose=True,
    log_func=print,
)

# tell logger to watch model
logger.watch(net.model)

net.train_and_evaluate(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=args.epochs,
    model_name="spiking_test",
    metrics=[],
    key_metric="validation loss",
    goal=args.goal,
    max_epoch_batches=args.epoch_batches,
    samplers=args.samplers,
    sample_freq=10000,
    logger=logger,
)
