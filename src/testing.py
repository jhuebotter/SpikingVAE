from models.scnn_classifier import SpikingConvolutionalClassifier
import utils as u

# set important parameters
parser = u.get_argparser()
args = parser.parse_args()
args.loss = "mse"
args.dataset = "mnist"
args.hidden_sizes = [100]
args.conv_channels = "16, 32"
args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
args.epoch_batches = 100
args.epochs = 5
args.lr = 0.005
args.batch_size = 20
args.steps = 20
args.seed = 3
args.samplers = ["plot_output_spikes", "plot_output_potential", "plot_cummulative_potential"]
# choose the devices for computation (GPU if available)
device = u.get_backend(args)

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
    kernel_size=5,
    stride=1,
    padding=2,
    pooling_kernel=2,
    pooling_stride=2,
    activation="lif",
    activation_out="lif",
    pooling="avg",
    steps=args.steps,
    threshold=1,
    decay=0.99,
    pool_threshold=0.75,
    n_out=len(train_loader.dataset.targets.unique()),
    verbose=True,
    log_func=print,
)

net.train_and_evaluate(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=args.epochs,
    model_name="spiking_test",
    metrics=["accuracy"],
    key_metric="validation accuracy",
    goal="maximize",
    epoch_batches=args.epoch_batches,
    samplers=args.samplers,
    sample_freq=1500,
)
