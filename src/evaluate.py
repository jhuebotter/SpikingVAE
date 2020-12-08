from models.scnn_autoencoder_new import SpikingConvolutionalAutoencoderNew
import utils as u
import losses
import metrics as met
import samplers as sam
from logger import WandBLogger

# set important parameters
parser = u.get_argparser()
args = parser.parse_args("")
args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
args.loss = "custom"

# set plotting options
args.metrics = ["correlation",
                "spikedensity",
                "meanactivity",
                "pctactive",
                "pctactiveperexample"]
args.samplers = ["plot_filters",
                 "plot_activity_matrix",
                 "plot_output_spikes",
                 "plot_reconstruction",
                 "plot_output_potential",
                 "plot_cummulative_potential",
                 "plot_histograms"
                 ]

if not args.load:
    args.load = "/media/justus/Data/OneDrive/Projects/SpikingVAE/results/checkpoints/SAE_sparse_latest.pth"


# make experiments reproducible
if args.seed:
    u.set_seed(args.seed)

# choose the devices for computation (GPU if available)
device = u.get_backend(args)

# initialize metrics
metrics = met.get_metrics(args.metrics)

# initialize samplers
samplers = sam.get_samplers(args.samplers)

# initialize logger
logger = WandBLogger(
    args=args,
    name=args.model,
)

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

decoder_params = dict(decoder=args.decoder,
                      device=device,
                      scaling=args.steps * args.scale)

loss_fn = losses.get_loss_function(loss=args.loss,
                                   verbose=args.verbose,
                                   spiking=True,
                                   params=dict(beta1=args.beta1,
                                               beta2=args.beta2,
                                               lambd1=args.lambd1,
                                               lambd2=args.lambd2,
                                               l1=args.l1,
                                               l2=args.l2,
                                               example2=args.example2,
                                               neuron2=args.neuron2,
                                               neuron1=args.neuron1,
                                               layers=(len(args.conv_channels) + len(args.hidden_sizes)) * 2
                                               )
                                   )

# init network
net = SpikingConvolutionalAutoencoderNew(
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
    steps=args.steps,
    threshold=1,
    decay=args.decay,
    adapt_threshold=args.adapt_threshold,
    threshold_width=args.threshold_width,
    delta_threshold=args.delta_threshold,
    rho=args.rho,
    epsilon=args.epsilon,
    inactivity_threshold=args.inactivity_threshold,
    delta_w=args.delta_w,
    encoder_params=encoder_params,
    decoder_params=decoder_params,
    grad_clip=args.grad_clip,
    use_extra_grad=args.extra_grad,
    verbose=True,
    log_func=print,
    reset=args.reset,
)

# load model data from disk
checkpoint = net.load_checkpoint(args.load, net.model, net.optimizer)
epoch = checkpoint["epoch"]

# validate model on dataset
val_results, samples = net.evaluate(val_loader, metrics, samplers, args.sample_freq)

# Store results in log
summary = dict()
for key, value in val_results.items():
    summary[f"validation {key}"] = value
logger.save_summary(summary, epoch=epoch)
logger.log(summary, step=epoch)
for batch, sample in samples.items():
    for key, value in sample.items():
        logger.log({f"{batch}_{key}": value}, step=epoch)