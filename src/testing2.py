from models.scnn_autoencoder_new import SpikingConvolutionalAutoencoderNew
import utils as u
from logger import WandBLogger
import torch
import losses

torch.autograd.set_detect_anomaly(True)

# set important parameters
parser = u.get_argparser()
args = parser.parse_args()
args.dataset = "cifar10gray"
args.batch_size = 40
args.test_batch_size = 40
args.hidden_sizes = [100]
args.conv_channels = [16, 32] #  [int(item) for item in args.conv_channels.split(',')]
args.epoch_batches = 100
args.epochs = 1#0
args.grad_clip = 50

# loss
args.loss = "custom"
args.wd = 0.0
args.beta1 = 0.0#1
args.beta2 = 0.01
args.lambd1 = 0.0#1
args.lambd2 = 0.0#1
args.l1 = 0.0
args.l2 = 0.0
args.example2 = 0.0
args.neuron2 = 0.0
args.neuron1 = 0.01

#args.delta_w = 1.0
#args.inactivity_threshold = 1
args.lr = 0.0005
args.steps = 100
args.decay = 0.99
args.noise = 0.0
args.adapt_threshold = False
#args.seed = 3
#args.reset = True
args.encoder = "spike"
args.eval_first = False
args.extra_grad = False
args.scale = 0.2
args.decoder = "max"
args.model = "scnn_autoencoder_new"
args.metrics = []
args.samplers = ["plot_filters",
                 "plot_activity_matrix",
                 "plot_output_spikes",
                 "plot_reconstruction",
                 "plot_output_potential",
                 "plot_cummulative_potential",
                 "plot_histograms"
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
    padding=0, #2 if args.kernel_size >= 5 else 1,
    #pooling_kernel=1,
    #pooling_stride=1,
    #activation="lif",
    #activation_out="lif",
    #pooling="avg",
    steps=args.steps,
    threshold=1,
    decay=args.decay,
    adapt_threshold=args.adapt_threshold,
    threshold_width=args.threshold_width,
    delta_threshold=args.delta_threshold,
    rho=args.rho,
    epsilon=args.epsilon,
    #pool_threshold=0.75,
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

# tell logger to watch model
logger.watch(net.model)

net.train_and_evaluate(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=args.epochs,
    model_name="spiking_test",
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
