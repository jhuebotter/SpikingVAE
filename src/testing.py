import torch.nn as nn
from models.cnn_autoencoder import ConvolutionalAutoencoder
from models.scnn_classifier import SpikingConvolutionalClassifier
import utils as u
from tqdm import tqdm


# set important parameters
parser = u.get_argparser()
args = parser.parse_args()
args.loss = "mse"
args.dataset = "mnist"
args.hidden_sizes = [200]
args.conv_channels = "20, 50"
args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
args.epoch_batches = 50
args.epochs = 10
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
    steps=100,
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
    key_metric="validation loss",
    goal="minimize",
    epoch_batches=args.epoch_batches,
)



"""

with tqdm(total=len(train_loader)) as t:
    for batchidx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        y = net.model(data)
        t.update()

    #t.set_postfix(loss="{:05.4f}".format(loss_avg()))

        #print(data[0])
        #print(y["output"][0])
        #print(labels[0])

        #break



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

children = [c for c in net.model.children()]

print("CHILDREN")

for c in children:
    print(c)
    print(type(c))
    print("\n")


print("CHILDREN LAYERS")

for c in children:
    for layer in c.modules():
        print(layer)
        print(type(layer))
        print("\n")


print("MODULES")

modules = [m for m in net.model.modules() if not issubclass(type(m), nn.modules.container.ModuleDict)]

for m in modules:
    print(m)
    print(type(m))
    print("\n")

"""