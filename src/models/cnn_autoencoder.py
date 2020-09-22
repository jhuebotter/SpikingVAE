import torch
import torch.nn as nn
from models.base_model import BaseModel
import utils as u


class ConvolutionalAutoencoder(BaseModel):
    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        conv2d_channels,
        hidden_sizes,
        dataset,
        loss,
        optimizer,
        learning_rate,
        weight_decay,
        device,
        kernel_size=3,
        stride=1,
        padding=0,
        pooling_kernel=2,
        pooling_stride=1,
        activation="relu",
        activation_out="logsoftmax",
        pooling="avg",
        verbose=True,
        log_func=print,
    ):
        super(ConvolutionalAutoencoder, self).__init__(
            input_width=input_width,
            input_height=input_height,
            input_channels=input_channels,
            dataset=dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            loss=loss,
            verbose=verbose,
            log_func=log_func,
        )

        # initialize model
        if verbose:
            self.log_func("Initializing convolutional autoencoder...")

        self.task = "reconstruction"
        self.hidden_sizes = hidden_sizes
        self.conv2d_channels = conv2d_channels
        self.kernel_sizes = [kernel_size for i in range(len(conv2d_channels))]
        self.strides = [stride for i in range(len(conv2d_channels))]
        self.paddings = [padding for i in range(len(conv2d_channels))]
        self.pooling_kernels = [pooling_kernel for i in range(len(conv2d_channels))]
        self.pooling_strides = [pooling_stride for i in range(len(conv2d_channels))]
        self.model = CNNAutoencoderModel(
            input_width=self.input_width,
            input_height=self.input_height,
            input_channels=self.input_channels,
            conv2d_channels=self.conv2d_channels,
            hidden_sizes=self.hidden_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            pooling_kernels=self.pooling_kernels,
            pooling_strides=self.pooling_strides,
            activation=activation,
            activation_out=activation_out,
            pooling=pooling,
        )
        self.init_weights()
        self.model.to(self.device)

        if verbose:
            self.summary()
            self.log_func(f"Model initialized successfully!\n")

        # initialize optimizer
        self.optimizer = u.get_optimizer(
            optimizer, self.model, self.lr, self.wd, verbose
        )

    def summary(self):
        """Prints a model summary about itself."""

        x = (
            torch.randn(self.input_channels, self.input_width, self.input_height)
            .view(-1, self.input_channels, self.input_width, self.input_height)
            .to(self.device)
        )

        x_ = self.model.conv_encode(x)

        self.log_func(f"Input shape: {x.shape}")
        self.log_func(f"Shape after convolution: {x_[0].shape}")
        self.log_func(f"Network architecture:")
        self.log_func(self.model)
        self.log_func(f"Number of trainable parameters: {self.count_parameters()}")


class CNNAutoencoderModel(nn.Module):
    """Creates a CNN model."""

    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        conv2d_channels,
        hidden_sizes,
        kernel_sizes,
        strides,
        paddings,
        pooling_kernels,
        pooling_strides,
        activation="relu",
        activation_out="logsoftmax",
        pooling="avg",
    ):
        super(CNNAutoencoderModel, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.conv2d_channels = conv2d_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.pooling_kernels = pooling_kernels
        self.pooling_strides = pooling_strides
        self.hidden_sizes = hidden_sizes
        # self.fc_activations = [activation for i in range(len(hidden_sizes))] + [activation_out]

        # define convolutional encoder layers
        conv_encoder_args = dict(
            input_channels=self.input_channels,
            channels=self.conv2d_channels,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            activations=[activation for i in range(len(self.conv2d_channels))],
            pooling_funcs=[pooling for i in range(len(self.conv2d_channels))],
            pooling_kernels=self.pooling_kernels,
            pooling_strides=self.pooling_strides,
        )
        conv_encoder_parameters = u.get_conv2d_layers(**conv_encoder_args)
        self.conv_encoder_layers = u.build_layers(conv_encoder_parameters)

        # get flattend input size and reverse transformation
        x = torch.randn(self.input_channels, self.input_width, self.input_height).view(
            -1, self.input_channels, self.input_width, self.input_height
        )
        x_ = self.conv_encode(x)
        self._to_linear = x_[0].shape[0] * x_[0].shape[1] * x_[0].shape[2]
        self._from_linear = (x_[0].shape[0], x_[0].shape[1], x_[0].shape[2])

        # define fully connected encoder layers
        self.fc_encoder_activations = [activation for i in range(len(hidden_sizes))]

        fc_encoder_args = dict(
            input_size=self._to_linear,
            hidden_sizes=self.hidden_sizes[:-1],
            output_size=self.hidden_sizes[-1],
            activations=self.fc_encoder_activations,
        )
        fc_encoder_parameters = u.get_fc_layers(**fc_encoder_args)
        self.fc_encoder_layers = u.build_layers(fc_encoder_parameters)

        # define fully connected decoder layers
        self.fc_decoder_activations = [activation for i in range(len(hidden_sizes))]
        decoder_sizes = self.hidden_sizes[:-1]
        decoder_sizes.reverse()

        fc_decoder_args = dict(
            input_size=self.hidden_sizes[-1],
            hidden_sizes=decoder_sizes,
            output_size=self._to_linear,
            activations=self.fc_decoder_activations,
        )

        fc_decoder_parameters = u.get_fc_layers(**fc_decoder_args)
        self.fc_decoder_layers = u.build_layers(fc_decoder_parameters)

        # define convolutional decoder layers
        decoder_kernel_sizes = self.kernel_sizes
        decoder_kernel_sizes.reverse()
        decoder_convtranspose2d_channels = [self.input_channels] + self.conv2d_channels[:-1]
        decoder_convtranspose2d_channels.reverse()
        decoder_strides = self.strides
        decoder_strides.reverse()
        decoder_paddings = self.paddings
        decoder_paddings.reverse()
        unpooling_kernels = self.pooling_kernels
        unpooling_kernels.reverse()
        unpooling_strides = self.pooling_strides
        unpooling_strides.reverse()

        conv_decoder_args = dict(
            input_channels=x_[0].shape[0],
            channels=decoder_convtranspose2d_channels,
            kernel_sizes=decoder_kernel_sizes,
            strides=decoder_strides,
            paddings=decoder_paddings,
            activations=[activation for i in range(len(self.conv2d_channels) - 1)] + [activation_out],
            unpooling_funcs=[pooling for i in range(len(self.conv2d_channels))],
            unpooling_kernels=unpooling_kernels,
            unpooling_strides=unpooling_strides,
        )
        conv_decoder_parameters = u.get_convtranspose2d_layers(**conv_decoder_args)
        self.conv_decoder_layers = u.build_layers(conv_decoder_parameters)

    def conv_encode(self, x):
        """Passes data through convolutional layers.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        for layer in self.conv_encoder_layers.values():
            x = layer(x)

        return x

    def fc_encode(self, x):

        x = x.view(-1, self._to_linear)

        for layer in self.fc_encoder_layers.values():
            x = layer(x)

        return x

    def fc_decode(self, x):

        for layer in self.fc_decoder_layers.values():
            x = layer(x)

        return x

    def conv_decode(self, x):

        x = x.view(-1, *self._from_linear)

        for layer in self.conv_decoder_layers.values():
            x = layer(x)

        return x

    def encode(self, x):

        x = self.conv_encode(x)
        x = self.fc_encode(x)

        return x

    def decode(self, x):

        x = self.fc_decode(x)
        x = self.conv_decode(x)

        return x

    def forward(self, x):
        """Passes data through the network.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        z = self.encode(x)
        y = self.decode(z)

        result = dict(output=y, latent=z)

        return result
