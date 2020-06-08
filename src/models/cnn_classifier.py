import torch
import torch.nn as nn
from models.base_model import BaseModel
import utils as u


class ConvolutionalNeuralNetwork(BaseModel):
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
        pooling_kernel=2,
        pooling_stride=1,
        activation="relu",
        activation_out="logsoftmax",
        pooling="avg",
        n_out=2,
        verbose=True,
        log_func=print,
    ):
        super(ConvolutionalNeuralNetwork, self).__init__(
            input_width=input_width,
            input_height=input_height,
            input_channels=input_channels,
            dataset=dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
            log_func=log_func,
        )

        # initialize model
        if verbose:
            self.log_func("Initializing convolutional neural network...")

        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        self.conv2d_channels = conv2d_channels
        self.kernel_sizes = [kernel_size for i in range(len(conv2d_channels))]
        self.strides = [stride for i in range(len(conv2d_channels))]
        self.pooling_kernels = [pooling_kernel for i in range(len(conv2d_channels))]
        self.pooling_strides = [pooling_stride for i in range(len(conv2d_channels))]
        self.model = CNNModel(
            input_width=self.input_width,
            input_height=self.input_height,
            input_channels=self.input_channels,
            conv2d_channels=self.conv2d_channels,
            hidden_sizes=self.hidden_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            pooling_kernels=self.pooling_kernels,
            pooling_strides=self.pooling_strides,
            activation=activation,
            activation_out=activation_out,
            pooling=pooling,
            n_out=n_out,
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

        # initialize loss function
        self.loss_function = u.get_loss_function(loss, verbose)

    def summary(self):
        """Prints a model summary about itself."""

        x = (
            torch.randn(self.input_channels, self.input_width, self.input_height)
            .view(-1, self.input_channels, self.input_width, self.input_height)
            .to(self.device)
        )

        x_ = self.model.convs(x)

        self.log_func(f"Input shape: {x.shape}")
        self.log_func(f"Shape after convolution: {x_[0].shape}")
        self.log_func(f"Output shape: {self.n_out}")
        self.log_func(f"Network architecture:")
        self.log_func(self.model)
        self.log_func(f"Number of trainable parameters: {self.count_parameters()}")


class CNNModel(nn.Module):
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
        pooling_kernels,
        pooling_strides,
        activation="relu",
        activation_out="logsoftmax",
        pooling="avg",
        n_out=2,
    ):
        super(CNNModel, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.conv2d_channels = conv2d_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pooling_kernels = pooling_kernels
        self.pooling_strides = pooling_strides
        self.hidden_sizes = hidden_sizes
        self.fc_activations = [activation for i in range(len(hidden_sizes))] + [activation_out]
        self.n_out = n_out

        # define convolutional layers
        conv_args = dict(
            input_channels=self.input_channels,
            channels=self.conv2d_channels,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            activations=[activation for i in range(len(self.conv2d_channels))],
            pooling_funcs=[pooling for i in range(len(self.conv2d_channels))],
            pooling_kernels=self.pooling_kernels,
            pooling_strides=self.pooling_strides,
        )
        conv_parameters = u.get_conv2d_layers(**conv_args)
        self.conv_layers = u.build_layers(conv_parameters)

        # get flattend input size
        x = torch.randn(self.input_channels, self.input_width, self.input_height).view(
            -1, self.input_channels, self.input_width, self.input_height
        )
        x_ = self.convs(x)
        self._to_linear = x_[0].shape[0] * x_[0].shape[1] * x_[0].shape[2]

        # define fully connected layers
        fc_args = dict(
            input_size=self._to_linear,
            hidden_sizes=self.hidden_sizes,
            output_size=self.n_out,
            activations=self.fc_activations
        )
        fc_parameters = u.get_fc_layers(**fc_args)
        self.fc_layers = u.build_layers(fc_parameters)

    def convs(self, x):
        """Passes data through convolutional layers.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        for layer in self.conv_layers.values():
            x = layer(x)

        return x

    def forward(self, x):
        """Passes data through the network.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        x = self.convs(x)

        x = x.view(-1, self._to_linear)

        for fc_layer in self.fc_layers.values():
            x = fc_layer(x)

        result = dict(output=x)

        return result


"""

        conv_parameters = [
            dict(
                name="conv1",
                type="conv2d",
                parameters=dict(
                    in_channels=self.input_channels,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ),
            dict(
                name="activation1",
                type=f"relu",
                parameters=dict(),
            ),
            dict(
                name="pool1",
                type=f"{pooling}pool2d",
                parameters=dict(kernel_size=2, stride=1, ),
            ),
            dict(
                name="conv2",
                type="conv2d",
                parameters=dict(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ),
            dict(
                name="activation2",
                type=f"relu",
                parameters=dict(),
            ),
            dict(
                name="pool2",
                type=f"{pooling}pool2d",
                parameters=dict(kernel_size=2, stride=1, ),
            ),
            dict(
                name="conv3",
                type="conv2d",
                parameters=dict(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ),
            dict(
                name="activation3",
                type=f"relu",
                parameters=dict(),
            ),
            dict(
                name="pool3",
                type=f"{pooling}pool2d",
                parameters=dict(kernel_size=2, stride=1, ),
            ),
        ]


"""
