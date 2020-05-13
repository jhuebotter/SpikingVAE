import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.base_model import BaseModel
import src.utils as u


class ConvolutionalNeuralNetwork(BaseModel):
    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        dataset,
        loss,
        optimizer,
        learning_rate,
        weight_decay,
        device,
        log_interval,
        activation="relu",
        pooling="avg",
        n_out=2,
        verbose=True,
    ):
        super(ConvolutionalNeuralNetwork, self).__init__(
            input_width,
            input_height,
            input_channels,
            dataset,
            learning_rate,
            weight_decay,
            device,
            log_interval,
            activation,
            pooling,
            n_out,
            verbose,
        )

        if verbose:
            self.log_func("Initializing convolutional neural network...")

        self.model = CNNModel(
            self.input_width, self.input_height, self.input_channels, n_out=n_out,
        )
        self.model.to(self.device)

        if verbose:
            self.summary()
            self.log_func(f"Model initialized successfully!\n")

        self.optimizer = u.get_optimizer(
            optimizer, self.model, self.lr, self.wd, verbose
        )

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
        self.log_func(f"Flattend shape after convolution: {x_[0].shape}")
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
        activation="relu",
        pooling="avg",
        n_out=2,
        verbose=False,
    ):
        super(CNNModel, self).__init__()

        if verbose:
            self.log_func("Initializing CNN model...")

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.n_out = n_out

        # set activation function
        if activation.lower() == "relu":
            self.activation = F.relu
        else:
            raise NotImplementedError(
                f"The activation function {activation} is not implemented.\n"
                f"Valid options are: 'relu'."
            )

        # set pooling function
        if pooling.lower() == "max":
            self.pooling = nn.MaxPool2d
        elif pooling.lower() == "avg":
            self.pooling = nn.AvgPool2d
        else:
            raise NotImplementedError(
                f"The pooling function {pooling} is not implemented.\n"
                f"Valid options are: 'max', 'avg'."
            )

        # define layers
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pool1 = self.pooling(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pool2 = self.pooling(kernel_size=2, stride=1)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pool3 = self.pooling(kernel_size=2, stride=1)

        x = torch.randn(self.input_width, self.input_height).view(
            -1, 1, self.input_width, self.input_height
        )
        x_ = self.convs(x)
        self._to_linear = x_[0].shape[0] * x_[0].shape[1] * x_[0].shape[2]

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, self.n_out)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance1)
                # define threshold
                m.threshold = 1

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)
                # define threshold
                m.threshold = 1

    def convs(self, x):
        """Passes data through convolutional layers.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool3(x)

        return x

    def forward(self, x):
        """Passes data through the network.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)


if __name__ == "__main__":

    cnn = ConvolutionalNeuralNetwork(
        input_width=28,
        input_height=28,
        input_channels=1,
        dataset="mnist",
        learning_rate=0.001,
        device="cuda",
        log_interval=10,
        loss="crossentropy",
        optimizer="adam",
        weight_decay=0.001,
    )
