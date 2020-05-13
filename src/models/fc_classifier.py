import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
import src.utils as u


class FullyConnectedNeuralNetwork(BaseModel):
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
        print_freq,
        activation="relu",
        n_out=2,
        verbose=True,
    ):
        super(FullyConnectedNeuralNetwork, self).__init__(
            input_width,
            input_height,
            input_channels,
            dataset,
            learning_rate,
            weight_decay,
            device,
            log_interval,
            activation,
            n_out,
            verbose,
        )

        if verbose:
            self.log_func("Initializing fully connected neural network...")

        self.model = FCModel(
            self.input_width, self.input_height, self.input_channels, n_out=n_out,
        )
        self.init_weights()
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

        self.log_func(f"Input shape: {x.shape}")
        self.log_func(f"Output shape: {self.n_out}")
        self.log_func(f"Network architecture:")
        self.log_func(self.model)
        self.log_func(f"Number of trainable parameters: {self.count_parameters()}")


class FCModel(nn.Module):
    """Creates a CNN model."""

    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        activation="relu",
        n_out=2,
        verbose=False,
    ):
        super(FCModel, self).__init__()

        if verbose:
            self.log_func("Initializing FC model...")

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

        # define layers
        x = torch.randn(self.input_width, self.input_height).view(
            -1, 1, self.input_width, self.input_height
        )
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.n_out)


    def forward(self, x):
        """Passes data through the network.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        x = x.view(-1, self._to_linear)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)


if __name__ == "__main__":

    fcn = FullyConnectedNeuralNetwork(
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
