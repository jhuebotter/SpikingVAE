import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
import utils as u


class FullyConnectedNeuralNetwork(BaseModel):
    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        hidden_sizes,
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
        log_func=print,
    ):
        super(FullyConnectedNeuralNetwork, self).__init__(
            input_width=input_width,
            input_height=input_height,
            input_channels=input_channels,
            dataset=dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            log_interval=log_interval,
            verbose=verbose,
            log_func=log_func,
        )

        # initialize model
        if verbose:
            self.log_func("Initializing fully connected neural network...")

        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        self.model = FCModel(
            input_width=self.input_width,
            input_height=self.input_height,
            input_channels=self.input_channels,
            hidden_sizes=self.hidden_sizes,
            activation=activation,
            n_out=self.n_out,
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
        hidden_sizes,
        activation="relu",
        n_out=2,
    ):
        super(FCModel, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.hidden_sizes = hidden_sizes
        self.fc_activations = [activation for i in range(len(hidden_sizes))] + ["softmax"]
        self.n_out = n_out

        # get flattend input size
        x = torch.randn(self.input_channels, self.input_width, self.input_height).view(
            -1, self.input_channels, self.input_width, self.input_height
        )
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        # define fully connected layers
        fc_args = dict(
            input_size=self._to_linear,
            hidden_sizes=self.hidden_sizes,
            output_size=self.n_out,
            activations=self.fc_activations
        )
        fc_parameters = u.get_fc_layers(**fc_args)
        self.fc_layers = u.build_layers(fc_parameters)

    def forward(self, x):
        """Passes data through the network.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        x = x.view(-1, self._to_linear)

        for fc_layer in self.fc_layers.values():
            x = fc_layer(x)

        return x