import torch
import torch.nn as nn
from models.base_model import BaseModel
import utils as u


class FullyConnectedAutoencoder(BaseModel):
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
        activation="relu",
        activation_out="logsoftmax",
        verbose=True,
        log_func=print,
    ):
        super(FullyConnectedAutoencoder, self).__init__(
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
            self.log_func("Initializing fully connected autoencoder...")

        self.task = "reconstruction"
        self.hidden_sizes = hidden_sizes
        self.model = FCAutoencoderModel(
            input_width=self.input_width,
            input_height=self.input_height,
            input_channels=self.input_channels,
            hidden_sizes=self.hidden_sizes,
            activation=activation,
            activation_out=activation_out,
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
        self.log_func(f"Output shape: {x.shape}")
        self.log_func(f"Network architecture:")
        self.log_func(self.model)
        self.log_func(f"Number of trainable parameters: {self.count_parameters()}")


class FCAutoencoderModel(nn.Module):
    """Creates a Fully connected autoencoder model."""

    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        hidden_sizes,
        activation="relu",
        activation_out="relu",
    ):
        super(FCAutoencoderModel, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.hidden_sizes = hidden_sizes
        self.fc_encoder_activations = [activation for i in range(len(hidden_sizes))]
        self.fc_decoder_activations = [activation for i in range(len(hidden_sizes)-1)] + [activation_out]

        # get flattend input size
        x = torch.randn(self.input_channels, self.input_width, self.input_height).view(
            -1, self.input_channels, self.input_width, self.input_height
        )
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        # define fully connected layers
        fc_encoder_args = dict(
            input_size=self._to_linear,
            hidden_sizes=self.hidden_sizes[:-1],
            output_size=self.hidden_sizes[-1],
            activations=self.fc_encoder_activations
        )
        fc_encoder_parameters = u.get_fc_layers(**fc_encoder_args)
        self.fc_encoder_layers = u.build_layers(fc_encoder_parameters)

        decoder_sizes = self.hidden_sizes[:-1]
        decoder_sizes.reverse()

        fc_decoder_args = dict(
            input_size=self.hidden_sizes[-1],
            hidden_sizes=decoder_sizes,
            output_size=self._to_linear,
            activations=self.fc_decoder_activations
        )

        fc_decoder_parameters = u.get_fc_layers(**fc_decoder_args)
        self.fc_decoder_layers = u.build_layers(fc_decoder_parameters)

    def encode(self, x):

        x = x.view(-1, self._to_linear)

        for layer in self.fc_encoder_layers.values():
            x = layer(x)

        return x

    def decode(self, x):

        for layer in self.fc_decoder_layers.values():
            x = layer(x)

        x = x.view(-1, self.input_channels, self.input_width, self.input_height)

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
