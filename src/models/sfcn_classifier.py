import torch
import torch.nn as nn
from models.base_model import SpikingBaseModel
from models.spiking_layers import LIF_sNeuron, LF_Unit
import utils as u


class SpikingFCNClassifier(SpikingBaseModel):
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
        activation="lif",
        activation_out="logsoftmax",
        steps=100,
        threshold=1,
        decay=0.99,
        n_out=2,
        verbose=True,
        log_func=print,
    ):
        super(SpikingFCNClassifier, self).__init__(
            input_width=input_width,
            input_height=input_height,
            input_channels=input_channels,
            dataset=dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            steps=steps,
            threshold=threshold,
            decay=decay,
            device=device,
            loss=loss,
            verbose=verbose,
            log_func=log_func,
        )

        # initialize model
        if verbose:
            self.log_func("Initializing spiking convolutional neural network...")

        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        self.model = SFCNModel(
            input_width=self.input_width,
            input_height=self.input_height,
            input_channels=self.input_channels,
            hidden_sizes=self.hidden_sizes,
            activation=activation,
            activation_out=activation_out,
            threshold=threshold,
            decay=decay,
            n_out=n_out,
            device=device,
            steps=steps,
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

        # TODO: implement a better solution an delete line as done by base model

        # initialize loss function
        self.loss_function = u.get_loss_function(loss, verbose, spiking=True)

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


class SFCNModel(nn.Module):
    """Creates a CNN model."""

    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        hidden_sizes,
        activation="lif",
        activation_out="lif",
        steps=100,
        threshold=1,
        decay=0.99,
        n_out=2,
        device="cuda",
    ):
        super(SFCNModel, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.hidden_sizes = hidden_sizes
        self.fc_activations = [activation for i in range(len(hidden_sizes))] + [activation_out]
        self.n_out = n_out
        self.device = device
        self.steps = steps

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
            activations=self.fc_activations,
            thresholds=[threshold for i in range(len(self.fc_activations))],
            decays=[decay for i in range(len(self.fc_activations))],
        )
        fc_parameters = u.get_sfc_layers(**fc_args)
        self.fc_layers = u.build_layers(fc_parameters)


    def forward(self, x, steps=100):
        """Passes data through the network.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """
        modules = [m for m in self.modules() if not issubclass(type(m), nn.modules.container.ModuleDict)]

        out_temps = [[] for m in modules if type(m) == LIF_sNeuron]

        membrane_potentials = []
        total_outs = []
        LF_outs = []

        with torch.no_grad():
            p = x.clone()
            p = p.view(p.size(0), -1)

            for i, (name, layer) in enumerate(self.fc_layers.items()):
                if type(layer) == LIF_sNeuron:
                    LF_outs.append(torch.zeros(p.size(), requires_grad=False, device=self.device))
                    total_outs.append(torch.zeros(p.size(), requires_grad=False, device=self.device))
                if type(layer) not in [LIF_sNeuron, Pooling_sNeuron]:
                    p = layer(p)
                    last = i == len(self.fc_layers.items())-1
                    membrane_potentials.append(torch.zeros(p.size(), requires_grad=last, device=self.device))

        for t in range(self.steps):
            rand_num = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3)).to(self.device)
            Poisson_d_input = ((torch.abs(x)/2) > rand_num).type(torch.cuda.FloatTensor)
            out = torch.mul(Poisson_d_input, torch.sign(x))
            out = out.view(out.size(0), -1)

            i = 0

            for name, layer in self.fc_layers.items():
                if type(layer) == LIF_sNeuron:
                    out, membrane_potentials[i] = layer(membrane_potentials[i])
                    LF_outs[i], total_outs[i], out = LF_Unit(layer.decay, LF_outs[i], total_outs[i], out, out_temps[i], t)
                    i += 1
                else:
                    membrane_potentials[i] = membrane_potentials[i] + layer(out)

        result = dict(output=membrane_potentials[-1], total_outs=total_outs, lf_outs=LF_outs, out_temps=out_temps)

        return result

