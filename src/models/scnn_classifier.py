import torch
import torch.nn as nn
from models.base_model import SpikingBaseModel
from models.spiking_layers import LIF_sNeuron, Pooling_sNeuron, LF_Unit
import utils as u

import math
import torch.nn.functional as F


class SpikingConvolutionalClassifier(SpikingBaseModel):
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
        padding=1,
        pooling_kernel=2,
        pooling_stride=1,
        activation="lif",
        activation_out="lif",
        pooling="avg",
        steps=100,
        threshold=1,
        decay=0.99,
        pool_threshold=0.75,
        n_out=2,
        verbose=True,
        log_func=print,
    ):
        super(SpikingConvolutionalClassifier, self).__init__(
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
        self.conv2d_channels = conv2d_channels
        self.kernel_sizes = [kernel_size for i in range(len(conv2d_channels))]
        self.strides = [stride for i in range(len(conv2d_channels))]
        self.paddings = [padding for i in range(len(conv2d_channels))]
        self.pooling_kernels = [pooling_kernel for i in range(len(conv2d_channels))]
        self.pooling_strides = [pooling_stride for i in range(len(conv2d_channels))]
        self.model = SCNNModel(
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
            threshold=threshold,
            decay=decay,
            pool_threshold=pool_threshold,
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

        x_ = self.model.convs(x)

        self.log_func(f"Input shape: {x.shape}")
        self.log_func(f"Shape after convolution: {x_[0].shape}")
        self.log_func(f"Output shape: {self.n_out}")
        self.log_func(f"Network architecture:")
        self.log_func(self.model)
        self.log_func(f"Number of trainable parameters: {self.count_parameters()}")


class SCNNModel(nn.Module):
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
        activation="lif",
        activation_out="lif",
        pooling="avg",
        steps=100,
        threshold=1,
        decay=0.99,
        pool_threshold=0.75,
        n_out=2,
        device="cuda",
    ):
        super(SCNNModel, self).__init__()

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
        self.fc_activations = [activation for i in range(len(hidden_sizes))] + [
            activation_out
        ]
        self.n_out = n_out
        self.device = device
        self.steps = steps

        # define convolutional layers
        conv_args = dict(
            input_channels=self.input_channels,
            channels=self.conv2d_channels,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            activations=[activation for i in range(len(self.conv2d_channels))],
            pooling_funcs=[pooling for i in range(len(self.conv2d_channels))],
            pooling_kernels=self.pooling_kernels,
            pooling_strides=self.pooling_strides,
            thresholds=[threshold for i in range(len(self.conv2d_channels))],
            decays=[decay for i in range(len(self.conv2d_channels))],
            pool_thresholds=[pool_threshold for i in range(len(self.conv2d_channels))],
        )
        sconv_parameters = u.get_sconv2d_layers(**conv_args)
        self.conv_layers = u.build_layers(sconv_parameters)

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
            activations=self.fc_activations,
            thresholds=[threshold for i in range(len(self.fc_activations))],
            decays=[decay for i in range(len(self.fc_activations))],
        )

        fc_parameters = u.get_sfc_layers(**fc_args)
        self.fc_layers = u.build_layers(fc_parameters)

    def convs(self, x):
        """Passes data through convolutional layers.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        conv_modules = [
            c for c in self.conv_layers.values() if type(c) in [nn.Conv2d, nn.AvgPool2d]
        ]

        for layer in conv_modules:
            x = layer(x)

        return x

    def forward(self, x, steps=100):
        """Passes data through the network.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """
        modules = [
            m
            for m in self.modules()
            if not issubclass(type(m), nn.modules.container.ModuleDict)
        ]

        out_temps = [[] for m in modules if type(m) == LIF_sNeuron]
        potential_history = [
            [] for m in modules if type(m) in [LIF_sNeuron, Pooling_sNeuron]
        ]
        cum_potential_history = [
            [] for m in modules if type(m) in [LIF_sNeuron, Pooling_sNeuron]
        ]
        input_history = []
        membrane_potentials = []
        leaky_cum_membrane_potentials = []
        total_outs = []
        LF_outs = []

        with torch.no_grad():
            p = x.clone()
            # print(p.size())
            for name, layer in self.conv_layers.items():
                # print(f"layer {name}: {layer}\n input shape: {p.size()}")
                if type(layer) == LIF_sNeuron:
                    LF_outs.append(
                        torch.zeros(p.size(), requires_grad=False, device=self.device)
                    )
                    total_outs.append(
                        torch.zeros(p.size(), requires_grad=False, device=self.device)
                    )
                if type(layer) not in [LIF_sNeuron, Pooling_sNeuron]:
                    p = layer(p)
                    membrane_potentials.append(
                        torch.zeros(p.size(), requires_grad=False, device=self.device)
                    )
                    leaky_cum_membrane_potentials.append(
                        torch.zeros(p.size(), requires_grad=False, device=self.device)
                    )
                # print("output shape:", p.size())
                # print()
            p = p.view(p.size(0), -1)
            for i, (name, layer) in enumerate(self.fc_layers.items()):
                # print(f"layer {name}: {layer}\n input shape: {p.size()}")
                if type(layer) == LIF_sNeuron:
                    LF_outs.append(
                        torch.zeros(p.size(), requires_grad=False, device=self.device)
                    )
                    total_outs.append(
                        torch.zeros(p.size(), requires_grad=False, device=self.device)
                    )
                if type(layer) not in [LIF_sNeuron, Pooling_sNeuron]:
                    p = layer(p)
                    last = i == len(self.fc_layers.items()) - 1
                    membrane_potentials.append(
                        torch.zeros(p.size(), requires_grad=last, device=self.device)
                    )
                    leaky_cum_membrane_potentials.append(
                        torch.zeros(p.size(), requires_grad=last, device=self.device)
                    )

                # print("output shape:", p.size())
                # print()

        for t in range(self.steps):
            # print(t)

            rand_num = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3)).to(
                self.device
            )
            Poisson_d_input = ((torch.abs(x) / 2) > rand_num).type(
                torch.cuda.FloatTensor
            )
            out = torch.mul(Poisson_d_input, torch.sign(x))
            input_history.append(out.detach())
            i, j = 0, 0
            for name, layer in self.conv_layers.items():
                # print(f"layer {name}: {layer}\n"
                #      f"input size: {out.size()}")
                if type(layer) == LIF_sNeuron:
                    # print("spiking threshold:", layer.threshold)
                    # print("membrane potentials size:", membrane_potentials[i].size())
                    out, membrane_potentials[i] = layer(membrane_potentials[i])
                    LF_outs[j], total_outs[j], out = LF_Unit(
                        layer.decay, LF_outs[j], total_outs[j], out, out_temps[j], t
                    )
                    i += 1
                    j += 1
                elif type(layer) == Pooling_sNeuron:
                    # print("spiking threshold:", layer.threshold)
                    # print("membrane potentials size:", membrane_potentials[i].size())
                    out, membrane_potentials[i] = layer(membrane_potentials[i])
                    i += 1
                else:
                    # print("")
                    current = layer(out)
                    membrane_potentials[i] = membrane_potentials[i] + current
                    potential_history[i].append(membrane_potentials[i].detach())

                    leaky_cum_membrane_potentials[i] = (
                        leaky_cum_membrane_potentials[i] + current
                    )
                    leaky_cum_membrane_potentials[i] = (
                        0.99 * leaky_cum_membrane_potentials[i].detach()
                        + leaky_cum_membrane_potentials[i]
                        - leaky_cum_membrane_potentials[i].detach()
                    )

                    cum_potential_history[i].append(leaky_cum_membrane_potentials[i].detach())
                # print("output size:", out.size())
                # print()

            out = out.view(out.size(0), -1)

            for name, layer in self.fc_layers.items():
                # print(f"layer {name}: {layer}\n input size: {out.size()}")
                if type(layer) == LIF_sNeuron:
                    # print("spiking threshold:", layer.threshold)
                    # print("membrane potentials size:", membrane_potentials[i].size())
                    out, membrane_potentials[i] = layer(membrane_potentials[i])
                    LF_outs[j], total_outs[j], out = LF_Unit(
                        layer.decay, LF_outs[j], total_outs[j], out, out_temps[j], t
                    )
                    i += 1
                    j += 1
                else:
                    current = layer(out)
                    membrane_potentials[i] = membrane_potentials[i] + current
                    potential_history[i].append(membrane_potentials[i].detach())

                    leaky_cum_membrane_potentials[i] = (
                        leaky_cum_membrane_potentials[i] + current
                    )
                    leaky_cum_membrane_potentials[i] = (
                        0.99 * leaky_cum_membrane_potentials[i].detach()
                        + leaky_cum_membrane_potentials[i]
                        - leaky_cum_membrane_potentials[i].detach()
                    )
                    cum_potential_history[i].append(leaky_cum_membrane_potentials[i].detach())

        #output = F.log_softmax(leaky_cum_membrane_potentials[-1] / steps, dim=1)
        output = leaky_cum_membrane_potentials[-1] / steps
        #print(leaky_cum_membrane_potentials[-1][0] / steps)
        #print(output[0])
        #print()
        result = dict(
            output=output,  # membrane_potentials[-1],
            total_outs=total_outs,
            lf_outs=LF_outs,
            out_temps=out_temps,
            input_history=input_history,
            potential_history=potential_history,
            cum_potential_history=cum_potential_history,
        )

        return result
