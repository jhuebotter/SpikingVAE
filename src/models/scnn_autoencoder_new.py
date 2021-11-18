import torch
import torch.nn as nn
from models.base_model import SpikingBaseModel
from models.spiking_layers import Conv2dLIF
from models.input_encoders import get_input_encoder
from models.output_decoders import get_output_decoder
import utils as u


class SpikingConvolutionalAutoencoderNew(SpikingBaseModel):
    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        conv2d_channels,
        hidden_sizes,
        loss,
        optimizer,
        learning_rate,
        weight_decay,
        device,
        kernel_size=3,
        stride=1,
        padding=1,
        #pooling_kernel=2,
        #pooling_stride=1,
        #activation="lif",
        #activation_out="lif",
        #pooling="avg",
        steps=100,
        threshold=1,
        decay=0.99,
        adapt_threshold=True,
        threshold_width=0.1,
        delta_threshold=0.0001,
        rho=0.0001,
        epsilon=0.05,
        inactivity_threshold=0,
        delta_w=0.1,
        #pool_threshold=0.75,
        encoder_params=dict(encoder="first"),
        decoder_params=dict(decoder="max"),
        grad_clip=0.0,
        use_extra_grad=True,
        verbose=True,
        log_func=print,
        reset=True,
    ):
        super(SpikingConvolutionalAutoencoderNew, self).__init__(
            input_width=input_width,
            input_height=input_height,
            input_channels=input_channels,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            steps=steps,
            threshold=threshold,
            decay=decay,
            device=device,
            loss=loss,
            grad_clip=grad_clip,
            use_extra_grad=use_extra_grad,
            verbose=verbose,
            log_func=log_func,
        )

        # initialize model
        if verbose:
            self.log_func("Initializing spiking convolutional autoencoder...")

        self.task = "reconstruction"
        self.hidden_sizes = hidden_sizes
        self.conv2d_channels = conv2d_channels
        self.kernel_sizes = [kernel_size for i in range(len(conv2d_channels))]
        self.strides = [stride for i in range(len(conv2d_channels))]
        self.paddings = [padding for i in range(len(conv2d_channels))]
        #self.pooling_kernels = [pooling_kernel for i in range(len(conv2d_channels))]
        #self.pooling_strides = [pooling_stride for i in range(len(conv2d_channels))]
        self.model = SCNNAutoencoderModelNew(
            input_width=self.input_width,
            input_height=self.input_height,
            input_channels=self.input_channels,
            conv2d_channels=self.conv2d_channels,
            hidden_sizes=self.hidden_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            #pooling_kernels=self.pooling_kernels,
            #pooling_strides=self.pooling_strides,
            #activation=activation,
            #activation_out=activation_out,
            #pooling=pooling,
            threshold=threshold,
            decay=decay,
            adapt_threshold=adapt_threshold,
            threshold_width=threshold_width,
            delta_threshold=delta_threshold,
            rho=rho,
            epsilon=epsilon,
            inactivity_threshold=inactivity_threshold,
            delta_w=delta_w,
            #pool_threshold=pool_threshold,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            device=device,
            steps=steps,
            reset=reset,
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
        self.loss_function = loss  # losses.get_loss_function(loss, verbose, spiking=True)

        self.input_layer = self.model.conv_encoder_layers["sconv2d1"]
        self.output_layer = self.model.conv_decoder_layers[f"sconvT2d{len(self.conv2d_channels)}"]

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


class SCNNAutoencoderModelNew(nn.Module):
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
        #pooling_kernels,
        #pooling_strides,
        #activation="lif",
        #activation_out="lif",
        pooling="avg",
        steps=100,
        threshold=1,
        decay=0.99,
        adapt_threshold=True,
        threshold_width=0.1,
        delta_threshold=0.0001,
        rho=0.0001,
        epsilon=0.05,
        inactivity_threshold=0,
        delta_w=0.01,
        #pool_threshold=0.75,
        encoder_params={"encoder": "first"},
        decoder_params={"decoder": "max"},
        device="cuda",
        reset=True,
    ):
        super(SCNNAutoencoderModelNew, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.conv2d_channels = conv2d_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        #self.pooling_kernels = pooling_kernels
        #self.pooling_strides = pooling_strides
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.steps = steps
        self.reset = reset

        # define convolutional encoder layers
        sconv_encoder_args = dict(
            input_channels=self.input_channels,
            channels=self.conv2d_channels,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            #activations=[activation for i in range(len(self.conv2d_channels))],
            #pooling_funcs=[pooling for i in range(len(self.conv2d_channels))],
            #pooling_kernels=self.pooling_kernels,
            #pooling_strides=self.pooling_strides,
            thresholds=[threshold for i in range(len(self.conv2d_channels))],
            decays=[decay for i in range(len(self.conv2d_channels))],
            adapt_thresh=adapt_threshold,
            threshold_widths=[threshold_width for i in range(len(self.conv2d_channels))],
            delta_thresholds=[delta_threshold for i in range(len(self.conv2d_channels))],
            rhos=[rho for i in range(len(self.conv2d_channels))],
            epsilons=[epsilon for i in range(len(self.conv2d_channels))],
            inactivity_thresholds=[inactivity_threshold for i in range(len(self.conv2d_channels))],
            delta_ws=[delta_w for i in range(len(self.conv2d_channels))],
            device=self.device,
            reset=self.reset,
            #pool_thresholds=[pool_threshold for i in range(len(self.conv2d_channels))],
        )
        sconv_encoder_parameters = u.get_sconv2dlif_layers(**sconv_encoder_args)
        self.conv_encoder_layers = u.build_layers(sconv_encoder_parameters).to(self.device)

        # get flattend input size
        x = torch.randn(self.input_channels, self.input_width, self.input_height).view(
            -1, self.input_channels, self.input_width, self.input_height
        ).to(self.device)

        x_ = self.conv_encode(x)
        self._to_linear = x_[0].shape[0] * x_[0].shape[1] * x_[0].shape[2]
        self._from_linear = (x_[0].shape[0], x_[0].shape[1], x_[0].shape[2])

        # define fully connected encoder layers
        #self.fc_encoder_activations = [activation for i in range(len(hidden_sizes))]

        sfc_encoder_args = dict(
            input_size=self._to_linear,
            hidden_sizes=self.hidden_sizes[:-1],
            output_size=self.hidden_sizes[-1],
            #activations=self.fc_encoder_activations,
            thresholds=[threshold for i in range(len(self.hidden_sizes))],
            decays=[decay for i in range(len(self.hidden_sizes))],
            adapt_thresh=adapt_threshold,
            threshold_widths=[threshold_width for i in range(len(self.hidden_sizes))],
            delta_thresholds=[delta_threshold for i in range(len(self.hidden_sizes))],
            rhos=[rho for i in range(len(self.hidden_sizes))],
            inactivity_thresholds=[inactivity_threshold for i in range(len(self.hidden_sizes))],
            delta_ws=[delta_w for i in range(len(self.hidden_sizes))],
            epsilons=[epsilon for i in range(len(self.hidden_sizes))],
            device=self.device,
            reset=self.reset,
        )

        sfc_encoder_parameters = u.get_sfclif_layers(**sfc_encoder_args)
        sfc_encoder_parameters[-1]["parameters"]["history"] = True
        self.fc_encoder_layers = u.build_layers(sfc_encoder_parameters)

        # define fully connected decoder layers
        #self.fc_decoder_activations = [activation for i in range(len(hidden_sizes))]
        decoder_sizes = self.hidden_sizes[:-1]
        decoder_sizes.reverse()

        sfc_decoder_args = dict(
            input_size=self.hidden_sizes[-1],
            hidden_sizes=decoder_sizes,
            output_size=self._to_linear,
            #activations=self.fc_decoder_activations,
            thresholds=[threshold for i in range(len(self.hidden_sizes))],
            decays=[decay for i in range(len(self.hidden_sizes))],
            adapt_thresh=adapt_threshold,
            threshold_widths=[threshold_width for i in range(len(self.hidden_sizes))],
            delta_thresholds=[delta_threshold for i in range(len(self.hidden_sizes))],
            rhos=[rho for i in range(len(self.hidden_sizes))],
            epsilons=[epsilon for i in range(len(self.hidden_sizes))],
            inactivity_thresholds=[inactivity_threshold for i in range(len(self.hidden_sizes))],
            delta_ws=[delta_w for i in range(len(self.hidden_sizes))],
            device=self.device,
            reset=self.reset,
        )

        sfc_decoder_parameters = u.get_sfclif_layers(**sfc_decoder_args)
        self.fc_decoder_layers = u.build_layers(sfc_decoder_parameters)

        # define convolutional decoder layers
        decoder_kernel_sizes = self.kernel_sizes
        decoder_kernel_sizes.reverse()
        decoder_convtranspose2d_channels = [self.input_channels] + self.conv2d_channels[:-1]
        decoder_convtranspose2d_channels.reverse()
        decoder_strides = self.strides
        decoder_strides.reverse()
        decoder_paddings = self.paddings
        decoder_paddings.reverse()
        #unpooling_kernels = self.pooling_kernels
        #unpooling_kernels.reverse()
        #unpooling_strides = self.pooling_strides
        #unpooling_strides.reverse()

        sconv_decoder_args = dict(
            input_channels=x_[0].shape[0],
            channels=decoder_convtranspose2d_channels,
            kernel_sizes=decoder_kernel_sizes,
            strides=decoder_strides,
            paddings=decoder_paddings,
            #activations=[activation for i in range(len(self.conv2d_channels) - 1)] + [activation_out],
            #unpooling_funcs=[pooling for i in range(len(self.conv2d_channels))],
            #unpooling_kernels=unpooling_kernels,
            #unpooling_strides=unpooling_strides,
            thresholds=[threshold for i in range(len(self.conv2d_channels))],
            decays=[decay for i in range(len(self.conv2d_channels))],
            adapt_thresh=adapt_threshold,
            threshold_widths=[threshold_width for i in range(len(self.conv2d_channels))],
            delta_thresholds=[delta_threshold for i in range(len(self.conv2d_channels))],
            rhos=[rho for i in range(len(self.conv2d_channels))],
            epsilons=[epsilon for i in range(len(self.conv2d_channels))],
            inactivity_thresholds=[inactivity_threshold for i in range(len(self.conv2d_channels))],
            delta_ws=[delta_w for i in range(len(self.conv2d_channels))],
            device=self.device,
            reset=self.reset,
            #pool_thresholds=[pool_threshold for i in range(len(self.conv2d_channels))],
        )

        sconv_decoder_parameters = u.get_sconvtranspose2dlif_layers(**sconv_decoder_args)
        sconv_decoder_parameters[-1]["parameters"]["history"] = True
        self.conv_decoder_layers = u.build_layers(sconv_decoder_parameters)

        # initialize input encoder
        self.input_encoder = get_input_encoder(**encoder_params)

        # initialize output decoder
        self.output_decoder = get_output_decoder(**decoder_params)



    def conv_encode(self, x):
        """Passes data through convolutional layers.

        :param x: Tensor with input data.

        :return Tensor with output data.
        """

        conv_modules = [
            c for c in self.conv_encoder_layers.values() if type(c) in [Conv2dLIF]
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
            m for m in self.modules()
            if not issubclass(type(m), nn.modules.container.ModuleDict)
        ][1:]

        for layer in modules:
            layer.reset_state()

        for t in range(self.steps):

            out = self.input_encoder.encode(x, t)

            for layer in self.conv_encoder_layers.values():
                out = layer(out)

            out = out.view(out.size(0), -1)

            for layer in self.fc_encoder_layers.values():
                out = layer(out)

            for layer in self.fc_decoder_layers.values():
                out = layer(out)

            out = out.view(-1, *self._from_linear)

            for layer in self.conv_decoder_layers.values():
                out = layer(out)


        output = self.output_decoder.decode(layer.cumulative_potential_history)

        total_outs = [l.total_out for l in modules]
        #LF_outs = [l.LF_out for l in modules]
        potential_history = [l.potential_history if l.history else [] for l in modules]
        cum_potential_history = [l.cumulative_potential_history if l.history else [] for l in modules]
        out_temps = [l.out_temp for l in modules]

        result = dict(
            output=output,
            total_outs=total_outs,
            #lf_outs=LF_outs,
            out_temps=out_temps,
            input_history=self.input_encoder.input_history,
            potential_history=potential_history,
            cum_potential_history=cum_potential_history,
        )

        self.input_encoder.reset()

        return result
