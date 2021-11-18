import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair
from torch._jit_internal import List


class LinearLIF(nn.Module):

    __constants__ = ["in_features", "out_features", "decay"]

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        decay=0.99,
        initial_threshold=1,
        adapt_thresh=True,
        threshold_width=0.1,
        delta_threshold=0.0005,
        rho=0.0001,
        epsilon=0.05,
        inactivity_threshold=0,
        delta_w=0.01,
        device="cpu",
        reset=True,
        history=True
    ):
        super(LinearLIF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.decay = decay
        self.initial_threshold = initial_threshold
        self.adapt_thresh = adapt_thresh
        self.max_threshold = initial_threshold + threshold_width
        self.min_threshold = initial_threshold - threshold_width
        self.delta_threshold = delta_threshold
        self.rho = rho
        self.epsilon = epsilon
        self.reset = reset
        self.device = device
        self.inactivity_threshold = inactivity_threshold
        self.delta_w = delta_w
        self.inactive_counter = None
        self.history = history

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_state()

        self.reset_parameters()

        self.threshold = (
            torch.ones(self.out_features, device=device)
            * self.initial_threshold
        )

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def reset_state(self):

        try:
            if self.training and self.inactivity_threshold and self.initialized:

                if self.inactive_counter is None:
                    self.inactive_counter = torch.zeros(
                        self.out_features, requires_grad=False, device=self.device
                    )

                self.inactive_counter = adapt_sensitivity(
                    self,
                    self.total_out,
                    self.inactive_counter,
                    self.inactivity_threshold,
                    self.delta_w,
                    self.variance,
                )
        except AttributeError:
            print("Not yet initialized.")

        self.initialized = False


    def initialize(self, input_current):

        device = input_current.device

        self.membrane_potential = torch.zeros(
            self.out_features, requires_grad=True, device=device
        )
        self.leaky_cumulative_membrane_potential = torch.zeros(
            self.out_features, requires_grad=True, device=device
        )
        self.LF_out = torch.zeros(self.out_features, device=device)
        self.total_out = torch.zeros(self.out_features, device=device)
        if self.history:
            self.potential_history = []
            self.cumulative_potential_history = []
        self.out_temp = []
        self.initialized = True

    def reset_parameters(self):

        self.fan_in = self.weight.size()[1]
        self.variance = math.sqrt(2.0 / self.fan_in)
        self.weight.data.normal_(0.0, 2 * self.variance)

    def forward(self, input):

        input_current = F.linear(input, self.weight, self.bias)

        if not self.initialized:
            self.initialize(input_current)

        self.membrane_potential = self.membrane_potential + input_current
        self.leaky_cumulative_membrane_potential = (
            self.leaky_cumulative_membrane_potential + input_current
        )
        self.leaky_cumulative_membrane_potential = (
            (self.decay * self.leaky_cumulative_membrane_potential.detach())
            + self.leaky_cumulative_membrane_potential
            - self.leaky_cumulative_membrane_potential.detach()
        )
        if self.history:
            self.potential_history.append(self.membrane_potential.detach())
            self.cumulative_potential_history.append(
                self.leaky_cumulative_membrane_potential  # .detach()
            )

        out_spikes, self.membrane_potential = lif_sneuron(
            self.membrane_potential, self.threshold, self.decay, reset=self.reset,
        )

        self.LF_out, self.total_out, out_spikes = LF_Unit(
            self.decay, self.LF_out, self.total_out, out_spikes, self.out_temp
        )

        if self.training and self.adapt_thresh:

            adapt_threshold(self, out_spikes)

        return out_spikes



def adapt_sensitivity(self, out_spikes, inactive_counter, inactivity_threshold, delta_w, var):

    summed_spikes = torch.sum(out_spikes, dim=0).detach()

    inactive = torch.where(
        summed_spikes == 0,
        torch.ones(summed_spikes.size(), device=summed_spikes.device),
        torch.zeros(summed_spikes.size(), device=summed_spikes.device),
    )

    inactive_counter += inactive
    inactive_counter *= inactive

    long_inactivity = torch.where(
        self.inactive_counter >= inactivity_threshold,
        torch.ones(self.threshold.size(), device=self.threshold.device),
        torch.zeros(self.threshold.size(), device=self.threshold.device)
    )

    self.weight.data += long_inactivity.repeat((self.weight.data.size(1), 1)).T * delta_w * var

    return inactive_counter



def adapt_threshold(self, out_spikes):

    summed_spikes = torch.sum(out_spikes, dim=0).detach()

    summed_potential = torch.sum(self.potential_history[-1], dim=0).detach()
    batch_size = self.potential_history[-1].size(0)
    mean_potential = torch.div(summed_potential, batch_size)

    inactive = torch.where(
        summed_spikes == 0,
        torch.ones(summed_spikes.size(), device=summed_spikes.device),
        torch.zeros(summed_spikes.size(), device=summed_spikes.device),
    )

    N = 1
    for j in [*self.potential_history[-1].size()][1:]:
        N *= j

    up = self.delta_threshold
    down = self.delta_threshold * self.epsilon  # is this! the desired temporal sparsity parameter?
    factor = summed_spikes / batch_size
    self.threshold = self.threshold + up * factor
    self.threshold = self.threshold - down

    at_min_potential = torch.where(
        self.threshold <= self.min_threshold,
        torch.ones(self.threshold.size(), device=self.threshold.device),
        torch.zeros(self.threshold.size(), device=self.threshold.device)
    )

    at_max_potential = torch.where(
        self.threshold >= self.max_threshold,
        torch.ones(self.threshold.size(), device=self.threshold.device),
        torch.zeros(self.threshold.size(), device=self.threshold.device)
    )

    # cap threshold at minimum
    self.threshold = nn.functional.threshold(self.threshold, self.min_threshold, self.min_threshold)

    # cap threshold at maximum
    self.threshold = -1 * nn.functional.threshold(- 1 * self.threshold, - 1 * self.max_threshold, - 1 * self.max_threshold)

    increase = torch.mul(at_min_potential, inactive)

    print(self)
    print(self.threshold.min(), self.threshold.mean(), self.threshold.max())
    print(factor.mean())
    print(inactive.sum())
    print(at_min_potential.sum())
    print(increase.sum())
    print()

    self.weight.data += increase.repeat((self.weight.data.size(1), 1)).T * self.rho * self.variance
    # self.weight.data -= decrease.repeat((self.weight.data.size(1), 1)).T * self.rho * self.variance


class LinearLIFold(nn.Module):

    __constants__ = ["in_features", "out_features", "decay"]

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        decay=0.99,
        initial_threshold=1,
        device="cpu",
    ):
        super(LinearLIF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.decay = decay
        self.initial_threshold = initial_threshold
        self.device = device
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_state()

        self.reset_parameters()

    def reset_state(self):

        self.initialized = False

    def initialize(self, input_current):

        device = input_current.device

        self.threshold = torch.ones(input_current.size()) * self.initial_threshold
        self.membrane_potential = torch.zeros(
            self.out_features, requires_grad=True, device=device
        )
        self.leaky_cumulative_membrane_potential = torch.zeros(
            self.out_features, requires_grad=True, device=device
        )
        self.LF_out = torch.zeros(self.out_features, device=device)
        self.total_out = torch.zeros(self.out_features, device=device)
        self.potential_history = []
        self.cumulative_potential_history = []
        self.out_temp = []
        self.initialized = True

    def reset_parameters(self):

        size = self.weight.size()
        fan_in = size[1]
        variance = math.sqrt(2.0 / fan_in)
        self.weight.data.normal_(0.0, 2 * variance)


    def forward(self, input):

        input_current = F.linear(input, self.weight, self.bias)

        if not self.initialized:
            self.initialize(input_current)

        self.membrane_potential = self.membrane_potential + input_current
        self.potential_history.append(self.membrane_potential.detach())
        self.leaky_cumulative_membrane_potential = (
            self.leaky_cumulative_membrane_potential + input_current
        )
        self.leaky_cumulative_membrane_potential = (
            (self.decay * self.leaky_cumulative_membrane_potential.detach())
            + self.leaky_cumulative_membrane_potential
            - self.leaky_cumulative_membrane_potential.detach()
        )
        self.cumulative_potential_history.append(
            self.leaky_cumulative_membrane_potential  # .detach()
        )

        out_spikes, self.membrane_potential = lif_sneuron(
            self.membrane_potential, self.threshold, self.decay
        )

        self.LF_out, self.total_out, out_spikes = LF_Unit(
            self.decay, self.LF_out, self.total_out, out_spikes, self.out_temp
        )

        return out_spikes

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dLIF(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
        decay=0.99,
        adapt_thresh=True,
        initial_threshold=1,
        threshold_width=0.1,
        delta_threshold=0.0005,
        rho=0.0001,
        epsilon=0.05,
        inactivity_threshold=0,
        delta_w=0.01,
        device="cpu",
        reset=True,
        history=True
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.decay = decay
        self.initial_threshold = initial_threshold
        self.adapt_thresh = adapt_thresh
        self.max_threshold = initial_threshold + threshold_width
        self.min_threshold = initial_threshold - threshold_width
        self.delta_threshold = delta_threshold
        self.rho = rho
        self.epsilon = epsilon
        self.device = device
        self.reset = reset
        self.inactivity_threshold = inactivity_threshold
        self.delta_w = delta_w
        self.history = history

        super(Conv2dLIF, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )

        self.reset_state()
        self.reset_parameters()

    def reset_parameters(self):

        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        variance = math.sqrt(2.0 / n)
        self.weight.data.normal_(0, variance)
        self.inactive_counter = None


    def reset_state(self):

        self.initialized = False

    def initialize(self, input_current):

        device = input_current.device

        self.threshold = (
            torch.ones(input_current.size(), device=device)
            * self.initial_threshold
        )
        self.membrane_potential = torch.zeros(
            input_current.size(), requires_grad=True, device=device
        )
        self.leaky_cumulative_membrane_potential = torch.zeros(
            input_current.size(), requires_grad=True, device=device
        )
        self.LF_out = torch.zeros(input_current.size(), device=device)
        self.total_out = torch.zeros(input_current.size(), device=device)
        if self.history:
            self.potential_history = []
            self.cumulative_potential_history = []
        self.out_temp = []
        self.initialized = True

    def _conv_forward(self, input, weight):

        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):

        input_current = self._conv_forward(input, self.weight)

        if not self.initialized:
            self.initialize(input_current)

        self.membrane_potential = self.membrane_potential + input_current
        self.leaky_cumulative_membrane_potential = (
            self.leaky_cumulative_membrane_potential + input_current
        )
        self.leaky_cumulative_membrane_potential = (
            (self.decay * self.leaky_cumulative_membrane_potential.detach())
            + self.leaky_cumulative_membrane_potential
            - self.leaky_cumulative_membrane_potential.detach()
        )
        if self.history:
            self.potential_history.append(self.membrane_potential.detach())
            self.cumulative_potential_history.append(
                self.leaky_cumulative_membrane_potential  # .detach()
            )

        out_spikes, self.membrane_potential = lif_sneuron(
            self.membrane_potential, self.threshold, self.decay, reset=self.reset,
        )

        self.LF_out, self.total_out, out_spikes = LF_Unit(
            self.decay, self.LF_out, self.total_out, out_spikes, self.out_temp
        )


        """
        if self.training and self.inactivity_threshold:

            if self.inactive_counter is None:
                self.inactive_counter = torch.zeros(
                    tuple([d for i, d in enumerate(input_current.size()) if i > 0]),
                    requires_grad=False,
                    device=self.input_current.device
                )

            self.inactive_counter = adapt_sensitivity(
                self,
                out_spikes,
                self.inactive_counter,
                self.inactivity_threshold,
                self.delta_w,
                self.variance,
            )

        #if self.training and self.adapt_thresh:

            #adapt_threshold(self, out_spikes)
        
        """

        return out_spikes


class _ConvTransposeNd(_ConvNd):
    """ copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
    ):
        if padding_mode != "zeros":
            raise ValueError(
                'Only "zeros" padding mode is supported for {}'.format(
                    self.__class__.__name__
                )
            )

        super(_ConvTransposeNd, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
        )

    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})".format(
                        k, k + 2, len(output_size)
                    )
                )

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = (
                    (input.size(d + 2) - 1) * stride[d]
                    - 2 * padding[d]
                    + kernel_size[d]
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class ConvTranspose2dLIF(_ConvTransposeNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=False,
        dilation=1,
        padding_mode="zeros",
        decay=0.99,
        adapt_thresh=True,
        initial_threshold=1,
        threshold_width=0.1,
        delta_threshold=0.0005,
        rho=0.0001,
        epsilon=0.05,
        inactivity_threshold=0,
        delta_w=0.01,
        device="cpu",
        reset=True,
        history=True,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        self.decay = decay
        self.initial_threshold = initial_threshold
        self.adapt_thresh = adapt_thresh
        self.max_threshold = initial_threshold + threshold_width
        self.min_threshold = initial_threshold - threshold_width
        self.delta_threshold = delta_threshold
        self.rho = rho
        self.epsilon = epsilon
        self.device = device
        self.reset = reset
        self.inactivity_threshold = inactivity_threshold
        self.delta_w = delta_w
        self.history = history

        super(ConvTranspose2dLIF, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
        )

        self.reset_state()
        self.reset_parameters()

    def reset_parameters(self):

        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        variance = math.sqrt(2.0 / n)
        self.weight.data.normal_(0, variance)
        self.inactive_counter = None


    def reset_state(self):
        self.initialized = False

    def initialize(self, input_current):

        device = input_current.device

        self.threshold = (
            torch.ones(input_current.size(), device=device)
            * self.initial_threshold
            # self.initial_threshold
        )  # torch.ones(input_current.size()) * self.initial_threshold
        self.membrane_potential = torch.zeros(
            input_current.size(), requires_grad=True, device=device
        )
        self.leaky_cumulative_membrane_potential = torch.zeros(
            input_current.size(), requires_grad=True, device=device
        )
        self.LF_out = torch.zeros(input_current.size(), device=device)
        self.total_out = torch.zeros(input_current.size(), device=device)
        if self.history:
            self.potential_history = []
            self.cumulative_potential_history = []
        self.out_temp = []
        self.initialized = True

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size
        )

        input_current = F.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

        if not self.initialized:
            self.initialize(input_current)

        self.membrane_potential = self.membrane_potential + input_current
        self.leaky_cumulative_membrane_potential = (
            self.leaky_cumulative_membrane_potential + input_current
        )
        self.leaky_cumulative_membrane_potential = (
            (self.decay * self.leaky_cumulative_membrane_potential.detach())
            + self.leaky_cumulative_membrane_potential
            - self.leaky_cumulative_membrane_potential.detach()
        )
        if self.history:
            self.potential_history.append(self.membrane_potential.detach())
            self.cumulative_potential_history.append(
                self.leaky_cumulative_membrane_potential  # .detach()
            )

        out_spikes, self.membrane_potential = lif_sneuron(
            self.membrane_potential, self.threshold, self.decay, reset=self.reset,
        )

        self.LF_out, self.total_out, out_spikes = LF_Unit(
            self.decay, self.LF_out, self.total_out, out_spikes, self.out_temp
        )

        """
        if self.training and self.inactivity_threshold:

            if self.inactive_counter is None:
                self.inactive_counter = torch.zeros(
                    self.out_features, requires_grad=False, device=self.input_current.device
                )

            self.inactive_counter = adapt_sensitivity(
                self,
                out_spikes,
                self.inactive_counter,
                self.inactivity_threshold,
                self.delta_w,
                self.variance,
            )

        #if self.training and self.adapt_thresh:

            #adapt_threshold(self, out_spikes)
        """

        return out_spikes


class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        # TODO: check if this works on cpu
        return input.gt(0).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        (input,) = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0.0] = 0  # ???????
        return grad_input


def lif_sneuron(membrane_potential, threshold, decay, reset=False):
    """

    """
    # potential can not be more negative than -threshold
    # nn.functional.threshold(membrane_potential, -threshold, -threshold, True)

    membrane_potential = torch.where(
        membrane_potential < (-1 * threshold), threshold, membrane_potential
    )

    # check exceed membrane potential
    # ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)

    ex_membrane = torch.where(
        membrane_potential > threshold,
        membrane_potential,
        torch.zeros(membrane_potential.size(), device=membrane_potential.device),
    )

    # TODO: switch between mask and ex membrane
    # TODO: include hyperpolarization

    if reset:
        # reset potential to 0
        membrane_potential = membrane_potential - ex_membrane
    else:
        # subtract the firing threshold and keep excess potential
        mask_membrane = torch.as_tensor(
            (ex_membrane - threshold) > 0, dtype=torch.int32, device=ex_membrane.device
        )
        # mask_membrane = nn.functional.threshold(-1 * ex_membrane, -0.0001, threshold)
        membrane_potential = membrane_potential - mask_membrane

    # generate spike
    out = SpikingNN().apply(ex_membrane)
    # decay
    membrane_potential = (
        decay * membrane_potential.detach()
        + membrane_potential
        - membrane_potential.detach()
    )
    out = out.detach() + torch.div(out, threshold) - torch.div(out, threshold).detach()

    return out, membrane_potential


class LIF_sNeuron(nn.Module):
    def __init__(self, threshold=1, decay=0.99):
        """
        Init method.
        """
        super().__init__()  # init the base class
        self.threshold = threshold
        self.decay = decay

    def forward(self, membrane_potential):
        """
        Forward pass of the function.
        """
        return lif_sneuron(membrane_potential, self.threshold, self.decay)


def pooling_sneuron(membrane_potential, threshold):

    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane  # hard reset
    # generate spike
    out = SpikingNN.apply(ex_membrane)

    return out, membrane_potential


class Pooling_sNeuron(nn.Module):
    def __init__(self, threshold=0.75):
        """
        Init method.
        """
        super().__init__()  # init the base class
        self.threshold = threshold

    def forward(self, membrane_potential):
        """
        Forward pass of the function.
        """
        return pooling_sneuron(membrane_potential, self.threshold)


def LF_Unit(l, LF_output, Total_output, out, out_temp):

    LF_output = l * LF_output + out
    Total_output = Total_output + out
    out_temp.append(out)

    return LF_output, Total_output, out_temp[-1]
