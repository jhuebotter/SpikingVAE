import torch
import torch.nn as nn


class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(0).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        (input,) = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0.0] = 0
        return grad_input


def lif_sneuron(membrane_potential, threshold, decay):
    """

    """
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane
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
        return lif_sneuron(
            membrane_potential, self.threshold, self.decay
        )


def pooling_sneuron(membrane_potential, threshold):

    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
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
        return pooling_sneuron(
            membrane_potential, self.threshold
        )

def LF_Unit(l, LF_output, Total_output, out, out_temp, t):
    LF_output = l * LF_output + out
    Total_output = Total_output + out
    out_temp.append(out)

    return LF_output, Total_output, out_temp[t]