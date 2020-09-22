import torch
from torch import Tensor
from typing import Optional
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F


def get_loss_function(loss, verbose=True, spiking=False, params={}):
    """Sets the requested loss function if available

    :param loss: str, name of the loss function to use during training.
    :param verbose: bool, flag to print statements.

    :return loss_function: func, loss function to use during training.
    """

    if loss.lower() == "crossentropy":
        loss_function = CustomCrossEntropyLoss()
    elif loss.lower() == "mse":
        loss_function = CustomMSELoss(size_average=not spiking)
    elif loss.lower() == "custom":
        print(params)
        loss_function = CustomBetaLoss(size_average=not spiking, **params)
    else:
        raise NotImplementedError(
            f"The loss function {loss} is not implemented.\n"
            f"Valid options are: 'crossentropy', 'mse', or 'custom'."
        )
    if verbose:
        print(f"Initialized loss function:\n{loss_function}\n")

    return loss_function


class CustomMSELoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(CustomMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, **result) -> Tensor:

        return F.mse_loss(result["output"], result["target"], reduction=self.reduction)


class CustomCrossEntropyLoss(_WeightedLoss):
    __constants__ = ["ignore_index", "reduction"]
    ignore_index: int

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(CustomCrossEntropyLoss, self).__init__(
            weight, size_average, reduce, reduction
        )
        self.ignore_index = ignore_index

    def forward(self, **result) -> Tensor:
        return F.cross_entropy(
            result["output"],
            result["target"],
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class CustomBetaLoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(
        self,
        beta1=0.0,
        beta2=0.0,
        lambd2=0.0,
        lambd1=0.0,
        l1=0.0,
        l2=0.0,
        layers=0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(CustomBetaLoss, self).__init__(size_average, reduce, reduction)
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.l1 = l1
        self.l2 = l2

        self.loss_labels = [
            "loss",
            "reconstruction loss",
            "burst loss 1",
            "potential loss 1",
            "burst loss 2",
            "potential loss 2",
            "L1 weight loss",
            "L2 weight loss",
        ]
        for i in range(layers):
            self.loss_labels.append(f"density layer {i+1}")
            self.loss_labels.append(f"pct inactive layer {i+1}")

    def forward(self, **result) -> Tensor:

        reconstruction_loss = F.mse_loss(
            result["output"], result["target"], reduction=self.reduction
        )

        l1_burst = 0.0
        l2_burst = 0.0
        l1_potential = 0.0
        l2_potential = 0.0
        l1_weights = 0.0
        l2_weights = 0.0
        spike_densities = []
        pct_inactive_neurons = []

        t = len(result["cum_potential_history"][0])
        batch_size = result["cum_potential_history"][0][-1].size(0)

        for i in range(len(result["cum_potential_history"])):

            N = 1
            for j in [*result["cum_potential_history"][i][-1].size()][1:]:
                N *= j

            print()
            print("hidden layer", i)
            print("units:", N)

            spikes = result["total_outs"][i]
            spike_densities.append(spikes.sum() / (N * t))
            print(
                "average normalized spike density:", spikes.sum() / (N * t * batch_size)
            )
            inactive = (spikes.sum(dim=0) == 0).sum(dtype=float) / N
            pct_inactive_neurons.append(inactive)
            print("pct inactive neurons:", inactive)

            activity = result["cum_potential_history"][i][-1]
            weights = result["weights"][i]

            """
            print("squared sum / N:", spikes.square().sum() / N)

            print(
                "average normalized cum membrane potential:",
                activity.sum() / (N * t * batch_size),
            )
            print("squared sum / N:", activity.square().sum() / N)

            activity = activity - target_activity
            print(
                "average normalized cum membrane potential - target:",
                activity.sum() / (N * t * batch_size),
            )
            print("squared sum - target / N:", activity.square().sum() / N)
            """

            l1_burst = l1_burst + self.lambd1 * spikes.div(t).abs().sum()
            l2_burst = l2_burst + self.lambd2 * spikes.div(t).square().sum()
            l1_potential = l1_potential + self.beta1 * activity.div(t).abs().sum()
            l2_potential = l2_potential + self.beta2 * activity.div(t).square().sum()
            l1_weights = l1_weights + self.l1 * weights.abs().sum()
            l2_weights = l2_weights + self.l2 * weights.square().sum()

        print()
        print("average reconstruction loss:", reconstruction_loss / batch_size)
        print("average burst loss 1:", l1_burst / batch_size)
        print("average burst loss 2:", l2_burst / batch_size)
        print("average potential loss 1:", l1_potential / batch_size)
        print("average potential loss 2:", l2_potential / batch_size)
        print("L1 weight loss:", l1_weights)
        print("L2 weight loss:", l2_weights)

        loss = reconstruction_loss + l1_burst + l2_burst + l1_potential + l2_potential + l1_weights + l2_weights

        losses = {
            "loss": loss,
            "reconstruction loss": reconstruction_loss,
            "burst loss 1": l1_burst,
            "burst loss 2": l2_burst,
            "potential loss 1": l1_potential,
            "potential loss 2": l2_potential,
            "L1 weight loss": l1_weights,
            "L2 weight loss": l2_weights,
        }

        for i, density in enumerate(spike_densities):
            losses.update({f"density layer {i+1}": density})

        for i, pct_inactive in enumerate(pct_inactive_neurons):
            losses.update({f"pct inactive layer {i+1}": pct_inactive * batch_size})

        return losses
