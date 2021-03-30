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
    print("loss parameters:")
    for k, v in params.items():
        print(k, v)

    if loss.lower() == "crossentropy":
        loss_function = CustomCrossEntropyLoss(verbose=verbose)
    elif loss.lower() == "mse":
        loss_function = CustomMSELoss(reduction="sum", verbose=verbose, **params)
    elif loss.lower() == "custom":
        loss_function = CustomSNNLoss(reduction="sum", verbose=verbose, **params)
    elif loss.lower() == "beta":
        loss_function = CustomBetaLoss(reduction="sum", verbose=verbose, **params)
    else:
        raise NotImplementedError(
            f"The loss function {loss} is not implemented.\n"
            f"Valid options are: 'crossentropy', 'mse', 'beta', or 'custom'."
        )
    if verbose:
        print(f"Initialized loss function:\n{loss_function}\n")

    return loss_function


class CustomMSELoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
            self,
            l1=0.0,
            l2=0.0,
            example2=0.0,
            neuron2=0.0,
            reduction: str = "mean",
            verbose=False,
            **kwargs,
    ) -> None:
        super(CustomMSELoss, self).__init__(reduction=reduction)
        self.l1 = l1
        self.l2 = l2
        self.example2 = example2
        self.neuron2 = neuron2
        self.verbose = verbose

        self.loss_labels = [
            "loss",
            "reconstruction loss",
            "L1 weight loss",
            "L2 weight loss",
            "example activity loss 2",
            "neuron activity loss 2",
            "mean pixelwise error",
            "own mse",
        ]

    def forward(self, **result) -> Tensor:

        batch_size = result["input"].size(0)

        reconstruction_loss = F.mse_loss(result["output"], result["target"], reduction=self.reduction).div(batch_size)

        own_mse = (result["output"] - result["target"]) ** 2
        own_mse = torch.mean(own_mse)
        pixelwise_error = (result["output"] - result["target"]).abs().mean()

        l1_weights = 0.0
        l2_weights = 0.0
        l2_example = 0.0
        l2_neuron = 0.0

        if "latent" in result.keys():
            z = result["latent"]
            l2_example = self.example2 * z.abs().sum(dim=1).square().sum().div(batch_size)
            l2_neuron = self.neuron2 * z.abs().sum(dim=0).square().sum().div(batch_size)

        for weights in result["weights"]:
            l1_weights = l1_weights + self.l1 * weights.abs().sum()
            l2_weights = l2_weights + self.l2 * weights.square().sum()

        if self.verbose:
            print()
            print("mean pixelwise error:", pixelwise_error)
            print("average reconstruction loss:", reconstruction_loss)
            print("L1 weight loss:", l1_weights)
            print("L2 weight loss:", l2_weights)
            print("average example activity loss 2:", l2_example)
            print("average neuron activity loss 2:", l2_neuron)
            print("own mse", own_mse)

        loss = reconstruction_loss + l1_weights + l2_weights + l2_neuron + l2_example

        losses = {
            "loss": loss,
            "reconstruction loss": loss,
            "L1 weight loss": l1_weights,
            "L2 weight loss": l2_weights,
            "example activity loss 2": l2_example,
            "neuron activity loss 2": l2_neuron,
            "mean pixelwise error": pixelwise_error,
            "own mse": own_mse,
        }

        return losses


class CustomCrossEntropyLoss(_WeightedLoss):
    __constants__ = ["ignore_index", "reduction"]
    ignore_index: int

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        verbose=False,
        **kwargs,
    ) -> None:
        super(CustomCrossEntropyLoss, self).__init__(
            weight, reduction=reduction
        )
        self.ignore_index = ignore_index
        self.verbose = verbose

    def forward(self, **result) -> Tensor:
        return F.cross_entropy(
            result["output"],
            result["target"],
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class CustomSNNLoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(
        self,
        beta1=0.0,
        beta2=0.0,
        lambd2=0.0,
        lambd1=0.0,
        l1=0.0,
        l2=0.0,
        example2=0.0,
        neuron2=0.0,
        neuron1=0.0,
        layers=0,
        reduction: str = "mean",
        verbose=False,
        **kwargs,
    ) -> None:
        super(CustomSNNLoss, self).__init__(reduction=reduction)
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.example2 = example2
        self.neuron2 = neuron2
        self.neuron1 = neuron1
        self.l1 = l1
        self.l2 = l2
        self.latent_index = (layers / 2) - 1
        self.verbose = verbose


        self.loss_labels = [
            "loss",
            "reconstruction loss",
            "burst loss 1",
            "potential loss 1",
            "burst loss 2",
            "potential loss 2",
            "L1 weight loss",
            "L2 weight loss",
            "example activity loss 2",
            "neuron activity loss 2",
            "neuron activity loss 1",
            "mean pixelwise error",
            "own mse",
        ]

        for i in range(layers):
            self.loss_labels.append(f"density layer {i+1}")
            self.loss_labels.append(f"pct inactive layer {i+1}")

    def forward(self, **result) -> Tensor:

        batch_size = result["cum_potential_history"][0][-1].size(0)

        reconstruction_loss = F.mse_loss(
            result["output"], result["target"], reduction=self.reduction
        ).div(batch_size)

        l1_burst = 0.0
        l2_burst = 0.0
        l1_potential = 0.0
        l2_potential = 0.0
        l1_weights = 0.0
        l2_weights = 0.0
        l2_example = 0.0
        l2_neuron = 0.0
        l1_neuron = 0.0
        spike_densities = []
        pct_inactive_neurons = []

        t = len(result["cum_potential_history"][0])

        for i in range(len(result["cum_potential_history"])):

            N = 1
            for j in [*result["cum_potential_history"][i][-1].size()][1:]:
                N *= j

            spikes = result["total_outs"][i]

            inactive = (spikes.sum(dim=0) == 0).sum(dtype=float) / N
            pct_inactive_neurons.append(inactive)

            spike_densities.append(spikes.sum() / (N * (1. - inactive) * t * batch_size))

            activity = result["cum_potential_history"][i][-1]
            weights = result["weights"][i]

            if self.verbose:
                print()
                print("layer", i+1)
                print("units:", N)
                print("pct inactive neurons:", inactive)
                print("average normalized spike density:", spike_densities[-1])


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

            l1_burst = l1_burst + self.lambd1 * spikes.div(t).abs().sum().div(batch_size)
            l2_burst = l2_burst + self.lambd2 * spikes.div(t).square().sum().div(batch_size)
            l1_potential = l1_potential + self.beta1 * activity.div(t).abs().sum().div(batch_size)
            l2_potential = l2_potential + self.beta2 * activity.div(t).square().sum().div(batch_size)
            l1_weights = l1_weights + self.l1 * weights.abs().sum()
            l2_weights = l2_weights + self.l2 * weights.square().sum()

            if i == self.latent_index:
                l2_example = self.example2 * spikes.div(t).sum(dim=1).square().sum().div(batch_size)
                l2_neuron = self.neuron2 * spikes.div(t).sum(dim=0).square().sum().div(batch_size)
                l1_neuron = self.neuron1 * spikes.div(t).sum().div(batch_size)


        own_mse = (result["output"] - result["target"]) ** 2
        own_mse = torch.mean(own_mse)
        pixelwise_error = (result["output"] - result["target"]).abs().mean()

        if self.verbose:
            print()
            print("pixelwise error:", pixelwise_error)
            print("average reconstruction loss:", reconstruction_loss)
            print("average burst loss 1:", l1_burst)
            print("average burst loss 2:", l2_burst)
            print("average potential loss 1:", l1_potential)
            print("average potential loss 2:", l2_potential)
            print("L1 weight loss:", l1_weights)
            print("L2 weight loss:", l2_weights)
            print("average example activity loss 2:", l2_example)
            print("average neuron activity loss 1:", l1_neuron)
            print("average neuron activity loss 2:", l2_neuron)
            print("own mse:", own_mse)

        loss = reconstruction_loss + l1_burst + l2_burst + l1_potential + l2_potential +\
               l1_weights + l2_weights + l2_example + l1_neuron + l2_neuron
        
        losses = {
            "loss": loss,
            "reconstruction loss": reconstruction_loss,
            "burst loss 1": l1_burst,
            "burst loss 2": l2_burst,
            "potential loss 1": l1_potential,
            "potential loss 2": l2_potential,
            "L1 weight loss": l1_weights,
            "L2 weight loss": l2_weights,
            "example activity loss 2": l2_example,
            "neuron activity loss 1": l1_neuron,
            "neuron activity loss 2": l2_neuron,
            "mean pixelwise error": pixelwise_error,
            "own mse": own_mse,
        }

        for i, density in enumerate(spike_densities):
            losses.update({f"density layer {i+1}": density})

        for i, pct_inactive in enumerate(pct_inactive_neurons):
            losses.update({f"pct inactive layer {i+1}": pct_inactive})

        return losses


class CustomBetaLoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(
        self,
        beta1=1.0,
        beta2=1.0,
        l1=0.0,
        l2=0.0,
        example2=0.0,
        neuron2=0.0,
        rate=0.9999,
        reduction: str = "mean",
        verbose=False,
        **kwargs,
    ) -> None:
        super(CustomBetaLoss, self).__init__(reduction=reduction)
        self.beta = beta1
        self.start = beta1
        self.end = beta2
        self.rate = rate
        self.l1 = l1
        self.l2 = l2
        self.example2 = example2
        self.neuron2 = neuron2
        self.verbose = verbose

        self.loss_labels = [
            "loss",
            "reconstruction loss",
            "KL loss",
            "L1 weight loss",
            "L2 weight loss",
            "example activity loss 2",
            "neuron activity loss 2",
            "mean pixelwise error",
            "mean KLD",
            "own mse",
            "beta",
        ]

    def forward(self, **result) -> Tensor:

        batch_size = result["input"].size(0)

        reconstruction_loss = F.mse_loss(
            result["output"], result["target"], reduction=self.reduction
        ).div(batch_size)

        own_mse = (result["output"] - result["target"]) ** 2
        own_mse = torch.mean(own_mse)
        pixelwise_error = (result["output"] - result["target"]).abs().mean()

        total_kld, dim_wise_kld, mean_kld = kl_divergence(result["mu"], result["logvar"])
        beta_loss = self.beta * total_kld
        l1_weights = 0.0
        l2_weights = 0.0
        l2_example = 0.0
        l2_neuron = 0.0

        if "latent" in result.keys():
            z = result["latent"]
            l2_example = self.example2 * z.abs().sum(dim=1).square().sum().div(batch_size)
            l2_neuron = self.neuron2 * z.abs().sum(dim=0).square().sum().div(batch_size)

        for weights in result["weights"]:
            l1_weights = l1_weights + self.l1 * weights.abs().sum()
            l2_weights = l2_weights + self.l2 * weights.square().sum()

        if self.verbose:
            print()
            print("mean pixelwise error:", pixelwise_error)
            print("average reconstruction loss:", reconstruction_loss)
            print("average KL loss:", beta_loss)
            print("mean KLD:", mean_kld)
            print("L1 weight loss:", l1_weights)
            print("L2 weight loss:", l2_weights)
            print("average example activity loss 2:", l2_example)
            print("average neuron activity loss 2:", l2_neuron)
            print("own mse", own_mse)
            print("beta:", self.beta)

        loss = reconstruction_loss + beta_loss + l1_weights + l2_weights

        losses = {
            "loss": loss,
            "reconstruction loss": reconstruction_loss,
            "KL loss": beta_loss,
            "L1 weight loss": l1_weights,
            "L2 weight loss": l2_weights,
            "example activity loss 2": l2_example,
            "neuron activity loss 2": l2_neuron,
            "mean pixelwise error": pixelwise_error,
            "mean KLD": mean_kld,
            "beta": torch.tensor(self.beta),
            "own mse": own_mse,
        }

        return losses


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld