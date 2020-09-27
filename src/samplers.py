import numpy as np
import torch
import io
from PIL import Image
from torchvision.utils import make_grid
import wandb
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import pandas as pd
import seaborn as sns

LAYERS = [2]
MAX_SAMPLES = 20


def get_samplers(sampler_names):

    samplers = dict()

    if "plot_reconstruction" in sampler_names:
        samplers.update(
            {
                "plot_reconstruction": ReconstructionSampler(
                    cbar=True,
                    max_samples=MAX_SAMPLES
                )
            }
        )
    if "plot_output_spikes" in sampler_names:
        samplers.update(
            {
                "plot_output_spikes": SpikeSampler(
                    layers=LAYERS,
                    plot_small=True,
                    plot_large=True,
                    max_samples=MAX_SAMPLES,
                )
            }
        )
    if "plot_cummulative_potential" in sampler_names:
        cumulative = True
    else:
        cumulative = False
    if "plot_output_potential" in sampler_names:
        samplers.update(
            {
                "plot_output_potential": PotentialSampler(
                    layers=LAYERS,
                    cumulative=cumulative,
                    max_samples=MAX_SAMPLES,
                )
            }
        )
    if "plot_activity_matrix" in sampler_names:
        samplers.update(
            {
                "plot_activity_matrix": MatrixSampler(
                    layers=LAYERS,
                    max_samples=MAX_SAMPLES,
                )
            }
        )
    if "plot_filters" in sampler_names:
        samplers.update(
            {
                "plot_filters": FilterSampler(
                    nrow=8,
                    max_samples=MAX_SAMPLES,
                )
            }
        )
    if "plot_histograms" in sampler_names:
        samplers.update(
            {
                "plot_histograms": SpikeHistSampler(
                    nrow=8,
                    max_samples=MAX_SAMPLES,
                    layers=LAYERS,
                )
            }
        )

    return samplers


class BaseSampler:
    def __init__(self, **kwargs):
        self.max_samples = kwargs.pop("max_samples")

    def sample(self, **result):
        # to be implemented by child classes
        pass

    def __call__(self, **results):
        return self.sample(**results)


class SpikeHistSampler(BaseSampler):
    def __init__(self, nrow=8, layers=[], **kwargs):

        super().__init__(**kwargs)
        self.nrow = nrow
        self.layers = layers

    def sample(self, **result):

        sample = dict()

        spikes = result["total_outs"]
        t = len(result["out_temps"][0])
        sample.update(
            plot_spike_hist(spikes, layers=self.layers, max_samples=self.max_samples, nrow=self.nrow, t=t)
        )

        print(sample)

        return sample


def plot_spike_hist(spikes, layers, max_samples, nrow=8, t=100):

    sample = dict()

    for idx in layers:
        layer_spikes = spikes[idx].cpu().numpy()
        batch_size = layer_spikes.shape[0]
        n = layer_spikes.shape[1]

        ncols = min(batch_size, 8)
        nrows = batch_size // ncols + 1
        if batch_size % ncols == 0:
            nrows -= 1

        img = plt.subplots(squeeze=False, sharey=True, sharex=True, figsize=(ncols, nrows*0.6))

        for i in range(batch_size):
            batch_spikes = layer_spikes[i, :] / t
            row = i // ncols
            col = i % ncols
            ax = plt.subplot2grid((nrows, ncols), (row, col))
            ax.hist(batch_spikes, range=(0.0, 1.0), density=True, bins=20)
            ax.tick_params(top=False, bottom=False, left=False, right=False,
                           labelleft=False, labelbottom=False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf)

        sample.update(
            {
                f"layer {idx} example hist":
                    wandb.Image(im, caption=f"layer {idx} example hist")
            }
        )

        buf.close()
        plt.close()

        ncols = min(n, 8)
        nrows = n // ncols + 1
        if n % ncols == 0:
            nrows -= 1

        img = plt.subplots(squeeze=False, sharey=True, sharex=True, figsize=(ncols, nrows*0.6))

        for i in range(n):
            neuron_spikes = layer_spikes[:, i] / t
            row = i // ncols
            col = i % ncols
            ax = plt.subplot2grid((nrows, ncols), (row, col))
            ax.hist(neuron_spikes, range=(0.0, 1.0), density=True, bins=20)
            ax.tick_params(top=False, bottom=False, left=False, right=False,
                              labelleft=False, labelbottom=False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf)
        im.show()

        sample.update(
            {
                f"layer {idx} neuron hist":
                    wandb.Image(im, caption=f"layer {idx} neuron hist")
            }
        )

        buf.close()
        plt.close()

    return sample


class FilterSampler(BaseSampler):
    def __init__(self, nrow=8, **kwargs):

        super().__init__(**kwargs)
        self.nrow = nrow

    def sample(self, **result):

        sample = dict()

        if "input weights" in result.keys():
            w_in = result["input weights"]
            sample.update(
                    plot_filters(w_in, name="input", nrow=self.nrow)
            )

        if "output weights" in result.keys():
            w_out = result["output weights"]
            sample.update(
                plot_filters(w_out, name="output", nrow=self.nrow)
            )

        return sample


def plot_filters(weights, name="input", nrow=8):

    sample = dict()

    grid = make_grid(weights, nrow=nrow, normalize=True, padding=1)
    rows = np.min((weights.shape[0] // nrow + 1, 64))
    img = plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.ioff()

    sample.update(
        {
            f"{name} layer filters":
                wandb.Image(img, caption=f"{name} filters")
        }
    )

    #plt.show()
    plt.close()

    return sample


class MatrixSampler(BaseSampler):
    def __init__(self, layers=[], **kwargs):

        super().__init__(**kwargs)
        self.layers = layers

    def sample(self, **result):

        sample = dict()

        try:
            sample.update(plot_activity_matrix(
                layers=self.layers,
                **result
            ))
        except:
            print("plot activity matrix failed")

        try:
            sample.update(plot_example_activity_correlation(
                layers=self.layers,
                **result
            ))
        except:
            print("plot example activity matrix failed")

        try:
            sample.update(plot_neuron_activity_correlation(
                layers=self.layers,
                **result
            ))
        except:
            print("plot neuron activity matrix failed")

        return sample


def plot_example_activity_correlation(layers=[], **result):

    out_spikes = result["out_temps"]

    if "labels" in result.keys():
        label = True
        labels = result["labels"].cpu()

    sample = dict()

    for idx in layers:
        layer = out_spikes[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].view(batch_size, -1).size()[1]
        batch_spikes = (
            torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        )
        mean_batch_spikes = torch.mean(batch_spikes, dim=0)

        df = pd.DataFrame(mean_batch_spikes.T.numpy())

        # TODO: check for non-finite values !

        if label:
            label_pal = sns.husl_palette(len(torch.unique(labels)), s=0.45)
            label_lut = dict(zip(torch.unique(labels).numpy(), label_pal))
            label_colors = pd.Series(labels.numpy()).map(label_lut)
            img = sns.clustermap(
                df.corr(),
                col_colors=label_colors,
                xticklabels=labels.numpy(),
                yticklabels=labels.numpy(),
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu_r",
            )

        else:
            img = sns.clustermap(df.corr(), vmin=-1.0, vmax=1.0,cmap="RdBu_r",)

        sample.update(
            {
                f"example activity correlation layer {idx + 1}":
                    wandb.Image(img.fig, caption=f"L{idx + 1}")
            }
        )

        #plt.show()
        plt.close()

    return sample


def plot_neuron_activity_correlation(layers=[], **result):
    out_spikes = result["out_temps"]

    sample = dict()

    for idx in layers:
        layer = out_spikes[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].view(batch_size, -1).size()[1]
        batch_spikes = (
            torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        )
        mean_batch_spikes = torch.mean(batch_spikes, dim=0)

        df = pd.DataFrame(mean_batch_spikes.numpy())
        #print(df)
        #print(df.corr())
        img = sns.clustermap(df.corr().dropna(axis=0, how="all").dropna(axis=1, how="all"),
                             vmin=-1.0,
                             vmax=1.0,
                             cmap="RdBu_r",
                             )

        sample.update(
            {
                f"neuron activity correlation layer {idx + 1}":
                    wandb.Image(img.fig, caption=f"L{idx + 1}")
            }
        )

        #plt.show()
        plt.close()

    return sample


def plot_activity_matrix(layers=[], **result):

    out_spikes = result["out_temps"]

    if "labels" in result.keys():
        label = True
        labels = result["labels"].cpu()

    sample = dict()

    for idx in layers:
        layer = out_spikes[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].view(batch_size, -1).size()[1]
        batch_spikes = (
            torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        )
        mean_batch_spikes = torch.mean(batch_spikes, dim=0)

        df = pd.DataFrame(mean_batch_spikes.T.numpy())

        if label:
            label_pal = sns.husl_palette(len(torch.unique(labels)), s=0.45)
            label_lut = dict(zip(torch.unique(labels).numpy(), label_pal))
            label_colors = pd.Series(labels.numpy()).map(label_lut)
            img = sns.clustermap(
                df,
                col_colors=label_colors,
                xticklabels=labels.numpy(),
                vmin=0.0,
                vmax=1.0,
            )

        else:
            img = sns.clustermap(df, vmin=0.0, vmax=1.0)

        sample.update(
            {
                f"activity matrix layer {idx + 1}":
                    wandb.Image(img.fig, caption=f"L{idx + 1}")
            }
        )

        ##plt.show()
        plt.close()

    return sample


class SpikeSampler(BaseSampler):
    def __init__(self, layers=[2], plot_small=True, plot_large=True, **kwargs):

        super().__init__(**kwargs)
        self.plot_small = plot_small
        self.plot_large = plot_large
        self.layers = layers

    def sample(self, **result):

        sample = dict()
        if self.plot_small:
            sample.update(
                plot_output_spikes(
                    layers=self.layers,
                    max_samples=self.max_samples,
                    small=True,
                    **result,
                )
            )
        if self.plot_large:
            sample.update(
                plot_output_spikes(
                    layers=self.layers,
                    max_samples=self.max_samples,
                    small=False,
                    **result,
                )
            )

        return sample


def plot_output_spikes(layers=[], max_samples=20, small=True, **result):
    """

    :param result:
    :return:
    """

    out_spikes = result["out_temps"]
    sample = dict()

    for idx in layers:
        layer = out_spikes[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].view(batch_size, -1).size()[1]
        batch_spikes = (
            torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        )
        images = []
        for example in range(min([batch_size, max_samples])):
            example_spikes = batch_spikes[:, example, :].T
            active = np.count_nonzero(example_spikes.sum(dim=1) >= 1)
            spike_indices = [np.where(i)[0] for i in example_spikes]
            fig = plt.figure(figsize=(2.5, 2) if small else (5, 4))
            ax = fig.add_subplot(111)
            plt.eventplot(spike_indices, colors="black")
            plt.xlim(0, timesteps)
            flag = ""
            if not small:
                plt.xlabel("Timestep")
                plt.ylabel("Neuron #")
                plt.title(
                    f"Spiking activity layer {idx+1} - {active} / {neurons} active"
                )
            else:
                # ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                flag = "small "
            # #plt.show()
            images.append(fig)
            plt.close()

        sample.update(
            {
                f"{flag}spikes layer {idx+1}": [
                    wandb.Image(img, caption=f"L{idx+1} Example {i+1}")
                    for i, img in enumerate(images)
                ]
            }
        )

    return sample


class ReconstructionSampler(BaseSampler):
    def __init__(self, cbar=True, **kwargs):

        self.cbar = cbar
        super().__init__(**kwargs)

    def sample(self, **result):

        sample = dict()
        sample.update(
            plot_reconstruction(cbar=self.cbar, max_samples=self.max_samples, **result)
        )

        return sample


def plot_reconstruction(cbar=True, max_samples=20, **result):
    """

    """

    inputs = result["input"].data.cpu().numpy()
    originals = []

    outputs = result["output"].data.cpu().numpy()
    reconstructions = []

    if "input_history" in result.keys():
        input_history = result["input_history"]
        encodings = []
        encoded_inputs = torch.zeros(input_history[0].size())
        for frame in input_history:
            encoded_inputs += frame.cpu()

    batch_size = inputs.shape[0]

    for example in range(min([batch_size, max_samples])):

        # plot input
        fig = tensor_to_greyscale(inputs[example])
        originals.append(fig)

        # plot encoding
        if "input_history" in result.keys():
            fig = tensor_to_greyscale(encoded_inputs[example], cbar=cbar)
            encodings.append(fig)

        # plot reconstruction
        fig = tensor_to_greyscale(outputs[example], cbar=cbar)
        reconstructions.append(fig)

    sample = dict(
        original=[
            wandb.Image(img, caption=f"Input {i+1}") for i, img in enumerate(originals)
        ],
        reconstruction=[
            wandb.Image(img, caption=f"Output {i+1}")
            for i, img in enumerate(reconstructions)
        ],
    )

    if "input_history" in result.keys():
        sample.update(
            {
                "encoding": [
                    wandb.Image(img, caption=f"Encoding {i+1}")
                    for i, img in enumerate(encodings)
                ]
            }
        )

    return sample


def tensor_to_greyscale(tensor, cbar=True, vmin=None, vmax=None):

    fig = plt.figure(figsize=(2.5 if cbar else 2, 2))
    ax = fig.add_subplot(111)
    cax = ax.matshow(tensor[0], interpolation="nearest", vmin=vmin, vmax=vmax)

    if cbar:
        fig.colorbar(cax)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.close()

    return fig


class PotentialSampler(BaseSampler):
    def __init__(self, layers=[2, 5], cumulative=True, **kwargs):

        super().__init__(**kwargs)
        self.layers = layers
        self.cumulative = cumulative

    def sample(self, **result):

        sample = dict()
        sample.update(
            plot_output_potential(
                layers=self.layers, max_samples=self.max_samples, **result
            )
        )
        if self.cumulative:
            sample.update(
                plot_cummulative_potential(
                    layers=self.layers, max_samples=self.max_samples, **result
                )
            )

        return sample


def plot_output_potential(layers=[], max_samples=20, **result):
    """

    :param result:
    :return:
    """

    potential_history = result["potential_history"]
    sample = dict()

    for idx in layers:
        layer = potential_history[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].view(batch_size, -1).size()[1]
        batch_history = (
            torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        )
        images = []
        for example in range(min([batch_size, max_samples])):
            example_history = batch_history[:, example, :]
            fig = plt.figure()
            plt.plot(example_history)
            plt.xlim(0, timesteps)
            plt.xlabel("Timestep")
            plt.ylabel("Membrane Potential")
            plt.title("Membrane Potential History")
            images.append(fig)
            plt.close()

        sample.update(
            {
                f"membrane potential layer {idx+1}": [
                    wandb.Image(img, caption=f"Example {i+1}")
                    for i, img in enumerate(images)
                ]
            }
        )

    return sample


def plot_cummulative_potential(layers=[], max_samples=20, **result):
    """

    :param result:
    :return:
    """

    potential_history = result["cum_potential_history"]
    sample = dict()

    for idx in layers:
        layer = potential_history[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].view(batch_size, -1).size()[1]
        batch_history = (
            torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        )
        images = []
        for example in range(min([batch_size, max_samples])):
            example_history = batch_history[:, example, :]
            fig = plt.figure()
            plt.plot(example_history)
            plt.xlim(0, timesteps)
            plt.xlabel("Timestep")
            plt.ylabel("Membrane Potential")
            plt.title("Cummulative Membrane Potential History")
            images.append(fig)
            plt.close()

        sample.update(
            {
                f"cummulative potential layer {idx+1}": [
                    wandb.Image(img, caption=f"Example {i+1}")
                    for i, img in enumerate(images)
                ]
            }
        )

    return sample


