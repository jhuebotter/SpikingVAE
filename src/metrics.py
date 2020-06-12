import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

LAYERS = [2]
MAX_SAMPLES = 20

def accuracy(**result):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """

    outputs = result["output"].data.cpu().numpy()
    labels = result["target"].data.cpu().numpy()

    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


def plot_reconstruction(**result):
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

    for example in range(max([batch_size, MAX_SAMPLES])):

        # plot input
        fig = tensor_to_greyscale(inputs[example])
        originals.append(fig)

        # plot encoding
        if "input_history" in result.keys():
            fig = tensor_to_greyscale(encoded_inputs[example])
            encodings.append(fig)

        # plot reconstruction
        fig = tensor_to_greyscale(outputs[example])
        reconstructions.append(fig)

    sample = dict(
        original=[wandb.Image(img, caption=f"Input {i+1}") for i, img in enumerate(originals)],
        reconstruction=[wandb.Image(img, caption=f"Output {i+1}") for i, img in enumerate(reconstructions)],
    )

    if "input_history" in result.keys():
        sample.update({"encoding": [wandb.Image(img, caption=f"Encoding {i+1}") for i, img in enumerate(encodings)]})

    return sample


def tensor_to_greyscale(tensor, cbar=True):

    fig = plt.figure(figsize=(2.5 if cbar else 2, 2))
    ax = fig.add_subplot(111)
    cax = ax.matshow(tensor[0], interpolation='nearest')

    if cbar:
        fig.colorbar(cax)

    # plt.title("Original input")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.close()

    return fig



def plot_output_spikes(**result):
    """

    :param result:
    :return:
    """

    out_spikes = result["out_temps"]
    sample = dict()

    for idx in LAYERS:
        layer = out_spikes[idx]
        timesteps = len(layer)
        batch_size, neurons = layer[0].size()
        batch_spikes = torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        images = []
        for example in range(max([batch_size, MAX_SAMPLES])):
            example_spikes = batch_spikes[:, example, :].T
            active = np.count_nonzero(example_spikes.sum(dim=1)>=1)
            spike_indices = [np.where(i)[0] for i in example_spikes]
            fig = plt.figure()
            plt.eventplot(spike_indices, colors="black")
            plt.xlim(0, timesteps)
            plt.xlabel("Timestep")
            plt.ylabel("Neuron #")
            plt.title(f"Spiking activity layer {idx+1} - {active} / {neurons} active")
            images.append(fig)
            plt.close()

        sample.update({f"spikes layer {idx+1}": [wandb.Image(img, caption=f"Example {i+1}") for i, img in enumerate(images)]})

    return sample

def plot_output_potential(**result):
    """

    :param result:
    :return:
    """

    potential_history = result["potential_history"]
    sample = dict()

    for idx in LAYERS:
        layer = potential_history[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].size()[1]
        batch_history = torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        images = []
        for example in range(max([batch_size, MAX_SAMPLES])):
            example_history = batch_history[:, example, :]
            fig = plt.figure()
            plt.plot(example_history)
            plt.xlim(0, timesteps)
            plt.xlabel("Timestep")
            plt.ylabel("Membrane Potential")
            plt.title("Membrane Potential History")
            images.append(fig)
            plt.close()

        sample.update({f"membrane potential layer {idx+1}": [wandb.Image(img, caption=f"Example {i+1}") for i, img in enumerate(images)]})

    return sample



def plot_cummulative_potential(**result):
    """

    :param result:
    :return:
    """

    potential_history = result["cum_potential_history"]
    sample = dict()

    for idx in LAYERS:
        layer = potential_history[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].size()[1]
        batch_history = torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        images = []
        for example in range(max([batch_size, MAX_SAMPLES])):
            example_history = batch_history[:, example, :]
            fig = plt.figure()
            plt.plot(example_history)
            plt.xlim(0, timesteps)
            plt.xlabel("Timestep")
            plt.ylabel("Membrane Potential")
            plt.title("Cummulative Membrane Potential History")
            images.append(fig)
            plt.close()

        sample.update({f"cummulative potential layer {idx+1}": [wandb.Image(img, caption=f"Example {i+1}") for i, img in enumerate(images)]})

    return sample



def get_metrics(metric_names):

    metrics = dict()

    if "accuracy" in metric_names:
        metrics.update({"accuracy": accuracy})

    return metrics


def get_samplers(sampler_names):

    samplers = dict()

    if "plot_reconstruction" in sampler_names:
        samplers.update({"plot_reconstruction": plot_reconstruction})
    if "plot_output_spikes" in sampler_names:
        samplers.update({"plot_output_spikes": plot_output_spikes})
    if "plot_output_potential" in sampler_names:
        samplers.update({"plot_output_potential": plot_output_potential})
    if "plot_cummulative_potential" in sampler_names:
        samplers.update({"plot_cummulative_potential": plot_cummulative_potential})

    return samplers


metrics = {"accuracy": accuracy}