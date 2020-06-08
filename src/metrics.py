import numpy as np
import wandb

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
    outputs = result["output"].data.cpu().numpy()

    sample = dict(
        original=[wandb.Image(x, caption=f"Input {i}") for i, x in enumerate(inputs)],
        reconstruction=[wandb.Image(y, caption=f"Output {i}") for i, y in enumerate(outputs)],
    )

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

    return samplers


metrics = {"accuracy": accuracy}