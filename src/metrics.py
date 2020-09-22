import numpy as np

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


def spike_density(**result):

    densities = []

    for i in range(len(result["total_outs"])):

        t = len(result["cum_potential_history"][i])
        batch_size = result["cum_potential_history"][i][-1].size(0)
        N = 1
        for j in [*result["cum_potential_history"][i][-1].size()][1:]:
            N *= j

        spikes = result["total_outs"][i]
        densities.append(spikes.sum() / (N * t * batch_size))

    return np.array(densities)



def get_metrics(metric_names):

    metrics = dict()

    if "accuracy" in metric_names:
        metrics.update({"accuracy": accuracy})
    if "spikedensity" in metric_names:
        metrics.update({"spike density": spike_density})

    return metrics



metrics = {"accuracy": accuracy}