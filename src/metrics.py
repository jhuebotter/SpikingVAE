import numpy as np

np.set_printoptions(threshold=np.inf)

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
    return np.sum(outputs == labels) / float(labels.size)


def mean_activity(**result):
    if "latent" in result.keys():
        z = result["latent"].detach().cpu().numpy()
    elif "total_outs" in result.keys():
        z = result["total_outs"][LAYERS[0]].detach().cpu().numpy()
    else:
        return 0

    z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)

    return np.mean(z)


def pct_active(**result):
    if "latent" in result.keys():
        z = result["latent"].detach().cpu().numpy()
    elif "total_outs" in result.keys():
        z = result["total_outs"][LAYERS[0]].detach().cpu().numpy()
    else:
        return 0

    n = z.shape[1]
    z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)
    n_ = z.shape[1]

    return n_ / n


def pct_active_per_example(**result):
    if "latent" in result.keys():
        z = result["latent"].detach().cpu().numpy()
    elif "total_outs" in result.keys():
        z = result["total_outs"][LAYERS[0]].detach().cpu().numpy()
    else:
        return 0

    batch_size = z.shape[0]
    z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)
    n_ = z.shape[1]
    if n_ == 0:
        return 0

    active = np.count_nonzero(z)

    return active / (batch_size * n_)


def example_activity_correlation(**result):
    if "latent" in result.keys():
        z = result["latent"].detach().cpu().numpy()
    elif "total_outs" in result.keys():
        z = result["total_outs"][LAYERS[0]].detach().cpu().numpy()
    else:
        return 0

    z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)

    if min(z.shape) < 2:
        print("not enough data for correlation")
        return 0.0

    z[-1, :] += 0.001
    z[:, -1] += 0.001

    c = np.abs(np.triu(np.corrcoef(z), 1))
    res = c.sum() / np.triu(np.ones(shape=c.shape)).sum()
    if np.isnan(res):
        return 0.0

    return res


def neuron_activity_correlation(**result):
    if "latent" in result.keys():
        z = result["latent"].detach().cpu().numpy()
    elif "total_outs" in result.keys():
        z = result["total_outs"][LAYERS[0]].detach().cpu().numpy()
    else:
        return 0

    z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)

    if min(z.shape) < 2:
        print("not enough data for correlation")
        return 0.0

    z[-1, :] += 0.0001
    z[:, -1] += 0.0001

    c = np.abs(np.triu(np.corrcoef(z, rowvar=False), 1))
    res = c.sum() / np.triu(np.ones(shape=c.shape)).sum()
    if np.isnan(res):
        return 0.0

    return res


def spike_density(**result):
    if "total_outs" in result.keys():
        t = len(result["cum_potential_history"][0])
        z = result["total_outs"][LAYERS[0]].detach().cpu().numpy()
        z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)
        batch_size = z.shape[0]
        N = 1
        for j in [*result["cum_potential_history"][LAYERS[0]][-1].size()][1:]:
            N *= j
        inactive = (z.sum(axis=0) == 0).sum(dtype=float) / N
        density = z.sum() / (batch_size * N * t * (1. - inactive))
    else:
        density = np.nan

    return density


def get_metrics(metric_names):
    metrics = dict()

    if "accuracy" in metric_names:
        metrics.update({"accuracy": accuracy})
    if "spikedensity" in metric_names:
        metrics.update({"spike density": spike_density})
    if "correlation" in metric_names:
        metrics.update({
            "neuron activity correlation": neuron_activity_correlation,
            "example activity correlation": example_activity_correlation,
        })
    if "meanactivity" in metric_names:
        metrics.update({"mean activity": mean_activity})
    if "pctactive" in metric_names:
        metrics.update({"latent pct active": pct_active})
    if "pctactiveperexample" in metric_names:
        metrics.update({"latent pct active per example": pct_active_per_example})

    return metrics


metrics = {"accuracy": accuracy}
