import numpy as np
import torch
import math
from scipy.spatial.distance import pdist, squareform

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


def encoding_density(**result):
    if "input_history" in result.keys():
        stacked_history = torch.stack(result["input_history"]).cpu().numpy()
        density = np.mean(stacked_history)
    else:
        density = np.nan

    return density


def within_euclidean(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()

        if "total_outs" in result.keys():

            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()
            batch_size = latent.shape[0]

        #for met in ["euclidean", "cityblock", "seuclidean", "sqeuclidean", "correlation"]:
        #print(met)

        distance = squareform(pdist(latent, metric="euclidean"))

        #print("distance matrix:", distance[0])
        #print("labels:", labels)

        within_class_distances = []
        outside_class_distances = []
        for l in np.unique(labels):
            #print("label:", l)
            idxs = np.argwhere(labels==l).reshape(-1)
            outside = np.array([x for x in np.arange(batch_size) if x not in idxs])
            #print("indexes:", idxs)
            #print("outside:", outside)

            for e, i in enumerate(idxs[:-1]):
                #print("index:", i)
                #print("within class distances:", distance[i][idxs][e+1:])
                #print("outside class distances:", distance[i][outside])
                within_class_distances += [d for d in distance[i][idxs][e+1:]]
                outside_class_distances += [d for d in distance[i][outside]]

        #print("mean within distance:", np.mean(within_class_distances), "+-", np.std(within_class_distances))
        #print("mean outside distance:", np.mean(outside_class_distances), "+-", np.std(outside_class_distances))

        return np.mean(within_class_distances)

    else:
        return np.nan


def outside_euclidean(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():
            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()
            batch_size = latent.shape[0]

        distance = squareform(pdist(latent, metric="euclidean"))

        outside_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)
            outside = np.array([x for x in np.arange(batch_size) if x not in idxs])
            for e, i in enumerate(idxs[:-1]):
                outside_class_distances += [d for d in distance[i][outside]]

        return np.mean(outside_class_distances)

    else:
        return np.nan


def within_seuclidean(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():

            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0).numpy()

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()

        V = np.var(latent, axis=0, ddof=1)
        V[V==0] = 0.0001
        distance = squareform(pdist(latent, metric="seuclidean", V=V))
        within_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)

            for e, i in enumerate(idxs[:-1]):
                within_class_distances += [d for d in distance[i][idxs][e + 1:] if not math.isnan(d)]

        return np.nanmean(within_class_distances)

    else:
        return np.nan


def outside_seuclidean(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():
            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0).numpy()

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()
            batch_size = latent.shape[0]

        V = np.var(latent, axis=0, ddof=1)
        V[V==0] = 0.0001
        distance = squareform(pdist(latent, metric="seuclidean", V=V))

        outside_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)
            outside = np.array([x for x in np.arange(batch_size) if x not in idxs])
            for e, i in enumerate(idxs[:-1]):
                outside_class_distances += [d for d in distance[i][outside] if not math.isnan(d)]

        return np.nanmean(outside_class_distances)

    else:
        return np.nan


def within_sqeuclidean(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():

            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()

        distance = squareform(pdist(latent, metric="sqeuclidean"))
        within_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)

            for e, i in enumerate(idxs[:-1]):
                within_class_distances += [d for d in distance[i][idxs][e + 1:]]

        return np.mean(within_class_distances)

    else:
        return np.nan


def outside_sqeuclidean(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():
            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()
            batch_size = latent.shape[0]

        distance = squareform(pdist(latent, metric="sqeuclidean"))

        outside_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)
            outside = np.array([x for x in np.arange(batch_size) if x not in idxs])
            for e, i in enumerate(idxs[:-1]):
                outside_class_distances += [d for d in distance[i][outside]]

        return np.mean(outside_class_distances)

    else:
        return np.nan


def within_manhattan(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():

            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()

        distance = squareform(pdist(latent, metric="cityblock"))
        within_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)

            for e, i in enumerate(idxs[:-1]):
                within_class_distances += [d for d in distance[i][idxs][e + 1:]]

        return np.mean(within_class_distances)

    else:
        return np.nan


def outside_manhattan(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():
            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()
            batch_size = latent.shape[0]

        distance = squareform(pdist(latent, metric="cityblock"))

        outside_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)
            outside = np.array([x for x in np.arange(batch_size) if x not in idxs])
            for e, i in enumerate(idxs[:-1]):
                outside_class_distances += [d for d in distance[i][outside]]

        return np.mean(outside_class_distances)

    else:
        return np.nan


def within_correlation(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():

            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()

        distance = squareform(pdist(latent, metric="correlation"))
        within_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)

            for e, i in enumerate(idxs[:-1]):
                within_class_distances += [d for d in distance[i][idxs][e + 1:]]

        return np.mean(within_class_distances)

    else:
        return np.nan


def outside_correlation(**result):
    if "labels" in result.keys():
        labels = result["labels"].cpu().numpy()
        if "total_outs" in result.keys():
            out_spikes = result["out_temps"]
            layer = out_spikes[LAYERS[0]]
            timesteps = len(layer)
            batch_size = layer[0].size()[0]
            neurons = layer[0].view(batch_size, -1).size()[1]
            batch_spikes = (
                torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
            )
            latent = torch.mean(batch_spikes, dim=0)

        elif "latent" in result.keys():
            latent = result["latent"].cpu().numpy()
            batch_size = latent.shape[0]

        distance = squareform(pdist(latent, metric="correlation"))

        outside_class_distances = []
        for l in np.unique(labels):
            idxs = np.argwhere(labels == l).reshape(-1)
            outside = np.array([x for x in np.arange(batch_size) if x not in idxs])
            for e, i in enumerate(idxs[:-1]):
                outside_class_distances += [d for d in distance[i][outside]]

        return np.mean(outside_class_distances)

    else:
        return np.nan


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
    if "encoding_density" in metric_names:
        metrics.update({"encoding density": encoding_density})
    if "latent_distances" in metric_names:
        metrics.update({
            "within class euclidean distance": within_euclidean,
            "outside class euclidean distance": outside_euclidean,
            "within class standardized euclidean distance": within_seuclidean,
            "outside class standardized euclidean distance": outside_seuclidean,
            "within class squared euclidean distance": within_sqeuclidean,
            "outside class squared euclidean distance": outside_sqeuclidean,
            "within class manhattan distance": within_manhattan,
            "outside class manhattan distance": outside_manhattan,
            "within class correlation distance": within_correlation,
            "outside class correlation distance": outside_correlation,
        })

    return metrics



metrics = {"accuracy": accuracy}
