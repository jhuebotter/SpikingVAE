import numpy as np
import torch
import io
from PIL import Image
from torchvision.utils import make_grid
import wandb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
from pathlib import Path

#mpl.use('Agg')
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


LAYERS = [2]
MAX_SAMPLES = 100
FORM = ""
DIR = ""

def get_samplers(sampler_names, scale=1.0):

    samplers = dict()

    if "plot_reconstruction" in sampler_names:
        samplers.update(
            {
                "plot_reconstruction": ReconstructionSampler(
                    cbar=True,
                    max_samples=MAX_SAMPLES,
                    form=FORM,
                    vmin=0.0,
                    vmax=1.2,
                    scale=scale,
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
                    form=FORM,
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
                    form=FORM,
                )
            }
        )
    if "plot_activity_matrix" in sampler_names:
        samplers.update(
            {
                "plot_activity_matrix": MatrixSampler(
                    layers=LAYERS,
                    max_samples=MAX_SAMPLES,
                    form=FORM,
                )
            }
        )
    if "plot_filters" in sampler_names:
        samplers.update(
            {
                "plot_filters": FilterSampler(
                    nrow=8,
                    max_samples=MAX_SAMPLES,
                    form=FORM,
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
                    form=FORM,
                )
            }
        )
    if "plot_encoding" in sampler_names:
        samplers.update(
            {
                "plot_encoding": PoissonEncodingSampler(
                    max_samples=MAX_SAMPLES,
                    form=FORM,
                )
            }
        )

    if "animate_reconstruction" in sampler_names:
        samplers.update(
            {
                "animate_reconstruction": ReconstructionAnimationSampler(
                    max_samples=MAX_SAMPLES,
                    layers=LAYERS,
                    form="mp4",
                )
            }
        )

    if "plot_pca" in sampler_names:
        samplers.update(
            {
                "plot_pca": PcaSampler(
                    max_samples=MAX_SAMPLES,
                    layers=LAYERS,
                    form=FORM,
                )
            }
        )

    return samplers


class BaseSampler:
    def __init__(self, **kwargs):
        self.max_samples = kwargs.pop("max_samples")
        self.form = kwargs.pop("form")

    def sample(self, **result):
        # to be implemented by child classes
        pass

    def __call__(self, **results):
        return self.sample(**results)


class PcaSampler(BaseSampler):
    def __init__(self, layers=[], **kwargs):

        super().__init__(**kwargs)
        self.layers = layers
        global PCA, StandardScaler
        from sklearn.decomposition import PCA, FastICA
        from sklearn.preprocessing import StandardScaler

    def sample(self, **result):
        sample = dict()
        # try:
        for b in [True, False]:
            if "out_temps" in result.keys():
                sample.update(
                    pca_spike_plot(result["out_temps"],
                             result["labels"],
                             self.layers,
                             self.form,
                             three_d=b)
                )
            if "latent" in result.keys():
                sample.update(
                    pca_z_plot(result["latent"],
                                   result["labels"],
                                   self.form,
                                   three_d=b)
                )

        # except:
        #    pass

        return sample


def pca_z_plot(z, labels, form="", three_d=True):

    sample = dict()

    x = z.cpu().numpy()

    labels = labels.cpu().numpy()
    dim = "3d" if three_d else "2d"

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if three_d:
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(x)
        pca_df = pd.DataFrame(data=principal_components,
                              columns=['principal component 1', 'principal component 2', 'principal component 3'])
        pca_df["label"] = labels
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = pca_df['principal component 1']
        y = pca_df['principal component 2']
        z = pca_df['principal component 3']
        scatter = ax.scatter(x, y, z, c=labels, label=labels, cmap="tab10")
        legend = ax.legend(*scatter.legend_elements(),
                           loc="lower left", title="Classes")
        ax.add_artist(legend)
    else:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        pca_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
        pca_df["label"] = labels
        sns.scatterplot(data=pca_df, x="principal component 1", y="principal component 2", hue="label", palette="tab10")

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"representation_PCA_layer_{dim}.{form}")
        plt.savefig(path, format=form)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)

    sample.update(
        {
            f"representation PCA layer {dim}":
                wandb.Image(im, caption=f"representation PCA layer {dim}")
        }
    )

    buf.close()
    plt.close()

    return sample


def pca_spike_plot(out_spikes, labels, layers=[], form="", three_d=True):

    sample = dict()

    labels = labels.cpu().numpy()
    dim = "3d" if three_d else "2d"

    for idx in layers:

        layer = out_spikes[idx]
        timesteps = len(layer)
        batch_size = layer[0].size()[0]
        neurons = layer[0].view(batch_size, -1).size()[1]
        batch_spikes = (
            torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        )
        mean_batch_spikes = torch.mean(batch_spikes, dim=0)
        x = mean_batch_spikes.numpy()

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        if three_d:
            pca = PCA(n_components=3)
            principal_components = pca.fit_transform(x)
            pca_df = pd.DataFrame(data=principal_components,
                                  columns=['principal component 1', 'principal component 2', 'principal component 3'])
            pca_df["label"] = labels
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = pca_df['principal component 1']
            y = pca_df['principal component 2']
            z = pca_df['principal component 3']
            scatter = ax.scatter(x, y, z, c=labels, label=labels, cmap="tab10")
            legend = ax.legend(*scatter.legend_elements(),
                                loc="lower left", title="Classes")
            ax.add_artist(legend)
        else:
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(x)
            pca_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
            pca_df["label"] = labels
            sns.scatterplot(data=pca_df, x="principal component 1", y="principal component 2", hue="label", palette="tab10")

        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"representation_PCA_layer_{idx}_{dim}.{form}")
            plt.savefig(path, format=form)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf)

        sample.update(
            {
                f"representation PCA layer {idx} {dim}":
                wandb.Image(im, caption=f"representation PCA layer {idx} {dim}")
            }
        )

        buf.close()
        plt.close()

    return sample


class ReconstructionAnimationSampler(BaseSampler):
    def __init__(self, layers=[], **kwargs):

        super().__init__(**kwargs)
        global Camera
        from celluloid import Camera
        self.layers = layers

    def sample(self, **result):

        sample = dict()
        #try:
        sample.update(
            reconstruction_animation(result["input_history"],
                                     result["target"],
                                     result["out_temps"],
                                     result["cum_potential_history"][-1],
                                     self.layers,
                                     self.form)
        )
        #except:
        #    pass

        return sample


def reconstruction_animation(input_history, target, out_spikes, output_potential_history, layers=[], form="gif", cbar=True, fz=15):

    sample = dict()

    t = len(input_history)

    output_history = torch.stack(output_potential_history)

    for layer in layers:

        layer_spikes = out_spikes[layer]
        batch_size = layer_spikes[0].size()[0]
        neurons = layer_spikes[0].view(batch_size, -1).size()[1]
        batch_spikes = (
            torch.stack(layer_spikes).detach().cpu().view(t, batch_size, neurons)
        )
        #output_spikes = out_spikes[-1]

        n = min(batch_size, MAX_SAMPLES)

        for example in range(n):

            example_history = output_history[:, example]

            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            dividers = [make_axes_locatable(axs[i]) for i in [0, 1, 2, 4]]  # , 5]]
            #divider = make_axes_locatable(axs[2])
            caxs = [d.append_axes('right', size='5%', pad=0.05) for d in dividers]
            #cax = divider.append_axes('right', size='5%', pad=0.05)
            #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
            camera = Camera(fig)
            example_target = np.squeeze(target[example].cpu().numpy())
            input_canvas = np.zeros(input_history[0][example].size())
            #output_canvas = np.copy(input_canvas)
            max_output = np.copy(input_canvas)
            input_now = np.copy(input_canvas)
            spike_indices = []
            vmax = 25

            for i in range(t+1):

                #fig.suptitle(f"t = {i}", fontsize=16)

                im0 = axs[0].matshow(example_target, interpolation="nearest", vmin=0, vmax=1)
                axs[0].set_title("Original", fontsize=fz)
                fig.colorbar(im0, cax=caxs[0])

                im1 = axs[1].matshow(np.squeeze(input_now), interpolation="nearest", vmin=0, vmax=1)
                axs[1].set_title("Network input at t", fontsize=fz)
                fig.colorbar(im1, cax=caxs[1])

                im2 = axs[2].matshow(np.squeeze(input_canvas), interpolation="nearest", vmin=0, vmax=vmax)
                axs[2].set_title("Summed input until t", fontsize=fz)
                fig.colorbar(im2, cax=caxs[2])

                axs[3].eventplot(spike_indices, colors="black")
                axs[3].set_title("Spiking latent representation", fontsize=fz)

                im4 = axs[4].matshow(np.squeeze(max_output), interpolation="nearest", vmin=0, vmax=vmax)
                axs[4].set_title("Network output potential", fontsize=fz)
                fig.colorbar(im4, cax=caxs[3])

                #im5 = axs[5].matshow(np.squeeze(output_canvas), interpolation="nearest", vmin=0, vmax=vmax)
                #axs[5].set_title("Network output spikes")
                #fig.colorbar(im5, cax=caxs[4])

                #ttl = axs[2].text(3.0, 3.0, f"t = {i}", horizontalalignment='left', verticalalignment='bottom')

                for n in [0, 1, 2, 4]:  # , 5]:
                    axs[n].axes.xaxis.set_visible(False)
                    axs[n].axes.yaxis.set_visible(False)

                plt.tight_layout()
                camera.snap()

                if i < t:
                    example_spikes = batch_spikes[:i + 1, example, :].T
                    example_output = example_history[:i + 1].view((i + 1, *example_history.size()[-2:])).cpu().numpy()
                    max_output = np.amax(example_output, axis=0)
                    spike_indices = [np.where(x)[0] for x in example_spikes]
                    input_now = input_history[i][example].cpu().numpy()
                    input_canvas += input_now
                    # output_now = output_spikes[i][example].cpu().numpy()
                    # output_canvas += output_now

            animation = camera.animate(interval=100)

            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"reconstruction_animation_e{example}_l{layer}.{form}")
            animation.save(path)
            print("reconstruction animation saved as", path)

            sample.update(
                {
                    f"reconstruction animation example {example} layer {layer}":
                        wandb.Video(str(path), caption=f"reconstruction animation example {example} layer {layer} ")
                }
            )

            plt.close()

    return sample



class PoissonEncodingSampler(BaseSampler):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def sample(self, **result):

        sample = dict()
        try:
            input_history = result["input_history"]
            sample.update(
                plot_poisson_encoding_hist(input_history, self.form)
            )
        except:
            pass

        return sample

def plot_poisson_encoding_hist(input_history, form=""):

    sample = dict()

    t = len(input_history)
    cum_spikes = np.zeros(input_history[0].size())

    for i in range(t):
        cum_spikes += input_history[i].cpu().numpy()
    m = int(max(cum_spikes.flatten()))
    plt.hist(cum_spikes.flatten() / t, range=(0, 1), density=True, bins=20, align="left")
    plt.ylim(0, 20)
    plt.tight_layout()

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"poisson_encoding_hist.{form}")
        plt.savefig(path, format=form)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)

    sample.update(
        {
            f"poisson encoding hist":
                wandb.Image(im, caption=f"poisson encoding hist")
        }
    )

    buf.close()
    #plt.show()
    plt.close()

    return sample


class SpikeHistSampler(BaseSampler):
    def __init__(self, ncol=8, layers=[], **kwargs):

        super().__init__(**kwargs)
        self.ncol = ncol
        self.layers = layers

    def sample(self, **result):

        sample = dict()

        try:
            spikes = result["total_outs"]
            t = len(result["out_temps"][0])
            sample.update(
                plot_spike_hist(spikes, layers=self.layers, ncol=self.ncol,
                                t=t, max_samples=self.max_samples, form=self.form)
            )
        except:
            pass

        try:
            z = result["latent"]
            sample.update(
                plot_z_hist(z, ncol=self.ncol, max_samples=self.max_samples, form=self.form)
            )
        except:
            pass

        return sample


def plot_z_hist(z, ncol=8, max_samples=100, form=""):

    sample = dict()

    layer_spikes = z.cpu().numpy()
    batch_size = z.shape[0]
    n = z.shape[1]
    max_z = np.max(layer_spikes)
    min_z = np.min(layer_spikes)

    ncols = min(batch_size, ncol)
    nrows = batch_size // ncols + 1
    if batch_size % ncols == 0:
        nrows -= 1

    img = plt.subplots(squeeze=False, sharey=True, sharex=True, figsize=(ncols, nrows * 0.6))

    for i in range(batch_size):
        batch_activity = layer_spikes[i, :]
        row = i // ncols
        col = i % ncols
        ax = plt.subplot2grid((nrows, ncols), (row, col))
        ax.hist(batch_activity, range=(min_z, max_z), density=False, bins=20)
        ax.set_ylim((0.0, len(batch_activity)))
        plt.xticks([0])
        ax.tick_params(top=False, bottom=True, left=False, right=False,
                       labelleft=False, labelbottom=False)

    plt.tight_layout()

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"z_example_hist.{form}")
        plt.savefig(path, format=form)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)

    sample.update(
        {
            f"z example hist":
                wandb.Image(im, caption=f"z example hist")
        }
    )

    buf.close()
    #plt.show()
    plt.close()

    m = n if n < max_samples else max_samples

    ncols = min(m, ncol)
    nrows = m // ncols + 1
    if m % ncols == 0:
        nrows -= 1

    img = plt.subplots(squeeze=False, sharey=True, sharex=True, figsize=(ncols, nrows * 0.6))

    for i in range(m):
        neuron_activity = layer_spikes[:, i]
        row = i // ncols
        col = i % ncols
        ax = plt.subplot2grid((nrows, ncols), (row, col))
        ax.hist(neuron_activity, range=(min_z, max_z), density=False, bins=20)
        ax.set_ylim((0.0, len(neuron_activity)))
        plt.xticks([0])
        ax.tick_params(top=False, bottom=True, left=False, right=False,
                       labelleft=False, labelbottom=False)

    plt.tight_layout()

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"z_neuron_hist.{form}")
        plt.savefig(path, format=form)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    # im.show()

    sample.update(
        {
            f"z neuron hist":
                wandb.Image(im, caption=f"z neuron hist")
        }
    )

    buf.close()
    #plt.show()
    plt.close()

    return sample



def plot_spike_hist(spikes, layers, ncol=8, t=100, max_samples=100, form=""):

    sample = dict()

    for idx in layers:
        batch_size = spikes[idx].size(0)
        layer_spikes = spikes[idx].view(batch_size, -1).cpu().numpy()
        n = layer_spikes.shape[1]

        #m = int(max(layer_spikes.flatten()))
        plt.hist(layer_spikes.flatten() / t, range=(0, 1), density=True, bins=20, align="left")
        plt.ylim(0, 20)
        plt.tight_layout()

        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"spike_total_hist_l{idx+1}.{form}")
            plt.savefig(path, format=form)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf)

        sample.update(
            {
                f"layer {idx} total hist":
                    wandb.Image(im, caption=f"layer {idx} total hist")
            }
        )

        buf.close()
        # plt.show()
        plt.close()


        ncols = min(batch_size, ncol)
        nrows = batch_size // ncols + 1
        if batch_size % ncols == 0:
            nrows -= 1

        img = plt.subplots(squeeze=False, sharey=True, sharex=True, figsize=(ncols, nrows*0.6))

        for i in range(batch_size):
            batch_spikes = layer_spikes[i, :] / t
            row = i // ncols
            col = i % ncols
            ax = plt.subplot2grid((nrows, ncols), (row, col))
            ax.hist(batch_spikes, range=(0.0, 1.0), density=False, bins=20)
            plt.xticks([0])
            ax.set_ylim((0.0, len(batch_spikes)))
            ax.tick_params(top=False, bottom=True, left=False, right=False,
                           labelleft=False, labelbottom=False)

        plt.tight_layout()

        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"spike_example_hist_l{idx+1}.{form}")
            plt.savefig(path, format=form)

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


        m = n if n < max_samples else max_samples

        ncols = min(m, ncol)
        nrows = m // ncols + 1
        if m % ncols == 0:
            nrows -= 1

        img = plt.subplots(squeeze=False, sharey=True, sharex=True, figsize=(ncols, nrows*0.6))

        for i in range(m):
            neuron_spikes = layer_spikes[:, i] / t
            row = i // ncols
            col = i % ncols
            ax = plt.subplot2grid((nrows, ncols), (row, col))
            ax.hist(neuron_spikes, range=(0.0, 1.0), density=False, bins=20)
            ax.set_ylim((0.0, len(neuron_spikes)))
            plt.xticks([0])
            ax.tick_params(top=False, bottom=True, left=False, right=False,
                              labelleft=False, labelbottom=False)

        plt.tight_layout()

        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"spike_neuron_hist_l{idx+1}.{form}")
            plt.savefig(path, format=form)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf)
        #im.show()

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
                    plot_filters(w_in, name="input", nrow=self.nrow, form=self.form)
            )

        if "output weights" in result.keys():
            w_out = result["output weights"]
            sample.update(
                plot_filters(w_out, name="output", nrow=self.nrow, form=self.form)
            )

        return sample


def plot_filters(weights, name="input", nrow=8, form=""):

    sample = dict()

    grid = make_grid(weights, nrow=nrow, normalize=True, padding=1)
    rows = np.min((weights.shape[0] // nrow + 1, 64))
    img = plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.ioff()

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"{name}_filters.{form}")
        plt.savefig(path, format=form)

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
            if "out_temps" in result.keys():
                sample.update(plot_spiking_activity_matrix(
                    layers=self.layers, form=self.form,
                    **result
                ))
            if "latent" in result.keys():
                sample.update(plot_activity_matrix(form=self.form,
                    **result
                ))
        except:
            print("plot activity matrix failed")

        try:
            if "out_temps" in result.keys():
                sample.update(plot_spiking_example_activity_correlation(
                    layers=self.layers, form=self.form,
                    **result
                ))
            if "latent" in result.keys():
                sample.update(plot_example_activity_correlation(form=self.form,
                    **result
                ))
        except:
            print("plot example activity matrix failed")

        try:
            if "out_temps" in result.keys():
                sample.update(plot_spiking_neuron_activity_correlation(
                    layers=self.layers, form=self.form,
                    **result
                ))
            if "latent" in result.keys():
                sample.update(plot_neuron_activity_correlation(
                    form=self.form, **result
                ))
        except:
            print("plot neuron activity matrix failed")

        return sample


def plot_example_activity_correlation(form="", **result):

    if "labels" in result.keys():
        label = True
        labels = result["labels"].cpu()

    z = result["latent"].detach().cpu().numpy()
    z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)
    z[-1, :] += 0.001
    z[:-1, -1] += 0.001

    df = pd.DataFrame(z.T)

    if label:
        label_pal = sns.color_palette("tab10")
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
            center=0.0,
        )

    else:
        img = sns.clustermap(df.corr(), #.dropna(axis=0, how="all").dropna(axis=1, how="all"),
                             vmin=-1.0,
                             vmax=1.0,
                             cmap="RdBu_r",
                             center=0.0
                             )

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"example_activity_corr.{form}")
        plt.savefig(path, format=form)

    sample = dict()

    sample.update(
        {
            "example activity correlation z":
                wandb.Image(img.fig, caption="z correlation")
        }
    )

    #plt.show()
    plt.close()

    return sample


def plot_spiking_example_activity_correlation(layers=[], form="", **result):

    #out_spikes = result["out_temps"]

    if "labels" in result.keys():
        label = True
        labels = result["labels"].cpu()

    sample = dict()

    t = len(result["cum_potential_history"][0])

    for idx in layers:
        z = result["total_outs"][idx].detach().cpu().numpy()
        z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)
        z[-1, :] += 0.001
        z[:-1, -1] += 0.001
        #layer = out_spikes[idx]
        #timesteps = len(layer)
        #batch_size = z.shape[0]
        #neurons = layer[0].view(batch_size, -1).size()[1]
        #batch_spikes = (
        #    torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        #)
        #mean_batch_spikes = torch.mean(batch_spikes, dim=0)

        df = pd.DataFrame(z.T / t)

        # TODO: check for non-finite values !

        if label:
            label_pal = sns.color_palette("tab10")
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
                center=0.0,
            )

        else:
            img = sns.clustermap(df.corr(), vmin=-1.0, vmax=1.0, cmap="RdBu_r", center=0.0)

        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"spiking_example_activity_corr_l{idx+1}.{form}")
            plt.savefig(path, format=form)

        sample.update(
            {
                f"example activity correlation layer {idx + 1}":
                    wandb.Image(img.fig, caption=f"L{idx + 1}")
            }
        )

        #plt.show()
        plt.close()

    return sample


def plot_neuron_activity_correlation(form="", **result):

    z = result["latent"].detach().cpu().numpy()
    z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)
    z[-1, :] += 0.001
    z[:-1, -1] += 0.001

    sample = dict()

    df = pd.DataFrame(z)

    img = sns.clustermap(df.corr(),  #.dropna(axis=0, how="all").dropna(axis=1, how="all"),
                         vmin=-1.0,
                         vmax=1.0,
                         cmap="RdBu_r",
                         center=0.0
                         )

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"neuron_activity_corr.{form}")
        plt.savefig(path, format=form)

    sample.update(
        {
            "neuron activity correlation z":
                wandb.Image(img.fig, caption="z correlation")
        }
    )

    #plt.show()
    plt.close()

    return sample


def plot_spiking_neuron_activity_correlation(layers=[], form="", **result):
    #out_spikes = result["out_temps"]

    sample = dict()

    t = len(result["cum_potential_history"][0])

    for idx in layers:
        z = result["total_outs"][idx].detach().cpu().numpy()
        z = z[:, ~np.all(z == 0, axis=0)]  # drops all neurons that have 0 output for all examples (considered dead)
        z[-1, :] += 0.001
        z[:-1, -1] += 0.001
        # layer = out_spikes[idx]
        # timesteps = len(layer)
        # batch_size = z.shape[0]
        # neurons = layer[0].view(batch_size, -1).size()[1]
        # batch_spikes = (
        #    torch.stack(layer).detach().cpu().view(timesteps, batch_size, neurons)
        # )
        # mean_batch_spikes = torch.mean(batch_spikes, dim=0)

        df = pd.DataFrame(z / t)
        img = sns.clustermap(df.corr().dropna(axis=0, how="all").dropna(axis=1, how="all"),
                             vmin=-1.0,
                             vmax=1.0,
                             cmap="RdBu_r",
                             center=0.0
                             )

        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"spiking_neuron_activity_corr_l{idx+1}.{form}")
            plt.savefig(path, format=form)

        sample.update(
            {
                f"neuron activity correlation layer {idx + 1}":
                    wandb.Image(img.fig, caption=f"L{idx + 1}")
            }
        )

        #plt.show()
        plt.close()

    return sample


def plot_spiking_activity_matrix(layers=[], form="", **result):

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
            label_pal = sns.color_palette("tab10") #sns.husl_palette(len(torch.unique(labels)), s=0.45)
            label_lut = dict(zip(torch.unique(labels).numpy(), label_pal))
            label_colors = pd.Series(labels.numpy()).map(label_lut)
            img = sns.clustermap(
                df,
                col_colors=label_colors,
                xticklabels=labels.numpy(),
                vmin=0.0,
                vmax=1.0,
                cmap="RdBu_r",
                center=0.0,
            )

        else:
            img = sns.clustermap(df, vmin=0.0, vmax=1.0, cmap="RdBu_r", center=0.0)

        plt.setp(img.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)

        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"spiking_activity_matrix_l{idx+1}.{form}")
            plt.savefig(path, format=form)

        sample.update(
            {
                f"activity matrix layer {idx + 1}":
                    wandb.Image(img.fig, caption=f"L{idx + 1}")
            }
        )

        #plt.show()
        plt.close()

    return sample


def plot_activity_matrix(form="", **result):

    z = result["latent"].T.cpu().numpy()

    sample = dict()

    df = pd.DataFrame(z)
    #print(df.head(100))
    #print(np.sum(np.abs(z), axis=0).sum())
    #print(np.sum(np.abs(z), axis=1).sum())
    #print(np.sum(np.square(z), axis=0).sum())
    #print(np.sum(np.square(z), axis=1).sum())


    if "labels" in result.keys():
        labels = result["labels"].cpu()
        label_pal = sns.color_palette("tab10")
        label_lut = dict(zip(torch.unique(labels).numpy(), label_pal))
        label_colors = pd.Series(labels.numpy()).map(label_lut)
        img = sns.clustermap(
            df,
            col_colors=label_colors,
            xticklabels=labels.numpy(),
            center=0.0,
            cmap="RdBu_r",
        )

    else:
        img = sns.clustermap(df)

    plt.setp(img.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)

    if form:
        Path(DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DIR, f"activity_matrix.{form}")
        plt.savefig(path, format=form)

    sample.update(
        {
            f"activity matrix z": wandb.Image(img.fig, caption=f"z")
        }
    )

    #plt.show()
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
                    form=self.form,
                    **result,
                )
            )
        if self.plot_large:
            sample.update(
                plot_output_spikes(
                    layers=self.layers,
                    max_samples=self.max_samples,
                    small=False,
                    form=self.form,
                    **result,
                )
            )

        return sample


def plot_output_spikes(layers=[], max_samples=20, small=True, form="", **result):
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

            if form:
                Path(DIR).mkdir(parents=True, exist_ok=True)
                s = "s_" if small else ""
                path = Path(DIR, f"output_spikes_{s}e{example+1}_l{idx+1}.{form}")
                plt.savefig(path, format=form)

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
    def __init__(self, cbar=True, vmin=0.0, vmax=1.0, scale=1.0, **kwargs):

        self.cbar = cbar
        self.vmin = vmin
        self.vmax = vmax
        self.scale = scale
        super().__init__(**kwargs)

    def sample(self, **result):

        sample = dict()
        sample.update(
            plot_reconstruction(cbar=self.cbar, max_samples=self.max_samples,
                                form=self.form, vmin=self.vmin, vmax=self.vmax,
                                scale=self.scale, **result)
        )

        return sample


def plot_reconstruction(cbar=True, max_samples=20, form="", vmin=None, vmax=None, scale=1.0, **result):
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

        encoded_inputs = torch.div(encoded_inputs, len(input_history) * scale)

    batch_size = inputs.shape[0]

    for example in range(min([batch_size, max_samples])):

        # plot input
        fig = tensor_to_greyscale(inputs[example], cbar=cbar, vmin=vmin, vmax=vmax)
        if form:
            Path(DIR).mkdir(parents=True, exist_ok=True)
            path = Path(DIR, f"original_e{example+1}.{form}")
            print(f"saving input as {path}")
            plt.savefig(path, format=form)
        originals.append(fig)
        plt.close()

        # plot encoding
        if "input_history" in result.keys():
            fig = tensor_to_greyscale(encoded_inputs[example], cbar=cbar, vmin=vmin, vmax=vmax)
            if form:
                path = Path(DIR, f"encoding_e{example+1}.{form}")
                plt.savefig(path, format=form)
            encodings.append(fig)
            plt.close()

        # plot reconstruction
        fig = tensor_to_greyscale(outputs[example], cbar=cbar, vmin=vmin, vmax=vmax)
        if form:
            path = Path(DIR, f"reconstruction_e{example+1}.{form}")
            plt.savefig(path, format=form)
        reconstructions.append(fig)
        plt.close()

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
    cax = ax.matshow(tensor[0], interpolation="none", vmin=vmin, vmax=vmax)

    if cbar:
        fig.colorbar(cax)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()

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
                layers=self.layers, max_samples=self.max_samples, form=self.form, **result
            )
        )
        if self.cumulative:
            sample.update(
                plot_cummulative_potential(
                    layers=self.layers, max_samples=self.max_samples, form=self.form, **result
                )
            )

        return sample


def plot_output_potential(layers=[], max_samples=20, form="", **result):
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

            if form:
                Path(DIR).mkdir(parents=True, exist_ok=True)
                path = Path(DIR, f"output_potential_e{example+1}_l{idx+1}.{form}")
                plt.savefig(path, format=form)

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


def plot_cummulative_potential(layers=[], max_samples=20, form="", **result):
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

            if form:
                Path(DIR).mkdir(parents=True, exist_ok=True)
                path = Path(DIR, f"cum_potential_e{example+1}_l{idx+1}.{form}")
                plt.savefig(path, format=form)

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


