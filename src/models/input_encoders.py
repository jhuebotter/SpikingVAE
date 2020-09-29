import torch


class BaseEncoder:
    def __init__(self, **kwargs):

        self.input_history = []

        self.device = kwargs.pop("device")

    def reset(self):

        self.input_history = []

    def encode(self, x, **kwargs):

        return x


class PoissonSpikeEncoder(BaseEncoder):
    def __init__(self, noise=0.0, std=1.0, scale=0.2, **kwargs):
        super().__init__(**kwargs)

        self.noise = noise
        self.std = std
        self.scale = scale
        self.n = None

    def reset(self):

        self.input_history = []
        self.n = None

    def encode(self, x, t):

        rand_num = torch.rand(size=x.size()).to(self.device)

        if self.noise and self.n == None:
            #self.n = torch.normal(0, self.std, size=x.size()).to(self.device)
            self.n = torch.rand(size=x.size()).to(self.device)

        if self.noise:
            x = x * (1 - self.noise) + self.n * self.noise

        out = ((torch.abs(x * self.scale)) > rand_num).type(
            torch.FloatTensor if self.device == "cpu" else torch.cuda.FloatTensor
        )

        #out = torch.mul(Poisson_d_input, torch.sign(x))
        self.input_history.append(out.detach())

        return out


class FirstSpikeEncoder(BaseEncoder):
    def __init__(self, noise=0.0, std=1.0, threshold=1.0, scale=0.2, leak=0.99, **kwargs):
        super(FirstSpikeEncoder, self).__init__(**kwargs)

        self.noise = noise
        self.std = std
        self.threshold = threshold
        self.scale = scale
        self.leak = leak

    def encode(self, x, t):

        if self.input_history == []:
            self.has_fired = torch.zeros(size=x.size()).to(self.device)
            self.cum_x = torch.zeros(size=x.size()).to(self.device)

        if self.noise:
            noise_tensor = torch.normal(0, self.std, size=x.size()).to(self.device)
            x = x * (1 - self.noise) + noise_tensor * self.noise

        self.cum_x = self.cum_x * self.leak + x * self.scale

        out = (self.cum_x > self.threshold).type(
            torch.FloatTensor if self.device == "cpu" else torch.cuda.FloatTensor
        )
        #out = torch.mul(out, torch.sign(x))
        out = torch.mul(out, 1 - self.has_fired)
        self.has_fired = self.has_fired + out

        self.input_history.append(out.detach())

        return out


class PotentialEncoder(BaseEncoder):
    def __init__(self, noise=0.0, std=1.0, scaling=1.0, **kwargs):

        self.noise = noise
        self.std = std
        self.scaling = scaling
        super().__init__(**kwargs)


    def encode(self, x, t):

        if self.noise:
            noise_tensor = torch.normal(0, self.std, size=x.size()).to(self.device)
            x = x * (1 - self.noise) + noise_tensor * self.noise

        out = x * self.scaling

        self.input_history.append(out.detach())

        return out


class NoisyEncoder(BaseEncoder):
    def __init__(self, noise=0.0, std=1.0, scaling=1.0, **kwargs):

        self.noise = noise
        self.std = std
        self.scaling = scaling
        super().__init__(**kwargs)

    def encode(self, x):

        if self.noise:
            #noise_tensor = torch.normal(0, self.std, size=x.size()).to(self.device)
            noise_tensor = torch.rand(size=x.size()).to(self.device)
            x = x * (1 - self.noise) + noise_tensor * self.noise

        out = x * self.scaling

        self.input_history.append(out.detach())

        return out


def get_input_encoder(**kwargs):

    name = kwargs.pop("encoder").lower()

    if name == "potential":
        return PotentialEncoder(**kwargs)

    elif name == "spike":
        return PoissonSpikeEncoder(**kwargs)

    elif name == "first":
        return FirstSpikeEncoder(**kwargs)

    elif name == "noisy":
        return NoisyEncoder(**kwargs)

    else:
        raise NotImplementedError(
            f"The input encoder {name} is not implemented.\n"
            f"Valid options are: 'potential', 'spike', 'first', and 'noisy'."
        )