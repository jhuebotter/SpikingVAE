import torch

class BaseDecoder:
    def __init__(self, **kwargs):

        self.device = kwargs.pop("device")

    def reset(self):

        pass

    def decode(self, x, **kwargs):

        return x


class LastPotentialDecoder(BaseDecoder):
    def __init__(self, scaling=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling = scaling

    def decode(self, output_history):

        #output_history = cum_potential_history[-1]
        last_potential = output_history[-1]
        if self.scaling != 1.0:
            last_potential = torch.div(last_potential, self.scaling)

        return last_potential


class MaxPotentialDecoder(BaseDecoder):
    def __init__(self, scaling=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling = scaling

    def decode(self, output_history):

        #output_history = cum_potential_history[-1]
        output_history = torch.stack(output_history)
        max_potentials = torch.max(output_history, dim=0).values
        if self.scaling != 1.0:
            max_potentials = torch.div(max_potentials, self.scaling)

        return max_potentials


class MeanPotentialDecoder(BaseDecoder):
    def __init__(self, scaling=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling = scaling

    def decode(self, output_history):

        #output_history = cum_potential_history[-1]
        output_history = torch.stack(output_history)
        mean_potentials = torch.mean(output_history, dim=0) #.values
        if self.scaling != 1.0:
            mean_potentials = torch.div(mean_potentials, self.scaling)

        print(mean_potentials.size())


        return mean_potentials


def get_output_decoder(**kwargs):

    name = kwargs.pop("decoder").lower()

    if name == "last":
        return LastPotentialDecoder(**kwargs)

    if name == "max":
        return MaxPotentialDecoder(**kwargs)

    if name == "mean":
        return MeanPotentialDecoder(**kwargs)

    else:
        raise NotImplementedError(
            f"The output decoder {name} is not implemented.\n"
            f"Valid options are: 'last', 'max'."
        )