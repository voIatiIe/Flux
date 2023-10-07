import torch
import typing as t

from flux.models.samplers.base import BaseSampler


class UniformSampler(BaseSampler):
    def __init__(
        self, *,
        dim,
        device: t.Optional[torch.device]=torch.device('cpu'),
    ) -> None:
        #TODO: Enable using dynamic integration domain

        lower = torch.tensor(0.0, dtype=torch.get_default_dtype()).to(device)
        upper = torch.tensor(1.0, dtype=torch.get_default_dtype()).to(device)

        prior = torch.distributions.Uniform(lower, upper)

        super().__init__(dim=dim, prior=prior)


class GaussianSampler(BaseSampler):
    def __init__(
        self, *,
        dim,
        device: t.Optional[torch.device]=torch.device('cpu'),
    ) -> None:
        #TODO: Enable using dynamic Gauss parameters

        sig = torch.tensor(1.0, dtype=torch.get_default_dtype()).to(device)
        mu = torch.tensor(0.5, dtype=torch.get_default_dtype()).to(device)

        prior = torch.distributions.normal.Normal(mu, sig)

        super().__init__(dim=dim, prior=prior)
