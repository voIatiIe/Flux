import typing as t

import torch

from flux.models.samplers.base import BaseSampler


class UniformSampler(BaseSampler):
    def __init__(
        self,
        *,
        dim,
        device: t.Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        lower = torch.tensor(0.0, dtype=torch.get_default_dtype()).to(device)
        upper = torch.tensor(1.0, dtype=torch.get_default_dtype()).to(device)

        prior = torch.distributions.Uniform(lower, upper)

        super().__init__(dim=dim, prior=prior)


class GaussianSampler(BaseSampler):
    def __init__(
        self,
        *,
        dim,
        device: t.Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        sig = torch.tensor(1.0, dtype=torch.get_default_dtype()).to(device)
        mu = torch.tensor(0.5, dtype=torch.get_default_dtype()).to(device)

        prior = torch.distributions.normal.Normal(mu, sig)

        super().__init__(dim=dim, prior=prior)


class SobolSampler(BaseSampler):
    def __init__(
        self,
        *,
        dim,
        device: t.Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        self.prior = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)

        super().__init__(dim=dim, prior=self.prior)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2 and x.shape[1] == self.dim, f"Shape mismatch! Expected: (:, {self.dim})"

        return torch.zeros(x.shape[0])

    def forward(self, n_points: int) -> torch.Tensor:
        x = self.prior.draw(n_points)
        log_j = -self.log_prob(x)

        return torch.cat([x, log_j.unsqueeze(-1)], -1)
