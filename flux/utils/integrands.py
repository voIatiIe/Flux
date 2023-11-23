import math
import torch

from flux.models.integrands import BaseIntegrand
from flux.utils.long_integrand import long_integrand


class GaussIntegrand(BaseIntegrand):
    def __init__(
        self,
        *,
        dim: int,
        mu: float = 0.5,
        sigma: float = 0.1,
    ) -> None:
        self.mu = mu
        self.sigma = sigma

        super().__init__(dim=dim)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        return (self.sigma * math.pi ** 0.5) ** (-self.dim) * torch.exp(-((x - self.mu) / self.sigma).square().sum(axis=1))

    @property
    def target(self):
        return math.erf(0.5 / self.sigma) ** self.dim


class LongIntegrand(BaseIntegrand):
    def __init__(self) -> None:
        super().__init__(dim=7)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        return long_integrand(x)

    @property
    def target(self):
        return 0.0
