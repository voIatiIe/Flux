import math
import torch

from flux.models.integrands import BaseIntegrand
from flux.utils.long_integrand import long_integrand
from flux.utils.short_integrand import short_integrand


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
        return math.erf(self.mu / self.sigma) ** self.dim


class CamelIntegrand(BaseIntegrand):
    def __init__(
        self,
        *,
        dim: int,
        sigma1: float = 0.2,
        sigma2: float = 0.2,
    ) -> None:
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.gauss_1 = GaussIntegrand(dim=dim, mu=0.25, sigma=sigma1)
        self.gauss_2 = GaussIntegrand(dim=dim, mu=0.75, sigma=sigma2)

        super().__init__(dim=dim)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        return (self.gauss_1.callable(x) + self.gauss_2.callable(x)) / 2

    @property
    def target(self):
        return (0.5 * (math.erf(0.25 / self.sigma1) + math.erf(0.75 / self.sigma2))) ** self.dim


class SinusoideIntegrand(BaseIntegrand):
    def __init__(
        self,
        *,
        dim: int,
    ) -> None:
        assert dim < 5, "SinusoideIntegrand only supports dim < 5"

        self.targets = {
            1: 0.00518051998,
            2: -0.004403255,
            3: 0.00102106,
            4: -0.000368924,
        }

        super().__init__(dim=dim)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin((32.0 / self.dim) * x.sum(axis=1))

    @property
    def target(self):
        return self.targets[self.dim]


class HalfSinusoideIntegrand(BaseIntegrand):
    def __init__(
        self,
        *,
        dim: int,
    ) -> None:
        assert dim < 5, "HalfSinusoideIntegrand only supports dim < 4"

        self.targets = {
            1: 0.317248497,
            2: 0.31605,
            3: 0.318949,
        }

        super().__init__(dim=dim)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.sin((32.0 / self.dim) * x.sum(axis=1)), min=0.0)

    @property
    def target(self):
        return self.targets[self.dim]


class LongIntegrand(BaseIntegrand):
    def __init__(self) -> None:
        super().__init__(dim=7)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        return long_integrand(x)

    @property
    def target(self):
        return -0.07082419


class ShortIntegrand(BaseIntegrand):
    def __init__(self) -> None:
        super().__init__(dim=7)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        return short_integrand(x)

    @property
    def target(self):
        return -0.0247043


class CircteIntegrand(BaseIntegrand):
    def __init__(
        self,
        *,
        dim: int,
        radius1: float = 0.2,
        radius2: float = 0.45,
    ) -> None:
        assert radius2 > radius1

        self.dim = dim
        self.radius1 = radius1
        self.radius2 = radius2

        self.scale = math.pi**(self.dim / 2) / math.gamma(self.dim / 2 + 1)

        super().__init__(dim=dim)

    def callable(self, x: torch.Tensor) -> torch.Tensor:
        sqrt = ((x - 0.5)**2).sum(axis=1).sqrt()

        return torch.where(
            (self.radius1 < sqrt) & (sqrt < self.radius2),
            torch.ones_like(sqrt),
            torch.zeros_like(sqrt),
        )

    @property
    def target(self):
        return self.scale * (self.radius2**self.dim - self.radius1**self.dim)
