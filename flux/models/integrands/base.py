import typing as t

from abc import (
    ABC,
    abstractmethod,
)
from torch import Tensor


class BaseIntegrand(ABC):
    def __init__(self, *, dim: int) -> None:
        self.calls = 0
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2 and x.shape[1] == self.dim, f"Shape mismatch! Expected: (:, {self.dim})"

        self.calls += x.shape[0]

        return self.callable(x)

    def reset(self) -> None:
        self.calls = 0

    @abstractmethod
    def callable(self, x: Tensor) -> Tensor:
        pass
