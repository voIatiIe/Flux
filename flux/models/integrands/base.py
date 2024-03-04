import torch
import typing as t
import numpy as np

from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from torch import Tensor


class BaseIntegrand(ABC):
    def __init__(self, *, dim: int) -> None:
        self.calls = 0
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.reshape(-1, self.dim))

        assert len(x.shape) == 2 and x.shape[1] == self.dim, f"Shape mismatch! Expected: (:, {self.dim})"

        self.calls += x.shape[0]

        return self.callable(x)

    def reset(self) -> None:
        self.calls = 0

    @abstractmethod
    def callable(self, x: Tensor) -> Tensor:
        pass

    @abstractproperty
    def target(self) -> float:
        pass
