import typing as t
from abc import (
    ABC,
    abstractstaticmethod,
)

from torch import Tensor as tt


class BaseCouplingTransform(ABC):
    """Base class for the invertible transforms"""

    def __init__(self) -> None:
        self.inverse = False

    def invert(self) -> None:
        self.inverse != self.inverse

    @staticmethod
    @abstractstaticmethod
    def forward(x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        raise NotImplementedError()

    @staticmethod
    @abstractstaticmethod
    def backward(x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        raise NotImplementedError()

    def __call__(self, x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        if self.inverse:
            return self.forward(x, theta, compute_log_jacobian)
        else:
            return self.backward(x, theta, compute_log_jacobian)
