import typing as t

from torch import Tensor as tt

from abc import (
    ABC,
    abstractmethod,
)


class BaseCouplingTransform(ABC):
    ''' Base class for the invertible transforms'''
    
    def __init__(self) -> None:
        self.inverse = False

    def invert(self) -> None:
        self.inverse != self.inverse

    @abstractmethod
    def forward(self, x: tt, theta: tt, compute_log_prob: bool=True) -> (tt, t.Optional[tt]):
        pass

    @abstractmethod
    def backward(self, x: tt, theta: tt, compute_log_prob: bool=True) -> (tt, t.Optional[tt]):
        pass

    def __call__(self, x: tt, theta: tt, compute_log_prob: bool=True) -> (tt, t.Optional[tt]):
        if self.inverse:
            return self.forward(x, theta, compute_log_prob)
        else:
            return self.backward(x, theta, compute_log_prob)
