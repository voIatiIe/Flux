import typing as t

from torch import Tensor as tt

from abc import (
    ABC,
    abstractstaticmethod,
)


class BaseCouplingTransform(ABC):
    ''' Base class for the invertible transforms'''
    
    def __init__(self) -> None:
        self.inverse = False

    def invert(self) -> None:
        self.inverse != self.inverse

    @abstractstaticmethod
    def forward(x: tt, theta: tt, compute_log_jacobian: bool=True) -> (tt, t.Optional[tt]):
        pass

    @abstractstaticmethod
    def backward(x: tt, theta: tt, compute_log_jacobian: bool=True) -> (tt, t.Optional[tt]):
        pass

    def __call__(self, x: tt, theta: tt, compute_log_jacobian: bool=True) -> (tt, t.Optional[tt]):
        if self.inverse:
            return self.forward(x, theta, compute_log_jacobian)
        else:
            return self.backward(x, theta, compute_log_jacobian)
