import typing as t

from torch import Tensor as tt

from flux.models.transforms.base import BaseCouplingTransform


class PWQuadraticCouplingTransform(BaseCouplingTransform):
    #TODO: implement this
    def forward(x: tt, theta: tt, compute_log_jacobian: bool=True) -> (tt, t.Optional[tt]):
        raise NotImplementedError()

    def backward(x: tt, theta: tt, compute_log_jacobian: bool=True) -> (tt, t.Optional[tt]):
        raise NotImplementedError()
