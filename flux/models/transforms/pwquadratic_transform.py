import typing as t

from torch import Tensor as tt

from flux.models.transforms.base import BaseCouplingTransform


class PWQuadraticCouplingTransform(BaseCouplingTransform):
    # TODO: implement this
    @staticmethod
    def forward(x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        raise NotImplementedError()

    @staticmethod
    def backward(x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        raise NotImplementedError()
