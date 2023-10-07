import typing as t

from torch import Tensor as tt

from flux.models.transforms.base import BaseCouplingTransform


class PWLinearCouplingTransform(BaseCouplingTransform):
    #TODO: implement this
    def forward(self, x: tt, theta: tt, compute_log_prob: bool=True) -> (tt, t.Optional[tt]):
        pass

    def backward(self, x: tt, theta: tt, compute_log_prob: bool=True) -> (tt, t.Optional[tt]):
        pass


class PWQuadraticCouplingTransform(BaseCouplingTransform):
    #TODO: implement this
    def forward(self, x: tt, theta: tt, compute_log_prob: bool=True) -> (tt, t.Optional[tt]):
        pass

    def backward(self, x: tt, theta: tt, compute_log_prob: bool=True) -> (tt, t.Optional[tt]):
        pass
