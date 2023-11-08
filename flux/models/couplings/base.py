import enum
import typing as t
from abc import (
    ABC,
    abstractmethod,
)

import torch
from torch import Tensor as tt

from flux.models.trainables import BaseTrainable
from flux.models.transforms import (
    BaseCouplingTransform,
    PWLinearCouplingTransform,
    PWQuadraticCouplingTransform,
)


class Mode(enum.Enum):
    TRAIN = "training"
    SAMPLE = "sampling"


class BaseCouplingCell(ABC, torch.nn.Module):
    def __init__(self, *, dim: int) -> None:
        super().__init__()

        self.mode: Mode = Mode.TRAIN
        self.dim = dim

    @abstractmethod
    def flow(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_and_compute_jacobian(self, xj: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == Mode.TRAIN:
            assert x.shape[1] == self.dim + 1, f"Shape mismatch! Expected: (:, {self.dim+1})"
            return self.transform_and_compute_jacobian(x)

        elif self.mode == Mode.SAMPLE:
            assert x.shape[1] == self.dim, f"Shape mismatch! Expected: (:, {self.dim})"
            return self.flow(x)

        else:
            raise ValueError(f"Unknown flow mode {self.mode}")

    @abstractmethod
    def invert(self) -> None:
        pass

    @abstractmethod
    def is_inverted(self) -> bool:
        pass


class CouplingCell(BaseCouplingCell):
    def __init__(
        self,
        *,
        dim: int,
        transform: BaseCouplingTransform,
        mask: t.List[bool],
        trainable: BaseTrainable,
    ):
        super().__init__(dim=dim)

        self.transform = transform

        self.mask = mask + [False]
        self.mask_complement = [not el for el in mask] + [False]

        self.trainable = trainable

    def transform_and_compute_jacobian(self, xj: tt) -> tt:
        x_n = xj[..., self.mask]
        x_m = xj[..., self.mask_complement]
        log_jacobian = xj[..., -1]

        yj = torch.zeros_like(xj).to(xj.device)

        yj[..., self.mask] = x_n
        yj[..., self.mask_complement], log_jacobian_y = self.transform(
            x_m, self.trainable(x_n), compute_log_jacobian=True
        )
        yj[..., -1] = log_jacobian + log_jacobian_y

        return yj

    def flow(self, x: tt) -> tt:
        x_n = x[..., self.mask[:-1]]
        x_m = x[..., self.mask_complement[:-1]]

        y = torch.zeros_like(x).to(x.device)

        y[..., self.mask] = x_n
        y[..., self.mask_complement], _ = self.transform(x_m, self.T(x_n), compute_log_jacobian=False)

        return y

    def invert(self) -> None:
        self.transform.invert()

    def is_inverted(self) -> bool:
        return self.transform.inverse


class BasePWLinearCouplingCell(CouplingCell):
    def __init__(
        self,
        *,
        dim: int,
        mask: t.List[bool],
        trainable: BaseTrainable,
    ) -> None:
        transform = PWLinearCouplingTransform()
        super().__init__(
            dim=dim,
            transform=transform,
            mask=mask,
            trainable=trainable,
        )


class BasePWQuadraticCouplingCell(CouplingCell):
    def __init__(
        self,
        *,
        dim: int,
        mask: t.List[bool],
        trainable: BaseTrainable,
    ) -> None:
        transform = PWQuadraticCouplingTransform()
        super().__init__(
            dim=dim,
            transform=transform,
            mask=mask,
            trainable=trainable,
        )
