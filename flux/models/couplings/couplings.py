import typing as t

import torch

from flux.models.couplings.base import (
    BasePWLinearCouplingCell,
    BasePWQuadraticCouplingCell,
)
from flux.models.trainables.trainable import DNNTrainable
from flux.utils.reshift import Reshift


class PWLinearCouplingCell(BasePWLinearCouplingCell):
    def __init__(
        self,
        *,
        dim: int,
        mask: t.List[bool],
        n_bins: int = 16,
        n_hidden: int = 3,
        dim_hidden: int = 32,
        hidden_activation: torch.nn.Module = torch.nn.LeakyReLU,
        input_activation: t.Optional[torch.nn.Module] = Reshift,
        output_activation: t.Optional[torch.nn.Module] = None,
    ) -> None:
        dim_in = sum(mask)
        dim_out = dim - dim_in
        out_shape = (dim_out, n_bins)

        trainable = DNNTrainable(
            dim_in=dim_in,
            out_shape=out_shape,
            n_hidden=n_hidden,
            dim_hidden=dim_hidden,
            hidden_activation=hidden_activation,
            input_activation=input_activation,
            output_activation=output_activation,
        )

        super().__init__(
            dim=dim,
            mask=mask,
            trainable=trainable,
        )


class PWQuadraticCouplingCell(BasePWQuadraticCouplingCell):
    def __init__(
        self,
        *,
        dim: int,
        mask: t.List[bool],
        n_bins: int = 32,
        n_hidden: int = 5,
        dim_hidden: int = 128,
        hidden_activation: torch.nn.Module = torch.nn.LeakyReLU,
        input_activation: t.Optional[torch.nn.Module] = Reshift,
        output_activation: t.Optional[torch.nn.Module] = None,
    ) -> None:
        dim_in = sum(mask)
        dim_out = dim - dim_in
        out_shape = (dim_out, 2 * n_bins + 1)

        trainable = DNNTrainable(
            dim_in=dim_in,
            out_shape=out_shape,
            n_hidden=n_hidden,
            dim_hidden=dim_hidden,
            hidden_activation=hidden_activation,
            input_activation=input_activation,
            output_activation=output_activation,
        )

        super().__init__(
            dim=dim,
            mask=mask,
            trainable=trainable,
        )
