import typing as t

import torch

from flux.models.couplings import (
    BaseCouplingCell,
    CouplingCell,
)


class BaseFlow(BaseCouplingCell):
    def __init__(
        self,
        *,
        dim: int,
        layers: t.List[CouplingCell],
    ) -> None:
        super().__init__(dim=dim)
        self._validate_layers(layers)

        self.layers = torch.nn.ModuleList(layers)

    def invert(self) -> None:
        for layer in self.layers:
            layer.invert()

        self.layers = self.layers[::-1]

    def is_inverted(self) -> bool:
        #TODO find out why doesnt work without not (something is messed up with it)
        return not all([l.is_inverted() for l in self.layers])

    def _validate_layers(self, layers: t.List[CouplingCell]) -> None:
        assert layers, "Empty flow! No layers found."
        for layer in layers:
            assert layer.mode == self.mode, "Mode mismatch!"

        inverted_ = [l.is_inverted() for l in layers]
        assert all(inverted_) or all([not el for el in inverted_]), "Layers directions out of sync!"

    def flow(self, x: torch.Tensor) -> torch.Tensor:
        output = x

        for l in self.layers:
            output = l.flow(output)

        return output

    def transform_and_compute_jacobian(self, xj: torch.Tensor) -> torch.Tensor:
        output = xj

        for l in self.layers:
            output = l.transform_and_compute_jacobian(output)

        return output


class BaseRepeatedCouplingCellFlow(BaseFlow):
    def __init__(
        self,
        *,
        dim: int,
        masks: t.List[t.List[bool]],
        cell: CouplingCell,
        cell_parameters: t.Optional[t.Dict[str, t.Any]],
    ) -> None:
        cell_parameters = cell_parameters or {}
        layers = [cell(dim=dim, mask=mask, **cell_parameters) for mask in masks]

        super().__init__(dim=dim, layers=layers)
