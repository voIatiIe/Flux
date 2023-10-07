import typing as t

from flux.utils.constants import (
    CellType,
    MaskingType,
)
from flux.models.flows.base import BaseRepeatedCouplingCellFlow


class RepeatedCouplingCellFlow(BaseRepeatedCouplingCellFlow):
    def __init__(
        self, *,
        dim: int,
        cell: CellType,
        n_cells: int,
        masking: MaskingType,
        cell_parameters: t.Optional[t.Dict[str, t.Any]] = None,
        masking_parameters: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> None:

        masking_parameters = masking_parameters or {}
        masks = masking.value(dim=dim, n_masks=n_cells, **masking_parameters)

        super().__init__(
            dim=dim,
            masks=masks,
            cell=cell.value,
            cell_parameters=cell_parameters,
        )
