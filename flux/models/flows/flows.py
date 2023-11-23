import math
import typing as t

from flux.models.flows.base import BaseRepeatedCouplingCellFlow
from flux.utils.constants import (
    CellType,
    MaskingType,
)


class RepeatedCouplingCellFlow(BaseRepeatedCouplingCellFlow):
    def __init__(
        self,
        *,
        dim: int,
        cell: CellType,
        n_cells: t.Optional[int] = None,
        masking: MaskingType,
        cell_parameters: t.Optional[t.Dict[str, t.Any]] = None,
        masking_parameters: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> None:
        assert dim > 1, "Dimension must be greater than one!"

        if n_cells is not None:
            assert n_cells > 1, "Number of cells must be greater than one!"
        else:
            if dim > 5:
                n_cells = math.ceil(math.log2(dim))
            else:
                n_cells = dim

        masking_parameters = masking_parameters or {}
        masks = masking.value(dim=dim, n_masks=n_cells, **masking_parameters)()

        super().__init__(
            dim=dim,
            masks=masks,
            cell=cell.value,
            cell_parameters=cell_parameters,
        )
