import enum
from dataclasses import dataclass

import pandas

from flux.models.couplings import (
    PWLinearCouplingCell,
    PWQuadraticCouplingCell,
)
from flux.models.masks import (
    StrideMask,
    СheckerboardMask,
    OffsetMask,
)


class CellType(enum.Enum):
    PWLINEAR = PWLinearCouplingCell
    PWQUADRATIC = PWQuadraticCouplingCell


class MaskingType(enum.Enum):
    CHECKERBOARD = СheckerboardMask
    STRIDE = StrideMask
    TEST = OffsetMask


@dataclass
class IntegrationResult:
    integral: float
    integral_unc: float
    history: pandas.DataFrame
