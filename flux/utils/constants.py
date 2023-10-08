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
)


class Mode(enum.StrEnum):
    TRAIN = "training"
    SAMPLE = "sampling"


class CellType(enum.Enum):
    PWLINEAR = PWLinearCouplingCell
    PWQUADRATIC = PWQuadraticCouplingCell


class MaskingType(enum.Enum):
    CHECKERBOARD = СheckerboardMask
    STRIDE = StrideMask


@dataclass
class IntegrationResult:
    integral: float
    integral_unc: float
    history: pandas.DataFrame
