import enum
import pandas

from dataclasses import dataclass

from flux.models.couplings import (
    PWLinearCouplingCell,
    PWQuadraticCouplingCell,
)
from flux.models.masks import (
    СheckerboardMask,
    StrideMask,
)


class Mode(enum.StrEnum):
    TRAIN = 'training'
    SAMPLE = 'sampling'


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
