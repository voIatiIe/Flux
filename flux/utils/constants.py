import enum
from dataclasses import dataclass
from datetime import timedelta

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
    survey_time: timedelta = timedelta()
    refine_time: timedelta = timedelta()


class Backend(enum.Enum):
    gloo = 'gloo'
    nlcc = 'nccl'
