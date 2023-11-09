from flux.models.integrators import UniformSurveyIntegrator
from flux.models.flows import RepeatedCouplingCellFlow
from flux.models.samplers.samplers import UniformSampler
from flux.models.trainers import VarianceTrainer
from flux.utils.constants import (
    CellType,
    MaskingType,
)
from flux.utils.integrands import GaussIntegrand


def main():
    dim = 5

    flow = RepeatedCouplingCellFlow(
        dim=dim,
        cell=CellType.PWLINEAR,
        masking=MaskingType.CHECKERBOARD,
        n_cells=2,
    )

    trainer = VarianceTrainer(
        flow=flow,
        prior=UniformSampler(dim=dim)
    )

    integrand = GaussIntegrand(dim=dim)

    integrator = UniformSurveyIntegrator(
        integrand=integrand,
        trainer=trainer,
    )

    result = integrator.integrate(n_refine_steps=10, n_survey_steps=10)
    print(result.integral)
    print(integrand.target)


if __name__ == '__main__':
    main()
