from flux.models.integrators import (
    UniformSurveyIntegrator,
    FSDPUniformSurveyIntegrator,
)
from flux.models.flows import RepeatedCouplingCellFlow
from flux.models.samplers.samplers import UniformSampler
from flux.models.trainers import VarianceTrainer
from flux.utils.constants import (
    CellType,
    MaskingType,
)
from flux.utils.integrands import GaussIntegrand
from flux.models.integrators.fsdp import integrate


def main_sequent():
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
        n_points=10000,
    )

    result = integrator.integrate(n_refine_steps=10, n_survey_steps=10)
    print(f'Result: {result.integral:.7e} +- {result.integral_unc:.7e}')
    print(f'Target: {integrand.target:.7e}')


def main_parallel():
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

    integrator = FSDPUniformSurveyIntegrator(
        integrand=integrand,
        trainer=trainer,
        n_points=10000,
    )

    integrator.train(n_refine_steps=10, n_survey_steps=10)


if __name__ == '__main__':
    integrand = GaussIntegrand(dim=5)
    integrate(integrand)
