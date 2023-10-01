from flux.models.integrand import BaseIntegrand
from flux.integration.base_integrator import BaseIntegrator
from flux.training.trainer import BaseTrainer


def func(x):
    return x[:,0] + x[:,1]


integrand = BaseIntegrand(
    dim=2,
    callable_=func,
    target=1
)

# b = BaseIntegrator(
#     integrand=integrand,
#     trainer=BaseTrainer(),
# )
