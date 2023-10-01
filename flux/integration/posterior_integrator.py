import torch
import typing as t

from flux.integration.base_integrator import BaseIntegrator
from flux.models.integrand import BaseIntegrand
from flux.training.sampler import BaseSampler, UniformSampler, GaussianSampler
from flux.training.trainer import BaseTrainer


class PosteriorSurveyIntegrator(BaseIntegrator):
    def __init__(
        self, *,
        integrand: BaseIntegrand,
        trainer: BaseTrainer,
        posterior: BaseSampler,
        n_iter: int = 10,
        n_iter_survey: t.Optional[int]=None,
        n_iter_refine: t.Optional[int]=None,
        n_points: t.Optional[int]=10000,
        n_points_survey: t.Optional[int]=None,
        n_points_refine: t.Optional[int]=None,
        use_survey: bool=False,
        verbosity: t.Optional[str]=None,
        **kwargs,
    ) -> None:
        super(PosteriorSurveyIntegrator, self).__init__(
            integrand=integrand,
            trainer=trainer,
            n_iter=n_iter,
            n_iter_survey=n_iter_survey,
            n_iter_refine=n_iter_refine,
            n_points=n_points,
            n_points_survey=n_points_survey,
            n_points_refine=n_points_refine,
            use_survey=use_survey,
            verbosity=verbosity,
            **kwargs,
        )
        self.posterior = posterior

    def sample_survey(self, *, n_points: t.Optional[int]=None, **kwargs) -> (torch.Tensor, float, float):
        n_points = n_points if n_points is not None else self.n_points_survey

        xj = self.posterior(n_points)
        x = xj[:, :-1]

        px = torch.exp(xj[:, -1])
        fx = self.integrand(x)

        return x, px, fx


class UniformSurveyIntegrator(PosteriorSurveyIntegrator):
    def __init__(
        self, *,
        integrand: BaseIntegrand,
        trainer: BaseTrainer,
        posterior: BaseSampler,
        n_iter: int = 10,
        n_iter_survey: t.Optional[int]=None,
        n_iter_refine: t.Optional[int]=None,
        n_points: t.Optional[int]=10000,
        n_points_survey: t.Optional[int]=None,
        n_points_refine: t.Optional[int]=None,
        use_survey: bool=False,
        verbosity: t.Optional[str]=None,
        device: torch.device=torch.device('cpu'),
        **kwargs,
    ) -> None:
        posterior = UniformSampler(
            dim=integrand.dim,
            device=device,
        )
        super(UniformSurveyIntegrator, self).__init__(
            integrand=integrand,
            trainer=trainer,
            posterior=posterior,
            n_iter=n_iter,
            n_iter_survey=n_iter_survey,
            n_iter_refine=n_iter_refine,
            n_points=n_points,
            n_points_survey=n_points_survey,
            n_points_refine=n_points_refine,
            use_survey=use_survey,
            verbosity=verbosity,
            **kwargs,
        )

class GaussianSurveyIntegrator(PosteriorSurveyIntegrator):
    def __init__(
        self, *,
        integrand: BaseIntegrand,
        trainer: BaseTrainer,
        posterior: BaseSampler,
        n_iter: int = 10,
        n_iter_survey: t.Optional[int]=None,
        n_iter_refine: t.Optional[int]=None,
        n_points: t.Optional[int]=10000,
        n_points_survey: t.Optional[int]=None,
        n_points_refine: t.Optional[int]=None,
        use_survey: bool=False,
        verbosity: t.Optional[str]=None,
        device: torch.device=torch.device('cpu'),
        **kwargs,
    ) -> None:
        posterior = GaussianSampler(
            dim=integrand.dim,
            device=device,
        )
        super(GaussianSurveyIntegrator, self).__init__(
            integrand=integrand,
            trainer=trainer,
            posterior=posterior,
            n_iter=n_iter,
            n_iter_survey=n_iter_survey,
            n_iter_refine=n_iter_refine,
            n_points=n_points,
            n_points_survey=n_points_survey,
            n_points_refine=n_points_refine,
            use_survey=use_survey,
            verbosity=verbosity,
            **kwargs,
        )
