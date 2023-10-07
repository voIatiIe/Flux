import pandas
import torch
import typing as t

from flux.models.integrators.base import BaseIntegrator
from flux.models.integrands.base import BaseIntegrand
from flux.models.samplers import BaseSampler
from flux.models.samplers.samplers import UniformSampler, GaussianSampler
from flux.models.trainers import BaseTrainer
from flux.utils.constants import IntegrationResult


class DefaultIntegrator(BaseIntegrator):
    def __init__(
        self, *,
        integrand: BaseIntegrand,
        trainer: BaseTrainer,
        n_iter: int = 10,
        n_iter_survey: t.Optional[int]=None,
        n_iter_refine: t.Optional[int]=None,
        n_points: t.Optional[int]=10000,
        n_points_survey: t.Optional[int]=None,
        n_points_refine: t.Optional[int]=None,
        use_survey: bool=False,
        verbosity: t.Optional[str]=None,
        **kwargs,
    ):
        super().__init__(verbosity=verbosity, **kwargs)
        self.integrand = integrand
        self.trainer = trainer

        self.n_iter_survey = n_iter_survey if n_iter_survey is not None else n_iter
        self.n_iter_refine = n_iter_refine if n_iter_refine is not None else n_iter

        self.n_points_survey = n_points_survey if n_points_survey is not None else n_points
        self.n_points_refine = n_points_refine if n_points_refine is not None else n_points

        self.use_survey = use_survey

    @staticmethod
    def get_empty_history():
        return pandas.DataFrame({
            'integral': pandas.Series([], dtype='float'),
            'uncertainty': pandas.Series([], dtype='float'),
            'n_points': pandas.Series([], dtype='int'),
            'phase': pandas.Series([], dtype='str')
        })

    def initialize(self, **kwargs):
        self.history = self.get_empty_history()

    def initialize_refine(self, **kwargs):
        pass

    def initialize_survey(self, **kwargs) -> None:
        pass

    def finalize_survey(self, **kwargs) -> None:
        pass

    def finalize_refine(self, **kwargs) -> None:
        pass

    # def sample_survey(self, **kwargs) -> (torch.Tensor, float, float):
    #     pass

    def sample_refine(self, *, n_points: t.Optional[int], **kwargs) -> (torch.Tensor, float, float):
        n_points = n_points if n_points is not None else self.n_points_refine

        #TODO: What is sample_forward?
        xj = self.trainer.sample_forward(n_points)
        x = xj[:, :-1]

        px = torch.exp(xj[:, -1])
        fx = self.integrand(x)

        return x, px, fx

    def process_survey_step(self, *, sample, integral, integral_var, train_result, **kwargs) -> None:
        x, _, _ = sample
        n_points = x.shape[0]
        integral_unc = (integral_var / n_points) ** 0.5

        self.history = self.history.append(
            {
                'integral': integral,
                'uncertainty': integral_unc,
                'n_points': n_points,
                'phase': 'survey',
                'train result': train_result,
            },
            ignore_index=True
        )
        self.logger.info(f"[SURVEY] Integral: {integral:.3e} +/- {integral_unc:.3e}")

    def process_refine_step(self, *, sample, integral, integral_var, **kwargs) -> None:
        x, _, _ = sample
        n_points = x.shape[0]
        integral_unc = (integral_var / n_points) ** 0.5
        self.history = self.history.append(
            {
                'integral': integral,
                'uncertainty': integral_unc,
                'n_points': n_points,
                'phase': 'survey',
                'train result': None,
            },
            ignore_index=True
        )
        self.logger.info(f"[REFINE] Integral: {integral:.3e} +/- {integral_unc:.3e}")

    def finalize(self, use_survey: bool=None, **kwargs) -> IntegrationResult:
        if use_survey is None:
            use_survey = self.use_survey

        data = self.integration_history
        if not use_survey:
            data = self.integration_history.loc[self.integration_history["phase"] == "refine"]

        #TODO: Derive correct formulas.
        integral = data["integral"].mean()
        integral_unc = 0.0
        #

        self.logger.info(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

        return IntegrationResult(
            integral=integral,
            integral_unc=integral_unc,
            history=self.history,
        )


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
        super().__init__(
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
        super().__init__(
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
        super().__init__(
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
