import pandas
import torch
import torch.multiprocessing as mp
import typing as t
import numpy as np

import vegas

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from flux.models.integrands.base import BaseIntegrand
from flux.models.integrators.base import DefaultIntegrator
from flux.models.samplers import BaseSampler
from flux.models.samplers.samplers import UniformSampler, SobolSampler
from flux.models.trainers import BaseTrainer
from flux.utils.constants import IntegrationResult
from flux.utils.pgm import ProcessGroupManager


class PosteriorSurveyIntegrator(DefaultIntegrator):
    def __init__(
        self,
        *,
        integrand: BaseIntegrand,
        trainer: BaseTrainer,
        posterior: BaseSampler,
        n_iter: int = 10,
        n_iter_survey: t.Optional[int] = None,
        n_iter_refine: t.Optional[int] = None,
        n_points: t.Optional[int] = 10000,
        n_points_survey: t.Optional[int] = None,
        n_points_refine: t.Optional[int] = None,
        use_survey: bool = False,
        verbosity: t.Optional[str] = None,
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

    def sample_survey(self, *, n_points: t.Optional[int] = None, **kwargs) -> (torch.Tensor, float, float):
        n_points = n_points if n_points is not None else self.n_points_survey

        if self.trainer.sample_forward:
            xj = self.trainer.sample(n_points)
        else:
            xj = self.posterior(n_points)

        x = xj[:, :-1]

        px = torch.exp(-xj[:, -1])
        fx = self.integrand(x)

        return x, px, fx


class UniformSurveyIntegrator(PosteriorSurveyIntegrator):
    def __init__(
        self,
        *,
        integrand: BaseIntegrand,
        trainer: BaseTrainer,
        n_iter: int = 10,
        n_iter_survey: t.Optional[int] = None,
        n_iter_refine: t.Optional[int] = None,
        n_points: t.Optional[int] = 10000,
        n_points_survey: t.Optional[int] = None,
        n_points_refine: t.Optional[int] = None,
        use_survey: bool = False,
        verbosity: t.Optional[str] = None,
        device: torch.device = torch.device("cpu"),
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


class SobolSurveyIntegrator(PosteriorSurveyIntegrator):
    def __init__(
        self,
        *,
        integrand: BaseIntegrand,
        trainer: BaseTrainer,
        n_iter: int = 10,
        n_iter_survey: t.Optional[int] = None,
        n_iter_refine: t.Optional[int] = None,
        n_points: t.Optional[int] = 10000,
        n_points_survey: t.Optional[int] = None,
        n_points_refine: t.Optional[int] = None,
        use_survey: bool = False,
        verbosity: t.Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        posterior = SobolSampler(
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


class VegasIntegrator:
    def __init__(self, dim: int):
        self.dim = dim
        self.integrator = vegas.Integrator([[0, 1]] * dim)

        self.history = pandas.DataFrame({
            "integral": pandas.Series([], dtype="float"),
            "uncertainty": pandas.Series([], dtype="float"),
            "n_points": pandas.Series([], dtype="int"),
            "phase": pandas.Series([], dtype="str"),
        })

    def train(
        self,
        integrand: BaseIntegrand,
        n_train_steps: int = 10,
        n_train_samples: int = 10000,
    ):
        self.integrator(integrand, nitn=n_train_steps, neval=n_train_samples)

    def sample(
        self,
        n_samples: int,
        integrand: BaseIntegrand,
    ):
        self.integrator.set(neval=n_samples)
        x, wx = next(self.integrator.random_batch())

        x = np.asarray(x).copy()

        n_eval = x.shape[0]
        wx = np.asarray(wx).copy() * n_eval

        px = 1 / torch.tensor(wx, dtype=torch.float32)
        fx = integrand(x)
        x = torch.tensor(x, dtype=torch.float32)

        return x, px, fx

    def refine(
        self,
        integrand: BaseIntegrand,
        n_refine_steps: int = 10,
        n_refine_samples: int = 10000,
    ) -> pandas.DataFrame:
        for _ in range(n_refine_steps):
            x, px, fx = self.sample(n_samples=n_refine_samples, integrand=integrand)

            n_points = x.shape[0]

            integral_var, integral = torch.var_mean(fx / px)

            integral = integral.cpu().item()
            integral_var = integral_var.cpu().item()
            integral_unc = (integral_var / n_points) ** 0.5

            row = pandas.DataFrame([{
                "integral": integral,
                "uncertainty": integral_unc,
                "n_points": n_points,
                "phase": "refine",
            }])
            self.history = pandas.concat(
                [self.history, row],
                ignore_index=True,
            )

        return self.history
