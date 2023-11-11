import torch
import torch.multiprocessing as mp
import typing as t

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from flux.models.integrands.base import BaseIntegrand
from flux.models.integrators.base import DefaultIntegrator
from flux.models.samplers import BaseSampler
from flux.models.samplers.samplers import (
    GaussianSampler,
    UniformSampler,
)
from flux.models.trainers import BaseTrainer
from flux.utils.constants import IntegrationResult
from flux.utils.fsdb import ProcessGroupManager


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


class GaussianSurveyIntegrator(PosteriorSurveyIntegrator):
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


class FSDPUniformSurveyIntegrator(UniformSurveyIntegrator):
    model_name_prefix = "fsdp"

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
            device=device,
            **kwargs,
        )

    def _train(
        self,
        rank: int,
        world_size: int,
        n_train_steps: int = 10,
    ) -> IntegrationResult:

        with ProcessGroupManager(rank, world_size):
            torch.cuda.set_device(rank)
            self.trainer.flow = self.trainer.flow.to(rank)
            self.trainer.flow = FSDP(
                self.trainer.flow,
                auto_wrap_policy=size_based_auto_wrap_policy,
            )

            self.survey(n_steps=n_train_steps)
            states = self.trainer.flow.state_dict()
            torch.save(states, f'model_{rank}.pt')

    def train(self, *, n_train_steps=10, **kwargs) -> IntegrationResult:
        world_size = torch.cuda.device_count()
        print(f'Found {world_size} GPUs')

        mp.spawn(
            self._train,
            args=(world_size, n_train_steps),
            nprocs=world_size,
            join=True,
        )
