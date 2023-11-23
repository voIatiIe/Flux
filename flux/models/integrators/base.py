import logging
import pandas
import typing as t
from abc import (
    ABC,
    abstractmethod,
)

import torch
from better_abc import abstract_attribute

from flux.models.integrands.base import BaseIntegrand
from flux.models.trainers import BaseTrainer
from flux.utils.constants import IntegrationResult
from flux.utils.logging import set_verbosity


class BaseIntegrator(ABC):
    set_verbosity = set_verbosity

    def __init__(self, *args, verbosity=None, **kwargs) -> None:
        self.trainer: BaseTrainer = abstract_attribute()
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__ + ":" + hex(id(self)))

        self.set_verbosity(verbosity)

    @abstractmethod
    def sample_survey(self, **kwargs) -> (torch.Tensor, float, float):
        raise NotImplementedError()

    @abstractmethod
    def sample_refine(self, **kwargs) -> (torch.Tensor, float, float):
        raise NotImplementedError()

    @abstractmethod
    def process_survey_step(self, *, sample, integral, integral_var, train_result, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def process_refine_step(self, *, sample, integral, integral_var, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, **kwargs) -> IntegrationResult:
        raise NotImplementedError()

    def survey_step(self, **kwargs) -> None:
        sampling_kwargs = kwargs.get("sampling_kwargs", {})
        training_args = kwargs.get("training_args", {})

        x, px, fx = self.sample_survey(**sampling_kwargs)
        integral_var, integral = torch.var_mean(fx / px)

        integral = integral.cpu().item()
        integral_var = integral_var.cpu().item()

        train_result = self.trainer.train_batch(x, px, fx, **training_args)

        self.process_survey_step(
            sample=(x, px, fx),
            integral=integral,
            integral_var=integral_var,
            train_result=train_result,
        )

    def refine_step(self, **kwargs) -> None:
        sampling_kwargs = kwargs.get("sampling_kwargs", {})

        x, px, fx = self.sample_refine(**sampling_kwargs)
        integral_var, integral = torch.var_mean(fx / px)

        integral = integral.cpu().item()
        integral_var = integral_var.cpu().item()

        self.process_refine_step(
            sample=(x, px, fx),
            integral=integral,
            integral_var=integral_var,
        )

    def survey(self, *, n_steps=10, **kwargs) -> None:
        survey_step_kwargs = kwargs.get("survey_step_kwargs", {})

        for step in range(n_steps):
            self.survey_step(**survey_step_kwargs)

    def refine(self, *, n_steps=10, **kwargs) -> None:
        refine_step_kwargs = kwargs.get("refine_step_kwargs", {})

        for step in range(n_steps):
            self.refine_step(**refine_step_kwargs)

    def integrate(self, *, n_survey_steps=10, n_refine_steps=10, **kwargs) -> IntegrationResult:
        self.logger.info("Starting integration")

        survey_kwargs = kwargs.get("survey_kwargs", {})
        regine_kwargs = kwargs.get("regine_kwargs", {})

        self.survey(n_steps=n_survey_steps, **survey_kwargs)
        self.refine(n_steps=n_refine_steps, **regine_kwargs)

        return self.finalize()


class DefaultIntegrator(BaseIntegrator):
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

        self.history = self.get_empty_history()

    @staticmethod
    def get_empty_history():
        return pandas.DataFrame(
            {
                "integral": pandas.Series([], dtype="float"),
                "uncertainty": pandas.Series([], dtype="float"),
                "n_points": pandas.Series([], dtype="int"),
                "phase": pandas.Series([], dtype="str"),
            }
        )

    def sample_refine(self, *, n_points: t.Optional[int] = None, **kwargs) -> (torch.Tensor, float, float):
        n_points = n_points if n_points is not None else self.n_points_refine

        xj = self.trainer.sample(n_points)
        x = xj[:, :-1]

        px = torch.exp(-xj[:, -1])
        fx = self.integrand(x)

        return x, px, fx

    def process_survey_step(self, *, sample, integral, integral_var, train_result, **kwargs) -> None:
        x, _, _ = sample
        n_points = x.shape[0]
        integral_unc = (integral_var / n_points) ** 0.5

        row = pandas.DataFrame([{
            "integral": integral,
            "uncertainty": integral_unc,
            "n_points": n_points,
            "phase": "survey",
            "train result": train_result,
        }])
        self.history = pandas.concat(
            [self.history, row],
            ignore_index=True,
        )
        self.logger.info(f"[SURVEY] Integral: {integral:.3e} +/- {integral_unc:.3e}")

    def process_refine_step(self, *, sample, integral, integral_var, **kwargs) -> None:
        x, _, _ = sample
        n_points = x.shape[0]
        integral_unc = (integral_var / n_points) ** 0.5

        row = pandas.DataFrame([{
                "integral": integral,
                "uncertainty": integral_unc,
                "n_points": n_points,
                "phase": "refine",
                "train result": None,
        }])
        self.history = pandas.concat(
            [self.history, row],
            ignore_index=True,
        )
        self.logger.info(f"[REFINE] Integral: {integral:.3e} +/- {integral_unc:.3e}")

    def finalize(self, use_survey: bool | None = None, **kwargs) -> IntegrationResult:
        if use_survey is None:
            use_survey = self.use_survey

        data = self.history

        if not use_survey:
            data = self.history.loc[self.history["phase"] == "refine"]

        variance = 1.0 / (1.0 / data["uncertainty"] ** 2).sum()

        integral = (data["integral"] / data["uncertainty"] ** 2).sum() * variance
        integral_unc = variance ** 0.5

        self.logger.info(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

        return IntegrationResult(
            integral=integral,
            integral_unc=integral_unc,
            history=self.history,
        )
