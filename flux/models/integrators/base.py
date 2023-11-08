import logging
from abc import (
    ABC,
    abstractmethod,
)

import torch
from better_abc import abstract_attribute

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
    def initialize(self, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def initialize_survey(self, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def initialize_refine(self, **kwargs) -> None:
        raise NotImplementedError()

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
    def finalize_survey(self, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def finalize_refine(self, **kwargs) -> None:
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
        self.logger.info("Initializing the survey phase")

        trainer_kwargs = kwargs.get("trainer_kwargs", {})
        # self.trainer.set_config(trainer_kwargs)

        survey_step_kwargs = kwargs.get("survey_step_kwargs", {})
        survey_initialize_kwargs = kwargs.get("survey_initialize_kwargs", {})
        survey_finalize_kwargs = kwargs.get("survey_finalize_kwargs", {})

        self.initialize_survey(**survey_initialize_kwargs)

        self.logger.info("Starting the survey phase")

        for step in range(n_steps):
            self.survey_step(**survey_step_kwargs)

        self.logger.info("Finalizing the survey phase")

        self.finalize_survey(**survey_finalize_kwargs)

    def refine(self, *, n_steps=10, **kwargs) -> None:
        self.logger.info("Initializing the refine phase")

        # TODO: Why is this needed?
        trainer_kwargs = kwargs.get("trainer_kwargs", {})
        # self.trainer.set_config(trainer_kwargs)
        #

        refine_step_kwargs = kwargs.get("refine_step_kwargs", {})
        refine_initialize_kwargs = kwargs.get("refine_initialize_kwargs", {})
        refine_finalize_kwargs = kwargs.get("refine_finalize_kwargs", {})

        self.initialize_refine(**refine_initialize_kwargs)

        self.logger.info("Starting the refine phase")

        for step in range(n_steps):
            self.refine_step(**refine_step_kwargs)

        self.logger.info("Finalizing the refine phase")

        self.finalize_refine(**refine_finalize_kwargs)

    def integrate(self, *, n_survey_steps=10, n_refine_steps=10, **kwargs) -> IntegrationResult:
        self.logger.info("Starting integration")

        initialize_kwargs = kwargs.get("initialize_kwargs", {})
        finalize_kwargs = kwargs.get("finalize_kwargs", {})
        survey_kwargs = kwargs.get("survey_kwargs", {})
        regine_kwargs = kwargs.get("regine_kwargs", {})

        self.logger.info("Initializing the integration")
        self.initialize(**initialize_kwargs)

        self.survey(n_steps=n_survey_steps, **survey_kwargs)
        self.refine(n_steps=n_refine_steps, **regine_kwargs)

        self.logger.info("Finalizing the integration")

        return self.finalize(**finalize_kwargs)
