import logging
import pandas
import torch
import typing as t

from abc import ABC, abstractmethod
from better_abc import abstract_attribute
from dataclasses import dataclass

from flux.utils.logging import set_verbosity
from flux.models.integrand import BaseIntegrand
from flux.training.trainer import BaseTrainer


@dataclass
class IntegrationResult:
    integral: float
    integral_unc: float
    history: pandas.DataFrame


class IntegratorAPI(ABC):
    set_verbosity = set_verbosity

    def __init__(self, *args, verbosity=None, **kwargs) -> None:
        self.trainer = abstract_attribute()
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__ + ':' + hex(id(self)))

        self.set_verbosity(verbosity)

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        pass

    @abstractmethod
    def initialize_survey(self, **kwargs) -> None:
        pass

    @abstractmethod
    def initialize_refine(self, **kwargs) -> None:
        pass

    @abstractmethod
    def sample_survey(self, **kwargs) -> (torch.Tensor, float, float):
        pass

    @abstractmethod
    def sample_refine(self, **kwargs) -> (torch.Tensor, float, float):
        pass

    @abstractmethod
    def process_survey_step(self, *, sample, integral, integral_var, train_result, **kwargs) -> None:
        pass

    @abstractmethod
    def process_refine_step(self, *, sample, integral, integral_var, **kwargs) -> None:
        pass

    @abstractmethod
    def finalize_survey(self, **kwargs) -> None:
        pass

    @abstractmethod
    def finalize_refine(self, **kwargs) -> None:
        pass

    @abstractmethod
    def finalize(self, **kwargs) -> IntegrationResult:
        pass

    def survey_step(self, **kwargs) -> None:
        sampling_kwargs = kwargs.get('sampling_kwargs', {})
        training_args = kwargs.get('training_args', {})

        x, px, fx = self.sample_survey(**sampling_kwargs)
        integral_var, integral = torch.var_mean(fx / px)

        integral = integral.cpu().item()
        integral_var = integral_var.cpu().item()

        train_result = self.trainer.train_on_batch(
            x, px, fx,
            **training_args
        )

        self.process_survey_step(
            sample=(x, px, fx),
            integral=integral,
            integral_var=integral_var,
            train_result=train_result,
        )
    
    def refine_step(self, **kwargs) -> None:
        sampling_kwargs = kwargs.get('sampling_kwargs', {})

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
        self.logger.info('Initializing the survey phase')

        trainer_kwargs = kwargs.get('trainer_kwargs', {})
        self.trainer.set_config(trainer_kwargs)

        survey_step_kwargs = kwargs.get('survey_step_kwargs', {})
        survey_initialize_kwargs = kwargs.get('survey_initialize_kwargs', {})
        survey_finalize_kwargs = kwargs.get('survey_finalize_kwargs', {})

        self.initialize_survey(**survey_initialize_kwargs)

        self.logger.info('Starting the survey phase')

        for _ in range(n_steps):
            self.survey_step(**survey_step_kwargs)

        self.logger.info('Finalizing the survey phase')

        self.finalize_survey(**survey_finalize_kwargs)

    def refine(self, *, n_steps=10, **kwargs) -> None:
        self.logger.info('Initializing the refine phase')

        #TODO: Why is this needed?
        trainer_kwargs = kwargs.get('trainer_kwargs', {})
        self.trainer.set_config(trainer_kwargs)
        #

        refine_step_kwargs = kwargs.get('refine_step_kwargs', {})
        refine_initialize_kwargs = kwargs.get('refine_initialize_kwargs', {})
        refine_finalize_kwargs = kwargs.get('refine_finalize_kwargs', {})

        self.initialize_refine(**refine_initialize_kwargs)

        self.logger.info('Starting the refine phase')

        for _ in range(n_steps):
            self.survey_step(**refine_step_kwargs)
        
        self.logger.info('Finalizing the refine phase')

        self.finalize_refine(**refine_finalize_kwargs)

    def integrate(self, *, n_survey_steps, n_refine_steps, **kwargs) -> IntegrationResult:
        self.logger.info('Starting integration')

        initialize_kwargs = kwargs.get('initialize_kwargs', {})
        finalize_kwargs = kwargs.get('finalize_kwargs', {})
        survey_kwargs = kwargs.get('survey_kwargs', {})
        regine_kwargs = kwargs.get('regine_kwargs', {})

        self.logger.info('Initializing the integration')
        self.initialize(**initialize_kwargs)

        self.survey(n_survey_steps, **survey_kwargs)
        self.refine(n_refine_steps, **regine_kwargs)

        self.logger.info('Finalizing the integration')

        return self.finalize(**finalize_kwargs)


class BaseIntegrator(IntegratorAPI):
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
        super(BaseIntegrator, self).__init__(verbosity=verbosity, **kwargs)
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
