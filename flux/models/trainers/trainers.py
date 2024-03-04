import torch

from flux.models.flows import BaseFlow
from flux.models.samplers import BaseSampler
from flux.models.trainers.base import BaseTrainer

from flux.utils.loss import variance_loss, dkl_loss


class VarianceTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        flow: BaseFlow,
        prior: BaseSampler,
        sample_prior: BaseSampler | None = None,
        average_grads: bool = False
    ) -> None:
        super().__init__(
            flow=flow,
            prior=prior,
            average_grads=average_grads,
            sample_prior=sample_prior,
            loss=variance_loss,
        )


class DKLTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        flow: BaseFlow,
        prior: BaseSampler,
        sample_prior: BaseSampler | None = None,
        average_grads: bool = False
    ) -> None:
        super().__init__(
            flow=flow,
            prior=prior,
            average_grads=average_grads,
            sample_prior=sample_prior,
            loss=dkl_loss,
        )
