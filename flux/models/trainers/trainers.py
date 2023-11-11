import torch

from flux.models.flows import BaseFlow
from flux.models.samplers import BaseSampler
from flux.models.trainers.base import BaseTrainer

from flux.utils.loss import variance_loss


class VarianceTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        flow: BaseFlow,
        prior: BaseSampler,
    ) -> None:
        super().__init__(flow=flow, prior=prior)
        self.loss = variance_loss
