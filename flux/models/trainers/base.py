import logging
import torch
import typing as t

from better_abc import abstract_attribute

from flux.models.samplers import BaseSampler
from flux.models.flows import BaseFlow
from flux.utils.logging import set_verbosity, VerbosityLevel
from flux.models.couplings.base import Mode


class BaseTrainer:
    set_verbosity = set_verbosity

    def __init__(
        self,
        *,
        flow: BaseFlow,
        prior: BaseSampler,
        verbosity: t.Optional[VerbosityLevel] = None,
    ) -> None:
        self.flow = flow
        self.prior = prior

        self.loss = abstract_attribute()

        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__ + ":" + hex(id(self)))
        self.set_verbosity(verbosity)

    def sample(self, n_points: int):
        if self.flow.is_inverted():
            self.flow.invert()

        x = self.prior(n_points)

        with torch.no_grad():
            xj = self.flow(x)

        return xj.detach()

    def train_batch(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        fx: torch.Tensor,
        *,
        n_epochs: int = 10,
        optimizer: torch.optim.Optimizer = None,
        minibatch_size: t.Optional[int] = None,
    ) -> None:

        minibatch_size = minibatch_size or x.shape[0]
        optimizer = optimizer or torch.optim.Adam(self.flow.parameters(), lr=1.e-4)

        for epoch in range(n_epochs):
            self.train_batch_step(
                x, px, fx, optimizer=optimizer, minibatch_size=minibatch_size,
            )

    def train_batch_step(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        fx: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        minibatch_size: int,
    ) -> None:
        i = 0
        while i < x.shape[0]:
            loss = self.train_minibatch(
                x[i:i+minibatch_size],
                px[i:i+minibatch_size],
                fx[i:i+minibatch_size],
                optimizer=optimizer,
            )
            print(loss)
            i += minibatch_size

    def train_minibatch(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        fx: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:

        if not self.flow.is_inverted():
            self.flow.invert()

        xj = torch.cat([x, torch.zeros(x.shape[0], 1).to(x.device)], dim=1)
        xj = self.flow(xj)
        print(xj)

        x = xj[:, :-1]
        log_qx = xj[:, -1] + self.prior.log_prob(x)

        optimizer.zero_grad()
        loss = self.loss(fx, px, log_qx)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()

        return loss
