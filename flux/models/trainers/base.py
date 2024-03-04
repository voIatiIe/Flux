import logging
import torch
import typing as t

from torchviz import make_dot

from flux.models.samplers import BaseSampler
from flux.models.flows import BaseFlow
from flux.utils.distributed import average_gradients
from flux.utils.logging import set_verbosity, VerbosityLevel


class BaseTrainer:
    set_verbosity = set_verbosity

    def __init__(
        self,
        *,
        flow: BaseFlow,
        prior: BaseSampler,
        sample_prior: BaseSampler | None,
        verbosity: t.Optional[VerbosityLevel] = None,
        average_grads: bool = False,
        loss: t.Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self.flow = flow
        self.prior = prior
        self.sample_prior = sample_prior
        self.average_grads = average_grads
        self.loss = loss

        self.last_loss: float | None = None
        self.sample_forward: bool = False
        self.step: int = 0

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
        minibatch_share: float = 1.0,
    ) -> None:
        # print(f'Train step {self.step}')
        assert 0.0 < minibatch_share <= 1.0, f"Invalid minibatch share: {minibatch_share}"

        optimizer = optimizer or torch.optim.Adam(self.flow.parameters(), lr=1e-3, weight_decay=1e-5)

        # for el in self.flow.parameters():
        #     print(f'PARAMETERS: {el}')

        for epoch in range(n_epochs):
            self.train_batch_step(
                x, px, fx, optimizer=optimizer, minibatch_share=minibatch_share,
            )

        self.process_train_batch_step(x, px, fx)
        self.step += 1

    def train_batch_step(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        fx: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        minibatch_share: float,
    ) -> None:
        minibatch_size = int(minibatch_share * x.shape[0])

        i = 0
        while i < x.shape[0]:
            loss = self.train_minibatch(
                x[i:i+minibatch_size],
                px[i:i+minibatch_size],
                fx[i:i+minibatch_size],
                optimizer=optimizer,
            )
            i += minibatch_size

            self.process_loss(loss)

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

        x = xj[:, :-1]
        log_qx = xj[:, -1] + self.prior.log_prob(x)

        optimizer.zero_grad()
        loss = self.loss(fx, px, log_qx)
        loss.backward()

        # print(f'{x=:}')
        # print(f'{xj=:}')
        # print(f'{fx=:}')
        # print(f'{px=:}')
        # print(f'{log_qx=:}')
        print(f'loss: {loss}')

        # if self.average_grads:
        #     average_gradients(self.flow)

        optimizer.step()

        loss = loss.detach().cpu().item()

        return loss


    def process_loss(self, loss: float) -> None:
        if self.last_loss is None:
            self.last_loss = loss
        else:
            self.last_loss = self.last_loss * (1 - 0.1) + loss * 0.9

    def process_train_batch_step(self, x: torch.Tensor, px: torch.Tensor, fx: torch.Tensor) -> None:
        if not self.sample_forward and self.step > 2:
            flat_loss_mean = (fx**2).mean().cpu().item()
            flat_loss_std = min((fx**2).std().cpu().item(), flat_loss_mean/4)

            # print(f'{self.last_loss=:} {flat_loss_mean=:} {flat_loss_std=:}')

            switch_loss_threshold = flat_loss_mean - flat_loss_std

            if self.last_loss < switch_loss_threshold:
                print("Switched to forward sampling mode")
                self.sample_forward = True
