import torch
import typing as t

from flux.models.trainables.base import BaseTrainable


class DNNTrainable(BaseTrainable):
    def __init__(
        self, *,
        dim_in: int,
        out_shape: torch.Tensor,
        n_hidden: int,
        dim_hidden: int,
        hidden_activation: torch.nn.Module = torch.nn.ReLU,
        input_activation: t.Optional[torch.nn.Module] = None,
        output_activation: t.Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(dim_in=dim_in, out_shape=out_shape)

        self.trainable = self.trainable_(
            n_hidden=n_hidden,
            dim_hidden=dim_hidden,
            hidden_activation=hidden_activation,
            input_activation=input_activation,
            output_activation=output_activation,
        )

    def trainable_(
        self, *,
        n_hidden: int,
        dim_hidden: int,
        hidden_activation: torch.nn.Module,
        input_activation: t.Optional[torch.nn.Module],
        output_activation: t.Optional[torch.nn.Module],
    ) -> torch.nn.Sequential:
        layers = []

        if input_activation:
            layers.append(input_activation())

        layers.append(torch.nn.Linear(self.dim_in, dim_hidden))
        layers.append(hidden_activation())

        for _ in range(n_hidden):
            layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            layers.append(hidden_activation())
        
        layers.append(torch.nn.Linear(dim_hidden, self.dim_out))
        if output_activation:
            layers.append(output_activation())

        return torch.nn.Sequential(*layers)
