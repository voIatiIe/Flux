import torch


class BaseTrainable(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        out_shape: torch.Tensor,
    ) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.out_shape = out_shape
        self.dim_out = 1
        for d in out_shape:
            self.dim_out *= d

        # self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trainable(x).view(x.shape[0], *self.out_shape)
