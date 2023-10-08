import typing as t

import torch
from torch import Tensor as tt

from flux.models.transforms.base import BaseCouplingTransform


class PWLinearCouplingTransform(BaseCouplingTransform):
    @staticmethod
    def forward(x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        N_, x_dim_, n_bins = theta.shape
        N, x_dim = x.shape

        assert N == N_, "Shape mismatch"
        assert x_dim == x_dim_, "Shape mismatch"

        theta = n_bins * torch.nn.Softmax(dim=2)(theta)

        bin_id = torch.floor(n_bins * x)
        bin_id = torch.clamp(bin_id, min=0, max=n_bins - 1)
        bin_id = bin_id.to(torch.int)

        if torch.any(torch.isnan(bin_id)).item():
            raise RuntimeError("NaN found!")
        if torch.any(bin_id < 0).item() or torch.any(bin_id > n_bins - 1).item():
            raise RuntimeError("Indexing error!")

        x_ = x - bin_id / n_bins
        slope = torch.gather(theta, dim=2, index=bin_id.unsqueeze(-1)).squeeze(-1)

        x_ *= slope

        log_jacobian = None
        if compute_log_jacobian:
            log_jacobian = torch.log(torch.prod(slope, dim=1))

        left_integral = torch.cumsum(theta, dim=2)
        left_integral = torch.roll(left_integral, shifts=1, dim=2)
        left_integral[:, :, 0] = 0
        left_integral = torch.gather(left_integral, dim=2, index=bin_id.unsqueeze(-1)).squeeze(-1)

        x_ += left_integral

        eps = torch.finfo(x_).eps
        x_ = torch.clamp(x_, min=eps, max=1 - eps)

        return x_, log_jacobian

    @staticmethod
    def backward(x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        N_, x_dim_, n_bins = theta.shape
        N, x_dim = x.shape

        assert N == N_, "Shape mismatch"
        assert x_dim == x_dim_, "Shape mismatch"

        theta = n_bins * torch.nn.Softmax(dim=2)(theta)

        left_integral = torch.cumsum(theta, dim=2)
        left_integral = torch.roll(left_integral, shifts=1, dim=2)
        left_integral[:, :, 0] = 0

        # TODO: Why do we need to detach here?
        overhead = (x.unsqueeze(-1) - left_integral).detach()
        overhead[overhead < 0] = 2

        bin_id = torch.argmin(overhead, dim=2)
        bin_id = torch.clamp(bin_id, min=0, max=n_bins - 1)

        if torch.any(torch.isnan(bin_id)).item():
            raise RuntimeError("NaN found!")
        if torch.any(bin_id < 0).item() or torch.any(bin_id > n_bins - 1).item():
            raise RuntimeError("Indexing error!")

        left_integral = torch.gather(left_integral, dim=2, index=bin_id.unsqueeze(-1)).squeeze(-1)
        slope = torch.gather(theta, dim=2, index=bin_id.unsqueeze(-1)).squeeze(-1)

        x_ = bin_id / n_bins + (x - left_integral) / slope

        eps = torch.finfo(x_).eps
        x_ = torch.clamp(x_, min=eps, max=1 - eps)

        log_jacobian = None
        if compute_log_jacobian:
            log_jacobian = -torch.log(torch.prod(slope, dim=1))

        # TODO: Why do we need to detach here?
        return x_.detach(), log_jacobian
