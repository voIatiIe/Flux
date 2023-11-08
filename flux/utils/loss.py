import torch


def variance_loss(
    fx: torch.Tensor,
    px: torch.Tensor,
    log_qx: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(fx ** 2 / (px * torch.exp(log_qx)))
