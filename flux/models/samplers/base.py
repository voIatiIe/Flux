import torch


class BaseSampler(torch.nn.Module):
    def __init__(self, *, dim, prior) -> None:
        super().__init__()

        self.dim = dim
        self.prior = prior

    def log_prob(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.dim, f'Shape mismatch! Expected: (:, {self.dim})'

        # Since the PDF(x) = PDF1(x1)*PDF(x2)*...*PDFdim(xdim), we have to sum log probability over all dimensions
        return torch.sum(self.prior.log_prob(x), -1)

    def forward(self, n_points):
        x = self.prior.sample((n_points, self.d))
        log_j = self.log_prob(x)

        return torch.cat([x, log_j.unsqueeze(-1)], -1)
