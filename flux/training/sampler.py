import torch
import typing as t


class BaseSampler(torch.nn.Module):
    def __init__(self, *, dim, prior) -> None:
        super(BaseSampler, self).__init__()

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


class UniformSampler(BaseSampler):
    def __init__(
        self, *,
        dim,
        device: t.Optional[torch.device]=torch.device('cpu'),
    ) -> None:
        #TODO: Enable using dynamic integration domain

        lower = torch.tensor(0.0, dtype=torch.get_default_dtype()).to(device)
        upper = torch.tensor(1.0, dtype=torch.get_default_dtype()).to(device)

        prior = torch.distributions.Uniform(lower, upper)

        super(UniformSampler, self).__init__(dim=dim, prior=prior)


class GaussianSampler(BaseSampler):
    def __init__(
        self, *,
        dim,
        device: t.Optional[torch.device]=torch.device('cpu'),
    ) -> None:
        #TODO: Enable using dynamic Gauss parameters

        sig = torch.tensor(1.0, dtype=torch.get_default_dtype()).to(device)
        mu = torch.tensor(0.5, dtype=torch.get_default_dtype()).to(device)

        prior = torch.distributions.normal.Normal(mu, sig)

        super(GaussianSampler, self).__init__(dim=dim, prior=prior)
