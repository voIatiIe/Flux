import typing as t

from torch import Tensor


class BaseIntegrand:
    def __init__(
        self, *,
        dim: int,
        callable: t.Callable,
        target: t.Optional[float]=None
    ) -> None:
        self.calls = 0
        self.dim = dim
        self.callable: callable = callable
        self.target = target

    def __call__(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2 and x.shape[1] == self.dim, f'Shape mismatch! Expected: (:, {self.dim})'

        self.calls += x.shape[0]

        return self.callable(x)
    
    def reset(self):
        self.calls = 0
