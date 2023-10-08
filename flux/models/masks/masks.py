import typing as t

from flux.models.masks.base import BaseMask


class СheckerboardMask(BaseMask):
    def masks(self) -> t.List[t.List[bool]]:
        masks = []

        mod = 1
        while mod < self.dim:
            mask = []
            val = False

            i = 0
            while i < self.dim:
                for _ in range(mod):
                    if i >= self.dim:
                        break
                    mask.append(val)
                    i += 1

                val = not val

            masks.append(mask)
            masks.append([not m for m in mask])

            mod *= 2

        return masks


class StrideMask(BaseMask):
    def __init__(
        self, *,
        dim: int,
        spread: int = 1,
        stride: t.Optional[int] = None,
        n_masks: t.Optional[int] = None,
    ) -> None:
        super().__init__(dim=dim, n_masks=n_masks)

        assert spread > 0, 'Spread must be greater than zero!'
        assert spread < dim, 'Spread must be less than dim!'
        if stride is not None:
            assert stride > 0, 'Stride must be greater than zero!'

        self.spread = spread
        self.stride = stride or spread

    def masks(self) -> t.List[t.List[bool]]:
        masks = []

        for i in range(0, self.dim,self.stride):
            mask = [False for _ in range(self.dim)]

            mask[i:i+self.spread] = [True for _ in range(self.spread)]
            mask = mask[:self.dim]

            masks.append(mask)
            masks.append([not m for m in mask])

        return masks
