import typing as t

from abc import ABC, abstractmethod


class BaseMask(ABC):
    def __init__(self, *, dim: int, n_masks: t.Optional[int] = None, **kwargs) -> None:
        self.dim = dim
        self.n_masks = n_masks

    def __call__(self) -> t.List[t.List[bool]]:
        masks = self.masks()

        return self.extend(masks)

    def extend(self, masks: t.List[t.List[bool]]) -> t.List[t.List[bool]]:
        if self.n_masks is not None:
            n_copies, n_addition = divmod(self.n_masks, len(masks))

            addition = masks[:n_addition]
            masks *= n_copies
            masks += addition
        
        return masks

    @abstractmethod
    def masks(self) -> t.List[t.List[bool]]:
        pass
