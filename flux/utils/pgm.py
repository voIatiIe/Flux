import os
import torch.distributed as dist


class ProcessGroupManager:
    def __init__(self, rank: int, world_size: int, backend: str) -> None:
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

    def __enter__(self) -> 'ProcessGroupManager':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size,
        )

        return self

    def __exit__(self, *args, **kwargs) -> None:
        dist.barrier()
        dist.destroy_process_group()
