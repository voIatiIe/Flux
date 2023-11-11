import os
import torch.distributed as dist


class ProcessGroupManager:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def __enter__(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group('nccl', rank=self.rank, world_size=self.world_size)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.barrier()
        dist.destroy_process_group()
