from torch.nn import Module
import torch.distributed as dist


def average_gradients(model: Module):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is None:
            continue

        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
