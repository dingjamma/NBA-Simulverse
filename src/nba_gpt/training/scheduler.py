"""
Cosine LR schedule with linear warmup.
"""
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
