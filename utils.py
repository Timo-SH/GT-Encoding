import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import math
import torch.nn.functional as F


class CosineWithWarmupLR:
    """Adapted from https://github.com/karpathy/nanoGPT
    Provides the CosineWithWarmupLR scheduler """

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
        epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.epoch = epoch
        self.step()

    def step(self):
        self.epoch += 1
        lr = self._get_lr(self.epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int):
        # 1) linear warmup for warmup_iters steps
        if epoch < self.warmup_iters:
            return self.lr * epoch / self.warmup_iters
        # 2) if epoch > lr_decay_iters, return min learning rate
        if epoch > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)

