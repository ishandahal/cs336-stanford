import math
import random
from typing import Iterable

import numpy as np
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x)
    x = x - max_val
    x_exp = torch.exp(x)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    largest = torch.max(logits)
    # Subtract the largest for numerical stability during exponentiation
    logits = logits - largest

    indices = targets.unsqueeze(-1)
    logits_of_targets = logits.gather(dim=-1, index=indices).squeeze(-1)
    # logsumexp does the exponentiation and then sum along the given dimension and takes a log in a numerical stable manner
    denom = torch.logsumexp(logits, dim=-1)
    # Logic below can be simplified but leaving it for learning purposes
    loss = logits_of_targets - denom  # log of division is a subtraction
    neg_loss = -loss
    return neg_loss.mean()


def learning_rate_scheduler(
    iter: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_annealing_iters: int,
) -> float:
    if iter <= warmup_iters:
        return (iter / warmup_iters) * lr_max
    if warmup_iters < iter < cosine_annealing_iters:
        cosine_part = 1 + math.cos(
            ((iter - warmup_iters) / (cosine_annealing_iters - warmup_iters)) * math.pi
        )
        return lr_min + ((0.5 * cosine_part) * (lr_max - lr_min))

    return lr_min


def gradient_clipping(
    parameter_list: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6
) -> None:
    accumulate = 0
    for parameter in parameter_list:
        if parameter.requires_grad:
            accumulate += (parameter.grad**2).sum()
    l2_norm = math.sqrt(accumulate)

    if l2_norm < max_norm:
        return
    for parameter in parameter_list:
        if parameter.requires_grad:
            parameter.grad *= max_norm / (l2_norm + eps)
