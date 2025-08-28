import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x)
    x = x - max_val
    x_exp = torch.exp(x)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
