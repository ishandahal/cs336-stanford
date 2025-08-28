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
