import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.linear_layer = nn.Parameter(
            self._init_model(in_features, out_features, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(self.linear_layer, x, "height width, ... width-> ... height")
        return out

    def _init_model(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        variance = 2 / (in_features + out_features)
        std = variance**0.5
        left_cutoff = -3 * std
        right_cutoff = 3 * std
        # weights = torch.empty((out_features, in_features), dtype=dtype, device=device)
        weights = torch.zeros((out_features, in_features), dtype=dtype, device=device)
        return torch.nn.init.trunc_normal_(
            weights, mean=0, std=std, a=left_cutoff, b=right_cutoff
        )
