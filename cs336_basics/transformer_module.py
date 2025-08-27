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
        # Using torch.empty caused issues with macbook
        # weights = torch.empty((out_features, in_features), dtype=dtype, device=device)
        weights = torch.zeros((out_features, in_features), dtype=dtype, device=device)
        return torch.nn.init.trunc_normal_(
            weights, mean=0, std=std, a=left_cutoff, b=right_cutoff
        )


class Embedding(nn.Module):
    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_layer = nn.Parameter(
            self._init_model(num_embedding, embedding_dim, device=device, dtype=dtype)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        embs = self.embedding_layer[
            token_ids
        ]  # Batch size * Sequence length * Embedding dim
        return embs

    def _init_model(
        self,
        num_embedding: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        left_cutoff = -3
        right_cutoff = 3
        # Macbook "mps" did not like `empty`
        # weights = torch.empty(
        #     (num_embedding, embedding_dim), dtype=dtype, device=device
        # )
        weights = torch.zeros(
            (num_embedding, embedding_dim), dtype=dtype, device=device
        )
        return torch.nn.init.trunc_normal_(weights, a=left_cutoff, b=right_cutoff)


class RMSLayerNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # Using higher precision
        x = x.to(torch.float32)
        gain = self.gain.to(torch.float32)
        root_mean_square = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps
        rms_norm = x / root_mean_square * gain
        # Return to original dtype
        return rms_norm.to(in_dtype)
