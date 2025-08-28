import torch
import torch.nn as nn
from einops import einsum, rearrange


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


class SwiGlu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = nn.Parameter(
            self._init_model(d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            self._init_model(d_model, d_ff, device=device, dtype=dtype)
        )
        self.w3 = nn.Parameter(
            self._init_model(d_ff, d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_partial_output = einsum(
            self.w1, x, "d_ff d_model, ... d_model -> ... d_ff"
        )
        silu = silu_partial_output * torch.nn.functional.sigmoid(silu_partial_output)
        gated = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        swiglu_partial = silu * gated
        swiglu = einsum(
            self.w2, swiglu_partial, "d_model d_ff, ... d_ff -> ... d_model"
        )
        return swiglu

    def _init_model(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        left_cutoff = -3
        right_cutoff = 3
        # weights = torch.empty((d_in, d_out), dtype=dtype, device=device)
        weights = torch.zeros((d_in, d_out), dtype=dtype, device=device)
        return torch.nn.init.trunc_normal_(weights, a=left_cutoff, b=right_cutoff)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotational Positional Embedding (RoPE) implementation.

    Applies RoPE to input tensors based on token positions.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: theta value for the RoPE
            d_k: dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute frequencies for each dimension pair
        # freqs_i = theta^(-2i/d_k) for i in [0, 1, ..., d_k//2-1]
        # freqs = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)

        # Corrected frequency calculation
        i = torch.arange(0, d_k // 2, dtype=torch.float32)
        freqs = theta ** (-2 * i / d_k)

        # Precompute position encodings for all possible positions
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # Compute angles: outer product of positions and frequencies
        angles = torch.outer(positions, freqs)  # shape: (max_seq_len, d_k//2)

        # Precompute cos and sin values
        cos_vals = torch.cos(angles)  # shape: (max_seq_len, d_k//2)
        sin_vals = torch.sin(angles)  # shape: (max_seq_len, d_k//2)

        # Register as buffers
        self.register_buffer("cos_vals", cos_vals.to(device))
        self.register_buffer("sin_vals", sin_vals.to(device))

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Optional tensor of shape (..., seq_len).
                        If None, uses sequential positions [0, 1, 2, ...]

        Returns:
            Tensor of the same shape as x with RoPE applied
        """
        seq_len = x.shape[-2]

        if token_positions is None:
            # Create sequential positions: [0, 1, 2, ..., seq_len-1]
            token_positions = torch.arange(seq_len, device=x.device)
            # Expand to match the batch dimensions of x
            batch_shape = x.shape[:-2]  # All dimensions except seq_len and d_k
            token_positions = token_positions.expand(*batch_shape, seq_len)

        # Get the cos and sin values for the specified token positions
        cos = self.cos_vals[token_positions]  # shape: (..., seq_len, d_k//2)
        sin = self.sin_vals[token_positions]  # shape: (..., seq_len, d_k//2)

        # Split x into pairs for rotation
        x1 = x[..., ::2]  # Even indices: (..., seq_len, d_k//2)
        x2 = x[..., 1::2]  # Odd indices: (..., seq_len, d_k//2)

        # Apply rotation
        # [cos -sin] [x1]   [x1*cos - x2*sin]
        # [sin  cos] [x2] = [x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave the results back to original shape
        # result = torch.empty_like(x)
        result = torch.zeros_like(x)
        result[..., ::2] = rotated_x1
        result[..., 1::2] = rotated_x2

        return result


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x)
    x = x - max_val
    x_exp = torch.exp(x)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    scale = q.shape[-1] ** 0.5
    # Dotproduct between query and key
    q_dot_k = einsum(
        q, k, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    )
    q_dot_k_scaled = q_dot_k / scale
    # Use the mask for causal self attention
    if mask is not None:
        q_dot_k_scaled = q_dot_k_scaled.masked_fill(~mask, float("-inf"))
    attention_weights = softmax(q_dot_k_scaled, dim=-1)
    attended_tokens = einsum(
        attention_weights,
        v,
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
    )
    return attended_tokens


class MultiHeadedAttentionWithRope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        assert d_model % num_heads == 0, "input dim not evenly divisible by num_heads"
        super().__init__()
        self.num_heads = num_heads
        self.d_kqv = d_model // num_heads
        self.key = Linear(d_model, d_model, device=device, dtype=dtype)
        self.query = Linear(d_model, d_model, device=device, dtype=dtype)
        self.value = Linear(d_model, d_model, device=device, dtype=dtype)
        # self.combined_kqv = Linear(d_model, 3 * d_model)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Instantiatie ROPE
        self.rope = RotaryPositionalEmbedding(
            theta=self.theta,
            d_k=self.d_kqv,
            max_seq_len=self.max_seq_len,
            device=device,
        )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:

        batch, sequence, d_model = x.shape
        k_w = self.key.get_parameter("linear_layer")
        q_w = self.query.get_parameter("linear_layer")
        v_w = self.value.get_parameter("linear_layer")
        # combined_qkv_weights = torch.stack([k_w, q_w, v_w], dim=0)
        # combined_qkv = einsum(x, combined_qkv_weights, "b s d, kqv k d -> b s kqv k")
        # key, query, value = rearrange(combined_qkv, "b s n d_m -> n b s d_m")

        # workaround for 'mps'
        # Combine weights: (d_model, 3 * d_out)
        combined_weight = torch.cat([k_w.T, q_w.T, v_w.T], dim=1)

        # Single projection: (b, s, d_model) @ (d_model, 3 * d_out) = (b, s, 3 * d_out)
        combined_qkv = x @ combined_weight  # same as F.linear(x, combined_weight)

        # Reshape to (b, s, 3, d_out)
        d_out = combined_qkv.size(-1) // 3
        combined_qkv = combined_qkv.view(batch, sequence, 3, d_out)

        # Rearrange to (3, b, s, d_out), then split
        qkv_rearranged = combined_qkv.permute(2, 0, 1, 3)  # (3, b, s, d_out)
        key, query, value = qkv_rearranged.unbind(0)  # 3 x (b, s, d_out)

        key = rearrange(
            key,
            "b s (h d) -> b h s d",
            h=self.num_heads,
        )
        key = self.rope(key, token_positions)  # Applying positional embeddings

        query = rearrange(
            query,
            "b s (h d) -> b h s d",
            h=self.num_heads,
        )
        query = self.rope(query, token_positions)  # Applying positional embeddings

        value = rearrange(
            value,
            "b s (h d) -> b h s d",
            h=self.num_heads,
        )

        mask = torch.tril(
            torch.ones((sequence, sequence), dtype=torch.bool, device=x.device)
        )
        result = scaled_dot_product_attention(
            query, key, value, mask
        )  # batch num_head sequence head_dim
        result = rearrange(result, "b h s d -> b s (h d)", h=self.num_heads)
        final_projection = self.out_proj(result)
        return final_projection


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.block = MultiHeadedAttentionWithRope(
            d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype
        )
        self.ff = SwiGlu(d_model, d_ff, device=device, dtype=dtype)
        self.rms_ln = RMSLayerNorm(d_model=d_model, device=device, dtype=dtype)
        self.rms_ln2 = RMSLayerNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.rms_ln(x)
        x_mha = self.block(x_norm) + x  # residual connection

        x_norm2 = self.rms_ln2(x_mha)
        ff_proj = self.ff(x_norm2) + x_mha  # residual connection
        return ff_proj


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.lm = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.ln = RMSLayerNorm(d_model, dtype=dtype, device=device)
        self.out_proj = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for transformer_block in self.lm:
            x = transformer_block(x)
        x = self.ln(x)
        logits = self.out_proj(x)
        # probs = softmax(logits, dim=-1)
        return logits
