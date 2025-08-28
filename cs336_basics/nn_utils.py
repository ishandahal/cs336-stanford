import math
import os
import random
from typing import IO, BinaryIO, Iterable

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


def data_loading(
    x: np.array,
    batch_size: int,
    context_length: int,
    deterministic_sampling: random.Random | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    size = len(x)

    # We also want the labels which requires an extra offset
    ceil = size - context_length - 1

    rng = deterministic_sampling if deterministic_sampling is not None else random

    ids, labels = [], []
    for _ in range(batch_size):
        start_index = rng.randint(0, ceil)
        end_index = start_index + context_length
        # if train_on_one_batch:
        # print(f"start and end indices: {start_index}, {end_index}")
        ids.append(x[start_index:end_index])
        labels.append(x[start_index + 1 : end_index + 1])
    # Torch gave a warning that converting list to np.array to tensor is faster
    # Ids are saved as uint16 which is not supported in torch so need to convert to long
    ids = torch.tensor(np.array(ids), device=device, dtype=torch.long)
    labels = torch.tensor(np.array(labels), device=device, dtype=torch.long)
    return ids, labels


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    "Save model checkpoint"
    obj = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    "Load model checkpoint"
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    return obj["iteration"]


def decoding(
    tokenizer,
    prompt: str,
    context_length: int,
    max_length: int,
    model: torch.nn.Module,
    temp: float,
    top_p: float,
    device: torch.device,
) -> str:
    eot_token = "<|endoftext|>"
    eot_token_id = tokenizer.encode(eot_token)[0]

    ids = tokenizer.encode(prompt)
    ids = ids[-context_length:]
    ids = torch.tensor(ids, device=device, dtype=torch.long)
    ids = rearrange(ids, "seq -> 1 seq")

    print(f"Prompt: {prompt}")
    print("Output:")

    result = ""

    counter = 0
    with torch.no_grad():
        while True:
            logits = model(ids)
            # Temperature scaling the logits
            probs = torch.softmax(logits / temp, dim=-1)
            prob_next_token = probs[0, -1, :]
            # most_likely = torch.argmax(next_token_prob, dim=-1)
            sorted_values, sorted_indices = prob_next_token.sort(
                dim=-1, descending=True
            )
            # Keep only top p probability choices
            cutoff_mask = torch.cumsum(sorted_values, dim=-1) < top_p
            cutoff_mask[0] = True  # Make sure we always have a choice
            top_p_indices = sorted_indices[cutoff_mask]
            top_p_probs = prob_next_token[top_p_indices]
            top_p_probs = top_p_probs / top_p_probs.sum()

            sample_idx = torch.multinomial(top_p_probs, num_samples=1)
            sample = top_p_indices[sample_idx]
            next_token = tokenizer.decode([sample.item()])
            if sample.item() == eot_token_id:
                break
            if counter > max_length:
                break
            result += next_token
            print(next_token, end="")

            counter += 1

            ids = torch.cat([ids, sample[None]], dim=1)

            # If the sequence of the text is longer than the context_length
            # only keep the last context_length tokens
            if ids.shape[1] > context_length:
                ids = ids[:, -context_length:]
    print("\n")
    return result


def gpt2_traininable_param_count(d_model, vocab_size, num_layers, d_ff):
    # Embedding layer
    embed = d_model * vocab_size

    # Transformer block
    layer_norm1 = d_model
    # Attention
    qkv = 3 * d_model * d_model
    o_proj = d_model * d_model
    attn = qkv + o_proj

    layer_norm2 = d_model
    layer_norm = layer_norm1 + layer_norm2
    # FF
    ff1 = d_model * d_ff
    ff2 = d_model * d_ff
    ff3 = d_ff * d_model
    ff = ff1 + ff2 + ff3

    mha = layer_norm + attn + ff

    # layer norm before final projection matrix to vocab_size
    layer_norm_final = d_model
    # output_projection = d_model * vocab_size # No need to count this because of weight tying

    params = embed + (num_layers * mha) + layer_norm_final
    return params


def get_parameter_count(d_model, num_layers, vocab_size):

    rms_norm = d_model

    # Attention
    qkv = d_model * d_model * 3
    o_proj = d_model * d_model
    attention = qkv + o_proj

    # ff
    w1 = d_model * d_model * 4  # d_ff = 4 * d_model
    w2 = d_model * d_model * 4
    ff = w1 + w2

    output = d_model * vocab_size  # Same as embedding (weight tying)

    return (rms_norm + attention + ff) * num_layers + rms_norm + output


def get_activation_count(
    batch_size, sequence_len, d_model, n_heads, vocab_size, num_layers
):
    tokens = batch_size * sequence_len * d_model
    na = batch_size * n_heads * sequence_len * sequence_len
    nl = batch_size * sequence_len * vocab_size

    rms_norm = tokens

    # Attention
    qkv = tokens
    qdotk = 2 * tokens
    softmax = na
    pv = tokens + na
    o_proj = tokens
    attention = qkv + qdotk + softmax + pv + o_proj

    # Ff
    ff1 = tokens
    silu = 4 * tokens
    ff2 = 4 * tokens
    ff = ff1 + silu + ff2

    model_acts = num_layers * (attention + rms_norm + ff)
    embedding = tokens
    cross_entropy = nl

    return model_acts + embedding + cross_entropy


def gpt2_matmul_flops(d_model, context_length, vocab_size, num_layers):
    # Tranformer block matmuls
    # Attention
    qkv_matmul = (
        3 * 2 * (context_length * d_model * d_model)
    )  # qkv requires 3 matrix multiplies and 2 for matmul flops
    q_dot_t = 2 * (context_length * d_model * d_model)
    qt_v = 2 * (context_length * context_length * d_model)
    concat_o_proj = 2 * (context_length * d_model * d_model)
    attention_flops = qkv_matmul + q_dot_t + qt_v + concat_o_proj

    # FeedForward
    ff1 = 2 * (context_length * d_model * d_model * 4)  # d_ff == d_model * 4 typically
    ff2 = 2 * (context_length * d_model * d_model * 4)
    ff3 = 2 * (context_length * d_model * d_model * 4)
    ff = ff1 + ff2 + ff3

    transformer_block_flops = attention_flops + ff

    # Projection to vocab_size
    output_proj = 2 * (context_length * d_model * vocab_size)

    total_flops = (num_layers * transformer_block_flops) + output_proj

    return total_flops
