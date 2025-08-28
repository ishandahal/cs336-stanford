import argparse
import random
import time

import numpy as np
import torch
from nn_utils import (
    cross_entropy,
    data_loading,
    decoding,
    gradient_clipping,
    learning_rate_scheduler,
    save_checkpoint,
)
from tokenizer import Tokenizer, elapsed_seconds
from transformer_module import AdamW, TransformerModel

import wandb


def get_batch(
    path: str,
    batch_size: int,
    context_length: int,
    deterministic_sampling: random.Random | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.memmap(path, dtype=np.uint16)
    return data_loading(x, batch_size, context_length, deterministic_sampling, device)


def evaluate_validation(model, args):
    """Evaluate the model on validation set"""
    # Track time for validation runs
    start = time.time()
    running_loss = 0
    steps = args.validation_steps
    rng = random.Random(42)

    # print("Evaluation on validation: ")
    for _ in range(steps):
        # print(f"Eval: iteration {i}")
        with torch.no_grad():
            x, y = get_batch(
                args.valid_path,
                args.batch_size,
                args.context_length,
                deterministic_sampling=rng,
                device=args.device,
            )
            logits = model(x)
            loss = cross_entropy(logits, y)
            running_loss += loss.item()
    end = time.time()
    print(f"Time elapsed for validation eval: {elapsed_seconds(start, end)}")
    # Average loss
    return running_loss / steps


def get_model(args):
    return TransformerModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=args.device,
        dtype=args.dtype,
        theta=args.theta,
    )


def get_tokenizer():
    vocab_filepath = "data/bpe_tiny_stories_vocab_10k.pkl"
    merges_filepath = "data/bpe_tiny_stories_merges_10k.pkl"
    special_tokens = ["<|endoftext|>"]
    return Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
