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
