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


def train(args):
    # run = wandb.init(
    #     project="gpt-2-run-layer-norm-ablation-v0.0",  # Specify project
    #     config=args,  # Track hyperparameters and metadata
    # )

    model = get_model(args)
    model = torch.compile(model, backend="aot_eager")  # For macbook
    tokenizer = get_tokenizer()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate_max,
        weight_decay=args.weight_decay,
    )

    print("Initiating model training")
    model.train()
    # Accumulate training loss
    running_loss = 0
    prompt = "Once upon a time"
    generated_samples = []
    warmup_iters = int(0.1 * args.steps)
    for i in range(args.steps):

        # Calculate current learning rate
        current_lr = learning_rate_scheduler(
            i,
            lr_max=args.learning_rate_max,
            lr_min=args.learning_rate_min,
            warmup_iters=warmup_iters,
            cosine_annealing_iters=args.steps,
        )
        # Update all parameter group
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        start = time.time()  # To track one batch
        i = i + 1  # For easier iteration logging
        x, y = get_batch(
            args.train_path,
            args.batch_size,
            args.context_length,
            deterministic_sampling=None,
            device=args.device,
        )

        # forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # Backward pass
        loss.backward()

        # If the norm of the gradients are too large, adjust
        gradient_clipping(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

        end = time.time()  # To track one batch training

        # Accumulate loss
        running_loss += loss.item()
        if not (i % 100):
            model.eval()
            average_val_loss = evaluate_validation(model, args)
            perplexity = torch.exp(average_val_loss)

            print(
                f"iteration: {i} | training running loss: {running_loss / (i)} | average val loss: {average_val_loss} | average perplexity: {perplexity}"
            )
            print(f"Time elapsed for one training step: {elapsed_seconds(start, end)}")
            if not (i % 500):
                print("\tGenerating some text: ")
                response = decoding(
                    tokenizer,
                    prompt,
                    context_length=args.context_length,
                    max_length=args.max_length,
                    model=model,
                    temp=args.temp,
                    top_p=args.top_p,
                    device=args.device,
                )
                # Collect generated samples
                generated_samples.append(response)
            # Logging loss to weights and biases
            # run.log(
            #     {
            #         "train_loss": running_loss / i,
            #         "validation_loss": average_val_loss,
            #         "validation_perplexity": perplexity,
            #     },
            #     step=i,
            # )

        # Serialize model and optimizer states
        # if not (i % 1000):
        #     print(f"check pointing model at iteration: {i}")
        #     save_checkpoint(
        #         model, optimizer, (i - 1), args.checkpoint_path
        #     )  # i is subtraction by 1 above for ease

    # Weights and bias specific logging
    # table = wandb.Table(["step", "prompt", "output"])
    # for j, (prompt, response) in enumerate(
    #     zip((prompt for _ in range(len(generated_samples))), generated_samples)
    # ):
    #     table.add_data(j, prompt, response)
    #
    # wandb.log({"generated text samples": table})
    # run.finish()
    print("Training complete")
