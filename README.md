# CS336 Spring 2025 Assignment 1: Ablations with no normalization within attention block

The purpose of this ablation experiment was to observe the training behavior of the model without layer normalization. 

It was immediately obvious that the training would not go well with previously optimal parameters and configurations. The loss climbed reach null after a few iterations of training. Looking at the activations I found that the final linear projection of the SWIGLU was unusually large. The default initialization (Xaviar initialization) was not adequate to contain the activations in a reasonable range. Scaling the initialization parameters be a factor of 0.01 was sufficient to bring the activations to a reasonable range. This stabilized training in the early iterations but the loss eventually climbed to null. It was also not possible to train at a lower learning rate. 

Next I observed that the output of the attention computation before exponentiation eventually reached levels at or above numerical stability. Exponentiation operation with those outputs was causing nan's to appear. The fix here was to clamp the output before exponentiation at the range of [-30,30]. This was sufficient eliminate the numerical instability and make progress in training the model. 

These two fixes allowed the model to train in a stable manner without layer normalization. 

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

