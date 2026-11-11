# SQLgenerator

A learning project for training an autoregressive text-to-SQL model from
scratch, with an emphasis on pre-training and distributed execution. The
tokenizer, dataset, SQL dialect, and prompt format are still to be chosen.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository, and run from its root:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests
```

`.python-version` selects Python 3.13.15. `pyproject.toml` declares the project
dependencies; `uv.lock` records the resolved direct and transitive versions.
Both teammates use the same Git revision and `uv sync --locked` to build their
own ignored `.venv`. No activation is needed with `uv run`. uv itself is a
bootstrap tool; use uv 0.12.3 or later (the Modal image uses 0.13.7).

Use `uv add <package>` for intentional dependency changes, review changes to
both the configuration and lockfile, and commit them together. Avoid manually
installing packages into `.venv`: the next sync removes undeclared packages.

PyTorch is pinned to 2.10.0. macOS uses its native PyPI build; Linux and Windows
use PyTorch's CUDA 12.8 index. The Modal image uses the same lockfile and Python
version. This reproduces the Python dependencies for each platform, not the
host GPU driver or identical floating-point results across hardware.
PyTorch is in the default `model` dependency group. The CPU-only tracking
image omits that group, avoiding CUDA libraries while keeping W&B's locked
dependencies. Local setup and GPU images include the group by default.

## Modal and W&B setup

Each teammate authenticates with their own account:

```bash
uv run --locked modal setup
uv run --locked modal profile list
```

The existing project workspace is `sanika-05`, with environment `main`, Volume
`sqlgen-data` (v2), and Secret `wandb-secret`. Teammates need access to the same
workspace to share these resources; do not share personal Modal tokens.
The volume is for future datasets and checkpoints; these smoke tests do not
write to it.

For a separate workspace, create its volume once:

```bash
uv run --locked modal volume create sqlgen-data --version 2 --env main
```

In the Modal dashboard, create `wandb-secret` with the field `WANDB_API_KEY`.
Enter credentials directly there, never in Git, chat, or command arguments.
W&B team/entity: `dkbsingh7-iit-delhi`; smoke-test project: `sqlgen-infra`.
The logging preflight creates that project privately if it does not exist;
the API key must have permission to create projects and log runs in the team.
Existing project visibility is preserved.

To verify W&B authentication and logging without allocating a GPU:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus none --log-wandb --wandb-entity dkbsingh7-iit-delhi
```

This creates a tracking-only run with `tracking_connected=true` and
`gpu_checks_run=0`; it makes no claim that GPU checks passed.

## GPU smoke tests

These commands run billable remote jobs. Each GPU function has a 300-second
timeout, its worker subprocess has a 180-second timeout, and NCCL has a
60-second collective timeout. Startup has separate deadlines, and the caller
cancels calls after three minutes for CPU checks or six minutes for GPU checks,
including queue time. Containers scale down after a short idle window.
No training or persistent deployment is started.
`--gpu-wait-seconds 600` allows a longer allocation wait without increasing
the GPU function's execution timeout. A wait timeout means the call did not
finish in that window; it is not evidence of an all-reduce failure if no
worker reached the test.

Run the single-GPU check first:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus 1
```

Then run the communication check:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus 2
```

To run both and record one W&B run:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus both --log-wandb --wandb-entity dkbsingh7-iit-delhi --wandb-project sqlgen-infra
```

`modal_smoke.py` allocates the GPUs, then launches `scripts/gpu_smoke_worker.py`
with `torchrun`, one process per GPU. A **rank** identifies a process in the
communication group; **world size** is the number of processes; **local rank**
selects its GPU on this machine. For two L4s, world size is 2 and local ranks
are 0 and 1. NCCL is the communication backend used by these NVIDIA GPUs.

Each worker prints the visible GPU count, its assigned device and GPU name,
and asserts a small CUDA computation. For two GPUs, rank 0 starts with `[1]`
and rank 1 with `[2]`; a SUM all-reduce must leave **both** with `[3]`. Both
workers assert the result and destroy the process group on completion.
The single-GPU check leaves its tensor at `[1]` without initializing NCCL.

When logging is requested, a CPU preflight checks project access before GPU
allocation. After the GPU checks, a separate CPU function uses `wandb-secret`
to log pass indicators, elapsed time, and a table of worker results. GPU
workers do not receive the W&B credential. Elapsed time excludes allocation
waiting and includes worker interpreter startup; it is not a communication
benchmark. Passing these tests verifies basic GPU
execution and communication, not training correctness or scaling efficiency.

If a check fails, inspect the first worker error in Modal's logs. CUDA checks
fail explicitly if the allocation is missing or unexpected; torchrun worker
failures propagate to the caller. A W&B authentication or permission failure
is reported as an error rather than silently switching to offline logging.

## Model

```python
import torch
from sqlgenerator import CausalLanguageModel, ModelConfig

model = CausalLanguageModel(ModelConfig(vocab_size=tokenizer.vocab_size))
input_ids = torch.tensor([[...]])
logits, loss = model(input_ids, labels=input_ids)
```

Provide token IDs from the tokenizer and data pipeline selected for the target
database schema. `generate` returns token IDs, leaving decoding and SQL
validation to the application.

## Parallelism

For one process, use the model directly. For a single-process multi-GPU
prototype, call `wrap_model(model, torch.device("cuda"), "data_parallel")`.
For scalable training, launch one process per GPU:

```bash
torchrun --nproc_per_node=2 <your_training_entrypoint>.py
```

Inside each worker, call `initialize_distributed()` and pass its returned
device to `wrap_model(model, device, "ddp")`. Dataset sharding, optimizer
configuration, checkpoint policy, and training loop are intentionally left to
the training application.
