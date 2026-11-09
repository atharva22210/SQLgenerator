# SQLgenerator

Minimal, unopinionated PyTorch building blocks for an autoregressive SQL
generator. The repository deliberately does not choose a tokenizer, dataset,
SQL dialect, prompt format, or pretrained checkpoint.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

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
