# Day 5: one-GPU batching experiment

The question is how much padding costs, and whether isolating packed documents
changes validation loss. The baseline is `pad`: one document per row, with
padding targets ignored. `naive_pack` joins documents under ordinary causal
attention. `masked_pack` resets positions, blocks attention across document
boundaries, and ignores the target at each new document's first token.

The hypothesis is that packing will process more non-padding tokens per second.
Whether it improves SQL validation loss is an empirical question. The current
masked implementation builds a dense boolean attention mask in PyTorch; it is
**not** a `cu_seqlens`/FlashAttention variable-length kernel. Its speed can be
lower even if its document isolation is more faithful. Do not assume the
photographed 3–5× speedup will occur.

All runs read `uint32` tokens from `/vol/pretrain_data/` on the `sqlgen-data`
Volume and use the tokenizer saved under `/vol/tokenizer/`. The trainer requires
the corpus `metadata.json` and all four train and four validation bin files.
It never retokenizes or downloads the original Hugging Face datasets.

## First check and full comparison

Run a short check before committing GPU time to three long runs. Set the
workspace and W&B entity to your own values; the API key is already supplied
to the remote job through the existing Modal `wandb-secret`.

```bash
uv run --locked modal run --env main --profile sanika-05 \
  train_pretrain.py --mode pad --max-steps 2 --warmup-steps 1 \
  --eval-every 1 --val-batches 2 --run-name day5-pad-check \
  --wandb-entity dkbsingh7-iit-delhi
```

When that finishes, run each mode with the **same** `sql_fraction`, seed,
sequence length, batch size, learning-rate schedule, and evaluation settings.
The default schedule is `cosine`:

```bash
uv run --locked modal run --env main --profile sanika-05 train_pretrain.py --mode pad --sql-fraction 0.5 --wandb-entity dkbsingh7-iit-delhi
uv run --locked modal run --env main --profile sanika-05 train_pretrain.py --mode naive_pack --sql-fraction 0.5 --wandb-entity dkbsingh7-iit-delhi
uv run --locked modal run --env main --profile sanika-05 train_pretrain.py --mode masked_pack --sql-fraction 0.5 --wandb-entity dkbsingh7-iit-delhi
```

To compare learning-rate schedules, hold the batch mode and all other settings
fixed. For example, these two runs use the same `pad` mode and seed:

```bash
uv run --locked modal run --env main --profile sanika-05 train_pretrain.py --mode pad --lr-schedule cosine --run-name pad-cosine-seed42 --wandb-entity dkbsingh7-iit-delhi
uv run --locked modal run --env main --profile sanika-05 train_pretrain.py --mode pad --lr-schedule inverse_sqrt --run-name pad-inverse-sqrt-seed42 --wandb-entity dkbsingh7-iit-delhi
```

Both schedules warm up linearly to `peak_lr` at update 50. Cosine then gently
falls to `min_lr` at update 500. Inverse square root follows
`max(min_lr, peak_lr * sqrt(warmup_steps / update))`; it is about 0.316 times
the peak at update 500 and does **not** have to reach `min_lr` then. With no
warmup, update 1 is the anchor and starts at the peak. Plot `train/lr` beside
validation loss to see what the schedule actually did.

Both schedules use bf16 mixed precision in training **and** validation:
autocast runs eligible CUDA operations in bf16 while model parameters and
AdamW state stay float32. This path does not need gradient scaling for bf16.
The trainer checks for CUDA bf16 support before loading data.

Each command requests **one A10 GPU**, 4 CPUs, and 16 GiB container RAM, with
an upper function timeout of 24 hours. Run them one after another if GPU
availability is limited. No GPU is requested by editing or testing the code.
The default model uses six layers, width 512, eight query heads, four KV heads,
and a 1408-wide SwiGLU. The Qwen tokenizer's ~151k tokens make this about
95 million tied-weight parameters despite the historical `transformer_35m.py`
filename. The safe initial training batch is one 512-token row per microbatch,
four accumulated microbatches. The pictured 8 × 2048 setting is available via
`--micro-batch-size 8 --seq-len 2048` after measuring GPU memory; it may not
fit on an A10. Default 500 updates use 1,024,000 token slots, before padding.

`--sql-fraction 0.5` means a 50% chance that each sampled document is SQL.
Because documents have different lengths, the actual SQL token fraction can
differ; inspect `data/observed_sql_token_fraction` in W&B. The three SQL files
are selected in proportion to their sizes in tokens. For a mixing experiment,
change only `--sql-fraction` and use a **new** `--run-name` so the old optimizer
checkpoint is never reused. For example, compare 0.25 and 0.75 with the same
batching mode and seed.

The trainer logs training loss, separate English and SQL validation loss,
pre-clipping gradient norm, learning rate, raw token slots/sec, real
non-padding tokens/sec, padding percentage, cumulative non-padding tokens,
and observed SQL token fraction. Validation uses the same fixed standalone
documents every 50 updates. It saves model, optimizer, and random-generator
state to `/vol/checkpoints/day5/<run-name>/latest.pt`; rerunning the **same**
command resumes at the next update. The training comparison changes both
padding and how many source documents fit in a fixed number of token slots,
so compare validation loss against cumulative real tokens as well as updates.
