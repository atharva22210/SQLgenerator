# Sequence packing on one GPU

Use [launch.py](launch.py) to run the packing comparison and [check.py](check.py)
for the bounded CUDA correctness check. The W&B project is `sqlgen-packing`.
Run names spell out the packing method, learning-rate schedule, and seed.
See [Experiment takeaways](takeaways.md) for the comparison and its evidence.

Fresh runs accept `--initializer-range` for the normal weight standard deviation;
the default remains `0.02`. See [weight initialization](../../model/initialization.md)
for the SwiGLU reference, fan-in distinction, and checkpoint compatibility.

Packing runs use FineWeb-Edu for English and the latest completed
[Grouped SQL build](../../distributed_training/README.md#grouped-sql-corpus) for SQL training
and validation. Resuming a Grouped SQL checkpoint keeps its original build.
Checkpoints from the earlier SQL sources require a new run name. New runs
use `/vol/checkpoints/packing`. Earlier snapshots and W&B runs are still
available at their original locations.


The question is how much padding costs, and whether isolating packed documents
changes validation loss. The baseline is `pad`: one document per row, with
padding targets ignored. `naive_pack` joins documents under ordinary causal
attention. `masked_pack` resets positions, blocks attention across document
boundaries, and ignores the target at each new document's first token.

The hypothesis is that packing will process more non-padding tokens per second.
Whether it improves SQL validation loss is an empirical question. The current
CUDA path uses PyTorch 2.10's public
[`varlen_attn`](https://pytorch.org/blog/pytorch-2-10-release-blog/), backed by
FlashAttention-2, for all three modes. The loader computes int32 cumulative
sequence lengths (`cu_seqlens`), unpadding indices, and maximum segment length
on CPU. Padding never enters attention. `pad` and `naive_pack` use one segment
per row; `masked_pack` uses one segment per document. Thus naive packing keeps
its intentional cross-document attention. CUDA training builds no dense 3D
mask. The CPU path retains a dense reference for correctness tests.

Training and validation now take the **first slice** of each sampled document,
including a document truncated to fill the end of a packed row. This preserves
its prompt/schema prefix but excludes tails beyond the available context;
it is a sampling-policy change, not a claim that random crops always train
worse. Use fresh run names when comparing with the previous random-crop
trainer. Default names include `prefix` to avoid resuming the old defaults.

All runs read `uint32` tokens from `/vol/bpe_tokenization/pretrain_data/` on the `sqlgen-data`
Volume and use the tokenizer saved under `/vol/bpe_tokenization/tokenizer/`. The trainer requires
the corpus `metadata.json` and all four train and four validation bin files.
It never retokenizes or downloads the original Hugging Face datasets.

## First check and full comparison

Run a short check before committing GPU time to three long runs. Set the
workspace and W&B entity to your own values; the API key is already supplied
to the remote job through the existing Modal `wandb-secret`.

The bounded correctness runner first executes the model tests, including
actual CUDA varlen attention/gradient comparisons for all three modes. If
they pass, it trains two `masked_pack` updates with the A10 defaults, evaluates
and checkpoints each update, then checks that saved model/optimizer tensors
are finite. It creates a unique run name and has a 15-minute function timeout:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.sequence_packing.check::main --wandb-entity dkbsingh7-iit-delhi
```

To run just the ordinary trainer for two updates instead:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.sequence_packing.launch --mode pad --max-steps 2 --warmup-steps 1 \
  --eval-every 1 --val-batches 2 --run-name packing-pad-check \
  --wandb-entity dkbsingh7-iit-delhi
```

The entrypoint calls `.spawn()`, prints the call ID and checkpoint path, and
exits without waiting. `--detach` is required to keep this ephemeral Modal App
alive after the local launcher exits. These serve different purposes; see
[Modal App lifetimes](https://modal.com/docs/guide/apps) and
[job submission](https://modal.com/docs/guide/job-queue). Monitor the App in
the Modal dashboard and metrics in W&B. Add `--wait` only when you want the
launcher to wait for and print the final result. The submission ID can also
retrieve the result later, in the same Modal profile/environment:

```python
import modal

call = modal.FunctionCall.from_id("fc-YOUR_CALL_ID")
result = call.get(timeout=0)  # Raises TimeoutError while still running.
```

When that finishes, run each mode with the **same** `sql_fraction`, seed,
sequence length, batch size, learning-rate schedule, and evaluation settings.
The default schedule is `cosine`:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 -m pretraining_experiments.sequence_packing.launch --mode pad --sql-fraction 0.5 --wandb-entity dkbsingh7-iit-delhi
uv run --locked modal run --detach --env main --profile sanika-05 -m pretraining_experiments.sequence_packing.launch --mode naive_pack --sql-fraction 0.5 --wandb-entity dkbsingh7-iit-delhi
uv run --locked modal run --detach --env main --profile sanika-05 -m pretraining_experiments.sequence_packing.launch --mode masked_pack --sql-fraction 0.5 --wandb-entity dkbsingh7-iit-delhi
```

To compare learning-rate schedules, hold the batch mode and all other settings
fixed. For example, these two runs use the same `pad` mode and seed:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 -m pretraining_experiments.sequence_packing.launch --mode pad --lr-schedule cosine --run-name pad-prefix-cosine-seed42 --wandb-entity dkbsingh7-iit-delhi
uv run --locked modal run --detach --env main --profile sanika-05 -m pretraining_experiments.sequence_packing.launch --mode pad --lr-schedule inverse_sqrt --run-name pad-prefix-inverse-sqrt-seed42 --wandb-entity dkbsingh7-iit-delhi
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
filename. Defaults are **4 rows × 2048 tokens per microbatch**, **2 accumulation
steps**, and **128 positions per row per loss chunk**. Each optimizer update
uses 16,384 token slots; 500 updates use 8,192,000 slots before padding. Loss
is still averaged over real next-token targets across both microbatches, not
averaged equally over rows with different padding. These settings target the
A10. They fit in the correctness run recorded below; sustained utilization
and throughput still need a longer benchmark before scaling duration.

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
state to `/vol/checkpoints/packing/<run-name>/latest.pt`; rerunning the **same**
command resumes at the next update. The training comparison changes both
padding and how many source documents fit in a fixed number of token slots,
so compare validation loss against cumulative real tokens as well as updates.

## Implementation checks and remaining costs

RoPE cosine/sine tables are nonpersistent model buffers, so checkpoints retain
their parameter keys. Each forward looks up positions once and shares the
rotations across layers and Q/K heads. Explicit masks on the reference path
are combined with causality once per forward; the varlen kernel uses its own
causal flag. Training supplies the loader's CPU-checked target count to the
chunked loss and keeps detached loss scalars on CUDA. There are no scalar
device reads between microbatches. The trainer copies loss and gradient norm
together once after each optimizer update for logging and completed-work
timing. Evaluation copies one scalar per corpus; checkpoint writes also
necessarily wait for device state. Tensor transfers use pinned CPU storage
and nonblocking copies.

CUDA AdamW uses `fused=True`: fewer optimizer kernels, with the same AdamW
formula and hyperparameters. Floating-point reduction/operation order can
change, so identical bits across optimizer or attention backends are not
promised. Output-loss activation checkpoints skip RNG-state preservation
because the recomputed linear projection and cross-entropy contain no random
operations. Re-enable preservation if adding dropout or sampling **inside**
that chunk. Saved training checkpoints still include Python, CPU PyTorch,
and CUDA RNG states for resuming.

Local checks, using Python 3.13 and the locked PyTorch 2.10.0 environment:

```bash
uv run --locked python -m unittest discover -s tests
```

Tests compare reference and layout loss/parameter gradients, target-weighted
gradient accumulation, prefix sampling, padding and document boundaries,
RoPE/mask reuse, and submission without waiting. The CUDA-only test compares
actual bf16 varlen/GQA forward and backward against dense math in every mode
and checks for scalar reads. It is skipped on CPU and should pass in the
remote GPU environment before a long run. The Modal image keeps the same
Python 3.13.15 / locked CUDA 12.8 PyTorch environment; no separate `flash-attn`
installation or dependency changes are required.

Remaining costs to profile:

- The ~151k vocabulary makes projection/cross-entropy and the tied embedding
  expensive. Chunk checkpointing saves memory by recomputing projections in
  backward; 128 is a memory/throughput tradeoff, not a free speedup.
- Padding is removed from attention, but projections, normalization, feed-forward
  layers, and vocabulary loss still process padded slots. Fully unpadding the
  whole transformer/loss would require another focused change.
- CPU sampling/pinning currently runs synchronously between updates; there is
  no worker prefetch pipeline. Profile CPU/GPU idle time before adding one,
  since prefetching must preserve the sampler's checkpoint/resume position.
- Every startup scans the token files to rebuild EOS/document offsets. Saving
  and validating an index during preparation could avoid reading the entire
  roughly 40 GB corpus while the allocated GPU waits for CPU data setup.
- Per-update logging still requires one device-to-host read. Less frequent
  logging could amortize that boundary. Validation and Volume checkpoint
  saves are excluded from the reported training-update throughput.

The two-update check is not a throughput benchmark. For an eventual speed
comparison, hold sampled batches, target normalization, dtype, hardware, and
model fixed and compare the dense and varlen backends separately from the
batch-size/sampling changes. Predict the result first: lower attention memory
and less synchronization are expected, but vocabulary work may dominate.
Measure steady-state effective tokens/sec, peak GiB, scalar reads, loss and
gradient agreement; record cold-start time separately. Bound the first check
to one A10 and two updates as above before considering longer runs.

## A10 correctness result: 15 November 2026

The detached correctness run completed successfully on one NVIDIA A10 with
PyTorch 2.10.0+cu128, bf16 compute, and float32 parameters. All 11 model tests
passed, including real varlen/GQA loss and gradient comparisons against dense
math for all three modes, plus the scalar-read profiler assertion. Training
then completed two `masked_pack` updates using 4 × 2048-token microbatches,
two accumulation steps, and loss chunks of 128. Both updates validated and
committed checkpoints; reloading the final checkpoint found finite model and
optimizer tensors. It contains 95,356,416 model parameters.

| Update | Train loss | English val loss | SQL val loss | Peak allocated GiB | Peak reserved GiB |
| --- | --- | --- | --- | --- | --- |
| 1 | 12.0364 | 11.8693 | 11.3270 | 3.156 | 3.738 |
| 2 | 11.6044 | 11.7378 | 11.2171 | 3.873 | 4.627 |

The run processed 32,768 non-padding tokens total. The reported training-only
rates were about 25.0k and 26.7k tokens/sec; these exclude data startup,
validation, checkpoint commits, and logging. Two updates do not establish
sustained throughput or convergence. Memory fit matched the expectation,
with substantial headroom rather than evidence of full GPU utilization.

Run: `a10-correctness-20261115T131212Z-4d56aee1`;
call: `fc-01M7FNCEBGP8HD7KDKEGXN87J0`.
[W&B metrics](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day5/runs/11d62c678d0247bcbacefb2e07723348),
[Modal run](https://modal.com/apps/sanika-05/main/ap-vSKzGys4EVCPHxtWGL6qPQ).
Historical checkpoint: `/vol/checkpoints/day5/a10-correctness-20261115T131212Z-4d56aee1/latest.pt`.

Earlier attempts exposed two trainer integration issues, fixed before this
successful run: completion validation now prefers the v2 per-source marker
over stale legacy split snapshots while checking identity/counts, and W&B run
IDs use a UUID rather than the removed `wandb.util.generate_id` helper. Dataset
files were not changed. Local regression checks ran 73 tests: 72 passed and
the CUDA-only test was skipped locally; that test passed on the A10 above.
