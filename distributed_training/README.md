# Distributed training experiments

**A · DDP correctness** compares the single-GPU reference with distinct
two-GPU batches and an intentional duplicate-data case. **B · Gradient
synchronization** compares syncing every microbatch with syncing on the last
microbatch. **C · [Memory and sharding](memory-and-sharding.md)** compares DDP
and FSDP with and without activation checkpointing. See the [experiment
takeaways](takeaways.md) for measured results and their limits.

The question is whether splitting the SAME global update across two GPUs
preserves its gradient, and how much communication `no_sync()` avoids.
Prediction: A1/A2 should agree within numerical tolerance; A3 repeats draws
and is not an equivalent effective batch. B2 should communicate less than B1,
but speedup depends on computation/communication overlap. Loss need not expose
A3's bug, so the first update also reports per-rank batch hashes.

## Read the files in this order

1. `distributed_training/experiment_plan.py`: names and controlled variables.
2. `distributed_training/ddp_core.py`: rank assignment and loss normalization.
3. `distributed_training/ddp_worker.py`: one process's training loop.
4. [launch.py](launch.py): Modal allocation, torchrun, and combined W&B plots.

Python 3.13.15, locked dependencies, PyTorch 2.10.0/CUDA 12.8, one physical
machine with two A10s. BF16 compute, FP32 weights/gradients/Adam moments,
fused AdamW, masked packing, QK normalization off, seed 42. The real tokenizer
has roughly 151k entries: the smaller model is about 95M parameters, not 35M.
The B preset uses 12 layers, width 768, 12 query/4 KV heads, FFN width 2048.
Parameter counts are printed/logged from the constructed model.

## Part A · DDP correctness

| Case | GPUs used | Microbatch per GPU | Accumulation | Sequences/update |
| --- | ---: | ---: | ---: | ---: |
| A1 single reference | 1 | 8 | 8 | 64 |
| A2 DDP distinct draws | 2 | 8 | 4 | 64 |
| A3 deliberately repeated draws | 2 | 8 | 4 | 64 occurrences, repeated across ranks |

Microbatch IDs are global within each update, and each ID has a deterministic
RNG based on `(seed, update, ID)`. A1 processes IDs 0..7; A2 splits them between
ranks; A3 gives both ranks IDs 0..3. These are draws from random document
sampling, not a guarantee of unique examples or disjoint underlying datasets.
For A1/A2, different rank assignment does not change the global examples or
packing boundaries. First-update hashes diagnose identical rank data.
Final weights are compared with maximum absolute and relative L2 differences.
Long BF16 trajectories may diverge slightly; local two-process tests compare
one update tightly, including AdamW and checkpointed vocabulary loss.

The true gradient objective is a mean over valid next-token targets across
ALL ranks and accumulation steps. If a local mean is L over n targets and the
global count is N, backward uses `L * n * world_size / N`. DDP's averaging
cancels `world_size`. Dividing by accumulation again would be a bug. Clip only
after the last synchronized backward. Count collectives still occur once per
update, even with no_sync enabled.

## Part B · Gradient synchronization

B1 and B2 use two GPUs, microbatch=4, accumulation=8, 100 updates. Only
`no_sync` changes. B1 synchronizes gradients every backward; B2 skips gradient
synchronization on microsteps 1..7 and synchronizes on step 8. Both forward and
backward are inside the context. DDP buckets/reduces gradients asynchronously
in production; the test-only communication hook counts buckets synchronously.

Useful-token throughput is GLOBAL summed non-padding tokens divided by the
slowest rank's update duration. The timed region includes transfers,
forward/backward, valid-count collective, clipping and AdamW. It excludes CPU
sampling, the starting barrier, metric reductions, W&B, validation and disk IO.
Thus it measures update throughput, not full job wall time. First 10 updates
(or all but the last for shorter runs) are excluded from the steady summary.
Memory metrics are the maximum per-rank allocated/reserved peak, not a sum
that pretends two 24 GB devices form one 48 GB memory space.

## Small initial GPU checks

These are correctness checks, not benchmarks. Each command allocates TWO A10s
for the suite. A1 uses only GPU 0, so GPU 1 is billed while idle for that case.
Data startup indexes the training corpus per rank. The completed Grouped SQL
build supplies an audited validation split. This can take longer than the tiny training run. Predict that
A1/A2 agree, A3's hashes repeat, and B1/B2 have close losses before running:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m distributed_training.launch --part a --max-steps 2 \
  --micro-batch-size 1 --seq-len 64 --wandb-entity dkbsingh7-iit-delhi

uv run --locked modal run --detach --env main --profile sanika-05 \
  -m distributed_training.launch --part b --max-steps 2 \
  --micro-batch-size 1 --seq-len 64 --wandb-entity dkbsingh7-iit-delhi
```

After checking those results, curriculum-size suites use the same commands
with `--max-steps`, `--micro-batch-size`, and `--seq-len` omitted. A defaults to
300 updates per case, B to 100. Verify memory fit first; the A microbatch=8
preset is larger than the packing experiment default. `--sql-fraction` controls
English/SQL document sampling probability and must match across comparisons.
Do not change several variables while interpreting a speed comparison.

## Results and boundaries

Only rank 0 logs per-update metrics to W&B (`sqlgen-distributed-training`) and writes the final
model/optimizer snapshot. Rank 0 validates the unwrapped model while other
ranks wait; a rank-only DDP forward could hang on collectives. Validation uses the audited SQL split: a fixed sample of 128 standalone documents every 25 updates
by default, plus a complete pass over all retained tasks at the final update.
The final full-set metric is `val/sql_loss_filtered_all`, distinct from the
periodic sampled `val/sql_loss`. The W&B comparison run overlays those sampled
points under `comparison/sql_loss_curve`. Set `--eval-every` and `--val-batches`
to change the cadence and sample size. No English validation loss is computed.

W&B groups each launch by experiment topic, Grouped SQL build ID prefix, sequence
length, and launch time. Run names and chart legends identify the method. Each
training run includes `charts/training_loss` and `charts/sql_validation_loss`.
A comparison run adds SQL-loss and
throughput bar charts. Snapshots and JSON results are stored under
`/vol/checkpoints/distributed_training/<suite>/<case>/`; the parent commits after worker exit.
Each invocation starts fresh with a unique suite ID. These are final snapshots,
not an automatic-resume training API. An unexpected worker error stops the
suite and retains previously committed cases. Part C also records CUDA OOMs
and continues to the next method.

## Local correctness tests

```bash
uv run --locked python -m unittest discover -s tests
```

The DDP test launches two tiny CPU/Gloo processes, uses unequal target counts,
compares gradients and AdamW updates with one-process reference, and verifies
no_sync changes gradient bucket communication from twice to once. Existing
CUDA-only tests are skipped without CUDA. Passing CPU tests does not establish
NCCL/BF16 performance or actual two-A10 correctness; use the small GPU checks
above before longer experiments.

## Spawn A1, A2, B1, B2 sequentially

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m distributed_training.launch --part ab --wandb-entity dkbsingh7-iit-delhi
```

The local entrypoint calls `run_suite.spawn()` and prints a call ID, then
returns. The detached suite continues remotely. A3 is excluded. Steps default
to 300, 300, 100, 100. Every suite first runs a tiny two-GPU NCCL/BF16 test of
checkpointed-loss gradients with and without no_sync, plus finite fused AdamW
updates. Failure aborts before the long runs. This gate tests kernels and
communication, not full-size batch memory fit. Errors in any case stop later
cases. Check each rank-0 W&B run and the final comparison run under the shared
suite group; the parent commits each completed checkpoint to the Volume.

## Grouped SQL corpus

The grouped SQL job publishes a versioned overlay with `sql_train.bin`,
`sql_val.bin`, and completion metadata. Distributed training reuses the prepared English files.
The packing comparison, its GPU check, and the distributed experiments use this corpus for SQL. Earlier SQL
files may still appear at the volume root as historical artifacts.

The job uses the [Parquet edition of SynSQL-2.5M](https://huggingface.co/datasets/iNeil77/SynSQL-2.5M)
and the [StarCoderData SQL directory](https://huggingface.co/datasets/bigcode/starcoderdata).
SynSQL rows contain schema, question, answer, and any supplied external context. Repository rows contain raw
SQL. Accept StarCoderData's access terms on Hugging Face and create the Modal
secret `huggingface-secret` with `HF_TOKEN` through the provider interface.

```bash
uv run --locked modal run --detach -m data.prepare_grouped_sql --submit-only
```

The submission prints the corpus directory. After the split audit,
`examples.json` contains up to five SynSQL and five repository documents per
split, drawn from different groups within each source. Long examples include
a marked excerpt. The binaries are ready for training after the job publishes
`metadata.json`.

The launcher pins the latest completed compatible build for the entire suite. To
reproduce a specific build, pass
`--grouped-sql-dir /vol/pretrain_data/grouped_sql/<corpus_id>`.

Preprocessing stages text and normalized SQL hashes in local SQLite scratch
space. Four CPU processes fingerprint SQL by default. A single writer consumes
their results in source order so deduplication stays deterministic. Use
`--fingerprint-workers 1` for the serial baseline. The job reports average CPU
use for staging and token counts as it writes each split. It strips leading
StarCoderData provenance tags before hashing and tokenizing repository SQL. It deduplicates normalized statements or scripts, then holds
out about 10% of the remaining database IDs and repository names separately.
A held-out script with a statement also present in training is dropped whole.
The audit records assigned and retained group counts, discarded documents,
and disjointness for both group types and statement hashes before tokenization
opens either output file. The audit uses the recorded repository provenance and
lexical SQL fingerprints. Semantic equivalents, forks with different recorded
names, and overlap with the English corpus are outside that check.

The final SQL loss now covers a mix of held-out text-to-SQL prompts and raw SQL
from held-out repositories when an overlay is selected. It remains a
token-weighted next-token metric. A full validation pass scores one
context-length prefix per retained document, which can be much larger than the
legacy validation pass.
