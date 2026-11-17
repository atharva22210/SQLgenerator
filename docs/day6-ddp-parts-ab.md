# Day 6: DDP correctness (A) and communication (B)

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
4. `modal_ddp_parts_ab.py`: Modal allocation, torchrun, and combined W&B plots.

Python 3.13.15, locked dependencies, PyTorch 2.10.0/CUDA 12.8, one physical
machine with two A10s. BF16 compute, FP32 weights/gradients/Adam moments,
fused AdamW, masked packing, QK normalization off, seed 42. The real tokenizer
has roughly 151k entries: the smaller model is about 95M parameters, not 35M.
The B preset uses 12 layers, width 768, 12 query/4 KV heads, FFN width 2048.
Parameter counts are printed/logged from the constructed model.

## Part A

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

## Part B

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
Data startup scans the training corpus per rank; rank 0 additionally filters
validation. This can take longer than the tiny training run. Predict that
A1/A2 agree, A3's hashes repeat, and B1/B2 have close losses before running:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  modal_ddp_parts_ab.py --part a --max-steps 2 \
  --micro-batch-size 1 --seq-len 64 --wandb-entity dkbsingh7-iit-delhi

uv run --locked modal run --detach --env main --profile sanika-05 \
  modal_ddp_parts_ab.py --part b --max-steps 2 \
  --micro-batch-size 1 --seq-len 64 --wandb-entity dkbsingh7-iit-delhi
```

After checking those results, curriculum-size suites use the same commands
with `--max-steps`, `--micro-batch-size`, and `--seq-len` omitted. A defaults to
300 updates per case, B to 100. Verify memory fit first; the A microbatch=8
preset is larger than the existing Day 5 default. `--sql-fraction` controls
English/SQL document sampling probability and must match across comparisons.
Do not change several variables while interpreting a speed comparison.

## Results and boundaries

Only rank 0 logs per-update metrics to W&B (`sqlgen-day6`) and writes the final
model/optimizer snapshot. Rank 0 validates the unwrapped model while other
ranks wait; a rank-only DDP forward could hang on collectives. English/SQL
validation uses fixed standalone documents and the SQL overlap filter. The
standard validation sample is 8 documents per corpus, not the full 4,388-task
SQL evaluation; log/interpret that sampling limitation.

All cases share a unique W&B group; a comparison run adds SQL-loss and
throughput bar charts. Snapshots and JSON results are stored under
`/vol/checkpoints/day6/<suite>/<case>/`; the parent commits after worker exit.
Each invocation starts fresh with a unique suite ID. These are final snapshots,
not an automatic-resume training API. A failed case stops the suite and retains
previously committed cases. FSDP and memory-ladder experiments (Part C) are not
implemented here.

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
