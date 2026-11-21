# Distributed training takeaways

These notes cover DDP correctness, gradient synchronization, and memory fit on
two Modal A10s. The [experiment guide](README.md) describes the launch settings,
and the [memory and sharding guide](memory-and-sharding.md) describes profiling
and the Grouped SQL rerun. The GPU measurements below used the earlier corpus.

## A · DDP correctness

A1 puts the entire effective batch on one GPU across accumulation steps. A2
assigns the same draws to two ranks. A3 intentionally gives both ranks the same
draws, so an apparently healthy loss can hide duplicated data. The initial
per-rank batch hashes diagnose identical draws on that update. They do not
measure overlap between the full training and validation datasets.

The two-process correctness test uses unequal numbers of valid targets per
rank and compares gradients and an AdamW update against a single-process
reference. Local loss means must be weighted by their valid target counts.
Otherwise a rank with a short, mostly masked batch would contribute as much
as a rank with many valid targets.

In the recorded 300-update suite, A2 processed **54,076 useful tokens/sec**
across both GPUs versus **28,021** for A1, a **1.93×** gain within the timed
training update. Final sampled SQL losses were **0.433** and **0.432**
respectively. The final weight comparison reported a relative L2 difference
of 0.064, so close loss curves alone do not establish identical weights.
A3 was excluded from this combined run.
[DDP comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/614piai7)

## B · Gradient synchronization

B1 synchronizes on every backward. B2 keeps local gradients for the first
seven microbatches and synchronizes once on the eighth. Both forward and
backward run inside `no_sync()` on the skipped microbatches. The local
collective-count test verifies that deferring synchronization reduces gradient
bucket collectives while preserving the accumulated gradient. The GPU
throughput effect depends on overlap with backward computation. Small
count-reduction collectives still run once per optimizer update.

In the recorded 100-update comparison, B2 sustained **30,549 useful
tokens/sec** versus **24,986** for B1, a **22%** throughput gain on this
configuration. The runs used the same model, batch settings, and seed. The
throughput comparison covers the timed training update, with validation and
startup excluded.
[Gradient synchronization comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/614piai7)

For FSDP, deferring gradient reduction holds full gradients between
microbatches and still requires weight gathers. The memory experiment used
DDP `no_sync()` and kept FSDP gradient reduction enabled on every microbatch.

## C · Memory and sharding

C3 used 24 blocks at width 1,536 and 855,894,528 total parameters with the
151,670-token vocabulary. The full workload used batch 4 per GPU, sequence
length 2,048, and two accumulation microbatches. The A10s reported 22.08 GiB
of usable memory per GPU.

| Method | Full-context result | Short-context peak per GPU | Short-context useful tokens/sec |
| --- | --- | ---: | ---: |
| DDP baseline | OOM on update 1 | 18.58 GiB | 427 |
| DDP + checkpointing 12 blocks | OOM on update 2 | 17.93 GiB | 402 |
| FSDP `SHARD_GRAD_OP` (ZeRO-2 style) | OOM on update 1 | 12.23 GiB | 330 |
| FSDP `FULL_SHARD` (ZeRO-3 style) | OOM on update 1 | 9.97 GiB | 269 |
| FSDP `FULL_SHARD` + checkpointing 12 blocks | **Completed 20 updates** | **9.34 GiB** | 265 |

The short-context workload used batch 1 per GPU, sequence length 64, and three
updates. Its throughput values come from one timed update. They show the
observed ordering at that size. They do not establish the same ordering at
2,048 tokens, where only one method fit. The full-context successful run
peaked at **18.69 GiB allocated per GPU** and averaged **4,676 useful
tokens/sec** over 14 timed updates, excluding warmup and profiling.
[Short-context comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/s37qyy4b) ·
[Full-context comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/0ii241bf)

- **Why partial checkpointing helped DDP but still OOMed.** It finished update
  1, then failed on update 2. The second update had to run with AdamW moments
  already allocated. The logs don't isolate which live allocations drove the
  peak.
- **Why gradient and optimizer sharding still OOMed.** `SHARD_GRAD_OP` retains
  gathered weights from forward through backward. It failed in a SwiGLU
  forward allocation on update 1, before AdamW had allocated its moments.
- **Why full sharding still needed checkpointing.** `FULL_SHARD` releases
  gathered block weights after forward and gathers them for backward.
  Saved activations can still be live alongside gathered weights and
  temporary buffers. Plain `FULL_SHARD` failed during backward while
  requesting 298 MiB with 21.27 GiB already allocated. Checkpointing 12 blocks
  made the same workload fit. That establishes the effect of checkpointing,
  while the exact activation share of the peak was not measured.

Our earlier 12 GB estimate for activations scaled up a hypothetical 3 GB C1
budget. It was not measured. A single BF16 SwiGLU intermediate at this shape
is 64 MiB, and backward can depend on multiple intermediates per block. An
ideal sharded-state budget also omits the temporary weight gathers, loss
buffers, and other allocations that can overlap with saved activations.

## What carries over to Grouped SQL

The communication patterns and parameter-count formulas carry over. Dataset
quality itself does not enter the memory budget. At the same sequence length,
a different tokenizer vocabulary changes the embedding and vocabulary-loss
sizes, while a different packed document-length distribution can change
attention kernel shapes, throughput, and short-lived allocations. The
observed OOM boundary is therefore a prediction to verify on the pinned
Grouped SQL build.

The next run is the full-shard checkpointed case against that confirmed
build. Add its memory, throughput, and sampled SQL validation curve here
once measured. Keep the earlier corpus's results as the historical baseline.
