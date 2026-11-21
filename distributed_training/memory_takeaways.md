# Memory and sharding: analytical takeaways

## Question

Which memory state prevents the large model from fitting, and which combination
of sharding and activation recomputation makes the fixed workload fit?

## Prediction

Sharding reduces replicated parameter, gradient, or optimizer storage. Saved
activations, temporary buffers and gathered parameters can still exceed VRAM.
Checkpointing trades repeated computation for fewer saved activations. These
predict directions, not an exact OOM boundary.

## Controlled settings

The historical workload used 24 layers, width 1536, the earlier corpus/tokenizer,
two A10s, batch 4 per GPU, sequence length 2048 and accumulation 2. A shorter
batch-1/length-64 check measured a different workload. Wrapper and checkpoint
choices varied; keep those two workload sizes distinct when interpreting results.

## Measured results

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

## Explanation

The successful full-context case combined full sharding with checkpointing.
Neither state-sharding formulas nor a single activation tensor determines peak
memory: temporary weight gathers and backward buffers can coexist with saved
activations. A memory optimization can make a run feasible while reducing its
short-context throughput; fitting and speeding up are different measurements.

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

## Limitations

The short-context throughput comes from one timed update. The earlier 12 GB
activation estimate was hypothetical. The precise allocation responsible for
every OOM was not isolated. Historical OOM boundaries do not transfer directly
to the 32K vocabulary, current document-length distribution, or 4096 context.

## Next experiment

Run the short correctness check first, then measure the full-shard checkpointed
case on one pinned current build. Report per-rank allocated/reserved peaks,
throughput over multiple steady updates, and sampled validation loss. Compare
with the historical baseline while identifying every changed workload dimension.
