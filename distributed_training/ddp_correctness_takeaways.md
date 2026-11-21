# DDP correctness: analytical takeaways

## Question

Does distributing the same global batch preserve the update, and can duplicate
rank data produce a plausible loss curve while training on the wrong batch?

## Prediction

A correctly weighted two-rank update should agree closely with a single-process
reference. Repeating the same data on both ranks should not be treated as a
larger effective batch. Floating-point reduction order may produce drift over
many BF16 updates even when the single-update math is correct.

## Controlled settings

A1 is the single-GPU reference; A2 partitions the same global draws across two
ranks; A3 deliberately duplicates draws. Model seed and global sample IDs are
controlled. Each rank's loss is weighted by its valid-target count, then DDP's
averaging is accounted for. The historical combined suite ran A1/A2 for 300
updates on the earlier corpus; A3 was excluded. Its results do not describe the
new flat-layout dataset or current defaults.

## Measured results

| Case | Timed global useful tokens/sec | Final sampled SQL loss |
| --- | ---: | ---: |
| A1 single reference | 28,021 | 0.432 |
| A2 two ranks, distinct draws | 54,076 | 0.433 |

A2 achieved 1.93× throughput within the timed training update. The final weight
comparison reported relative L2 difference 0.064.
[Recorded comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/614piai7).
CPU two-process tests separately compare gradients and one AdamW update with
unequal valid-target counts across ranks.

## Explanation

DDP averages gradients, not input data or target counts. Averaging local loss
means equally would give a short masked batch as much weight as a long one.
The explicit global target denominator makes the distributed gradient represent
the global token mean. Per-rank first-update hashes expose accidental duplicate
draws even when loss looks healthy.

## Limitations

Nearly equal final losses do not establish identical final weights; the measured
weight drift is material evidence of that distinction. A first-update hash does
not audit overlap across entire datasets. Timings exclude startup, validation,
and checkpoint IO. There is no repeated-seed throughput uncertainty estimate.

## Next experiment

Repeat the short single-update CUDA comparison on the pinned current build,
inspect rank draw IDs, then compare longer trajectories using both weight and
loss differences. Preserve the same global draws when changing world size.
