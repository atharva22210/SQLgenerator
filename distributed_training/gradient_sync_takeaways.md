# Gradient synchronization: analytical takeaways

## Question

Can synchronizing gradients once per optimizer update reduce communication cost
while preserving the accumulated update?

## Prediction

Skipping intermediate DDP reductions should reduce bucket collectives. Speedup
depends on how much communication was already hidden by backward computation;
fewer collectives do not guarantee the same percentage throughput gain.

## Controlled settings

B1 synchronizes each backward. B2 accumulates locally for seven microbatches
and synchronizes on the eighth. Both forward and backward are inside
`no_sync()` when synchronization is skipped. The historical 100-update comparison
kept model, global batch settings, seed and earlier corpus fixed. Small count
reductions still occur once per optimizer update.

## Measured results

| Method | Steady useful tokens/sec |
| --- | ---: |
| Sync every microbatch | 24,986 |
| Sync final microbatch | 30,549 |

Deferring synchronization improved measured throughput by approximately 22%.
[Recorded comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/614piai7).
The local collective-count test verifies fewer bucket collectives and compares
the accumulated gradient with the always-synchronize reference.

## Explanation

Intermediate backwards need only accumulate gradients locally. The final DDP
backward can reduce the accumulated bucket. This changes communication timing,
while the optimizer still sees the same global token-weighted gradient.

FSDP has different memory consequences: deferring reduction can retain full
gradients, while weight gathers still occur. The memory experiments therefore
used DDP `no_sync()` and left FSDP gradient reduction enabled each microbatch.

## Limitations

The timing scope excludes validation and startup. A short fixed configuration
cannot establish a universal speedup or a result on the updated corpus. Kernel
and communication overlap determine which savings translate into wall time.

## Next experiment

On the current build, repeat identical global updates with and without deferred
synchronization. Check gradients first, then profile overlap and throughput over
multiple steady updates; keep microbatch count fixed.
