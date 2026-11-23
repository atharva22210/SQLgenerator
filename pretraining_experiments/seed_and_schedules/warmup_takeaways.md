# Warmup duration follow-up

## Question and prediction

Is WSD loss higher than cosine during the stable phase after 300-update warmup?
We will measure the sign of `WSD loss − cosine loss` on updates 301–800. A positive
value means WSD is higher. Either ordering is a valid experimental result.
Longer warmup does not guarantee an ordering or greater stability.

## Controlled settings

Only two new runs: cosine seed 1 and WSD linear seed 1. Both use 300 warmup
updates, 1000 total updates, 4096 context, batch 8 × accumulation 4, 131.072M input
tokens, peak LR 6e-4, minimum LR 6e-5, and validation seed 12345. The other model,
optimizer and validation settings remain identical to the original suite.

WSD uses warmup 1–300, stable 301–800 (500 updates), and decay 801–1000 (200).
Its stable fraction changes to 0.5 so the fixed budget is preserved. Cosine decays
from the peak after update 300 through update 1000. The original five-run plan
and measured results stay unchanged.

Reuse the completed cosine seed 1 and WSD linear seed 1 from suite
`day07-full-2a6757d322c4`. The coordinator checks their configurations and published
dataset identity before launching new cases. If the flat Modal dataset has since
changed, it fails rather than comparing different datasets or silently rerunning
baselines. Restore/access the matching build before proceeding.

## Measured results

**Not run.** This change prepares the follow-up and runs local tests only.
No GPU smoke, training, or online W&B job is part of this implementation.

## Explanation to test

Report the stable-region mean gap, every pointwise gap, and above/below/equal
counts for training loss and both validation losses. The ten validation points
are updates 350,400,…,800; exclude warmup endpoint 300 and all decay updates.
Compare with the gap at update 300 to expose pre-existing numerical drift.
A positive average does not imply WSD is above cosine at every point.
Serial evaluation points are not independent replications.

Inspect pre-clipping gradient norms, clipping frequency and fixed-sample validation
loss at equal updates/tokens. More gradual warmup changes the state entering the
stable/decay phases. It cannot guarantee the model will settle earlier or that
WSD will have higher stable-phase loss. Compare each new run against its own
schedule's baseline, then compare schedules.

## Limitations

Only one seed per new configuration. Same-seed GPU drift occurred during the
original identical warmup; small differences need replication. Cosine seed 1
also replayed updates after checkpoint 600, so cost is not controlled by final
history length alone. Validation is sampled next-token loss, not SQL accuracy.

## Next experiment and execution

Preview locally without allocating hardware:

```bash
uv run --locked python -m pretraining_experiments.seed_and_schedules.launch \
  --dry-run --warmup-comparison
```

Future launch, only when compute is available:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.seed_and_schedules.launch \
  --warmup-comparison --wandb-entity dkbsingh7-iit-delhi
```

This submits only the two new cases, not the original five. New run labels include
`warmup300`, and a separate suite ID isolates checkpoints. The comparison run
combines both saved baselines and new results, plotting LR, training/validation
loss and pre-clipping gradient norm. The generated report retains the measured
values and limitations. No early-decay or mid-run branch experiment is implied.
