# Pretraining experiments

Read an experiment's question and prediction before launching it. Its takeaway
file separates measurements from explanations and uncertainty. Model code stays
in `model/`, corpus preparation/sampling in `data/`, and experiment launchers and analysis
in this package. Python 3.13.15 and the committed `uv.lock` define the environment.

| Experiment | Code / execution guide | Analytical takeaways |
| --- | --- | --- |
| Optimizer, batch size, and peak LR | [Design and commands](optimizer_batch_lr/README.md) | [Takeaways](optimizer_batch_lr/takeaways.md) |
| Sequence packing | [sequence_packing/launch.py](sequence_packing/launch.py), [commands](sequence_packing/README.md) | [Packing](sequence_packing/takeaways.md) |
| DDP correctness | [distributed_training/experiment_plan.py](../distributed_training/experiment_plan.py), [commands](../distributed_training/README.md) | [DDP correctness](../distributed_training/ddp_correctness_takeaways.md) |
| Gradient synchronization | [distributed_training/ddp_core.py](../distributed_training/ddp_core.py), `launch --part b` | [Gradient sync](../distributed_training/gradient_sync_takeaways.md) |
| Sharding and activation memory | [distributed_training/fsdp_checkpointing.py](../distributed_training/fsdp_checkpointing.py), [commands](../distributed_training/memory-and-sharding.md) | [Memory/sharding](../distributed_training/memory_takeaways.md) |
| Seed variation and learning-rate schedules | [Schedule launcher](seed_and_schedules/launch.py), [seed cases](seed_and_schedules/seed_noise.py) | [Results and smoke check](seed_and_schedules/takeaways.md) |

After training, use the [held-out SQL generation evaluator](sql_generation_eval/README.md) to export checkpoint predictions for judging.

Experiment launchers supply settings to [`training/trainer.py`](../training/trainer.py),
the shared single-GPU loop. The [`training/`](../training/) package also handles
configuration, data loading, schedules, checkpointing, dataset identity, and
W&B tracking. The distributed experiments retain their
rank-specific collective loop in `../distributed_training/ddp_worker.py`.

```text
training/
  trainer.py                     # shared single-GPU training loop
pretraining_experiments/
  sequence_packing/
    launch.py
    check.py
    README.md
    figures/
    takeaways.md
  seed_and_schedules/
    launch.py
    seed_noise.py
    analysis.py
    figures/
    takeaways.md
    validation_results.csv
distributed_training/            # earlier DDP/FSDP experiments
  launch.py
  ddp_worker.py
  ...
```

Already running Modal jobs continue using the source uploaded when they launched.

## Writing an experiment report

Keep one `takeaways.md` alongside the experiment code. Use the experiment's
name in the report, plots, and run labels so readers can find the evidence later.
Write for a teammate seeing the experiment for the first time.

```markdown
# <Experiment name>

## Why we ran it
## What we ran
## Sanity checks
## Results and interpretation
## Additional observations (optional)
## Conclusion
### What the experiment showed
### What remains uncertain
## Next experiment
## Sources
```

Start with why the question matters to the project and what you predicted before
the run. Describe what varied, what you held fixed, and the training and evaluation
budgets. In the sanity checks, compare the actual run with the expected schedule,
budget, or metrics where those checks apply.

Order results so readers have the reference points they need. For example, show
how much runs vary across seeds before using that spread to interpret a schedule
gain. Walk through the phases when timing helps explain the result. Put charts
beside the claims they support, label smoothing, and use tables for exact values
such as individual seed results and their sample spread.

In the conclusion, say what the results establish under the tested settings
and what decision that supports. If the results are inconclusive, say so plainly.
Explain what remains uncertain and which additional measurements would address it.
Link the original runs or data. Adapt the headings to the experiment and use an
additional observations section for findings outside the main comparison.

## Batch size and learning-rate scaling

The [fixed-token sweep](batch_lr_scaling/README.md) tests four batch sizes × four
learning rates at 131.072M training tokens per case. Its implementation is ready;
training has not been launched. Preview locally without allocating hardware.
Takeaways live [beside the experiment](batch_lr_scaling/takeaways.md).

## Completed seed and schedule results

Read the [detailed measured analysis](seed_and_schedules/takeaways.md), including
all five final losses, curve phases, seed variation, and reproducibility limits.
Both WSD runs beat all three cosine runs on the fixed validation sample; the
WSD-tail comparison remains inconclusive.

## Seed and schedule comparison

Question: how much variation do seeds produce, and does WSD improve loss at the
same training budget? Predict whether the stable phase keeps WSD loss higher
than cosine, whether it falls during the decay tail, and how large the seed
standard deviation will be. These are hypotheses, not guaranteed outcomes.

Five independent single-A10 runs execute in parallel:

| Case | Seed | LR after warmup |
| --- | ---: | --- |
| Cosine × 3 | 1, 2, 3 | Cosine decay through update 1000 |
| WSD linear | 1 | Stable through 800; linear decay over 801–1000 |
| WSD cosine | 1 | Stable through 800; cosine decay over 801–1000 |

All use 6 layers, width 512, FFN 1408, GQA 8 query/4 KV heads, QK norm disabled,
normal initialization std 0.02, masked packing, BF16 compute/FP32 parameters,
4096 context, microbatch 8, accumulation 4, and 1000 optimizer updates. AdamW:
peak LR 6e-4, minimum 6e-5, betas (0.9,0.95), weight decay 0.1, gradient clip 1.
Warmup lasts 100 updates. WSD stable/decay fractions are 0.7/0.2. Decay reaches
minimum on the final update; its first update is already below peak.

Validation uses 128 standalone documents per corpus every 50 updates, with
validation seed 12345 fixed across cases. Training seed controls both initial
weights and sampling. The SQL/English mix remains a 0.5 document sampling
probability, not a guaranteed 50% token fraction. The shared loop logs the
observed fraction and weights accumulated losses by actual predicted targets.

**4096 is the sole change to the screenshot's fixed settings.** At unchanged
8 × 4 batching, each update has 131,072 token slots: 131,072,000 per run,
655,360,000 for the suite, plus validation. Masked document boundaries reduce
predicted-target counts. Five A10s are billed concurrently; startup scans,
validation, and checkpoint IO add time. Throughput from the old tokenizer/corpus
is not a runtime guarantee. The full batch is never silently reduced to fit.

## Updated Modal data

The coordinator resolves the current completed dataset once from
`sqlgen-data:/bpe_tokenization/pretrain_data`, accepting the new flat layout.
It validates English/SQL tokenizer identity, completion audits, and binary
lengths, then passes the same manifest fingerprint/corpus ID to every worker.
Workers fail if the published build changes. Runs, checkpoints, and analysis
record this identity. This is a build/manifest fingerprint, not a new hash over
every GB of binary tokens. Do not replace the flat corpus while a suite runs.

The dataset inspected on 2026-11-22 has SQL corpus ID
`ff43d7752a904b649fad5b2d2dbb4df3`, vocabulary 32768, English train tokens
9,238,968,483 and SQL train tokens 7,830,836,989. This documents what was inspected;
future launches resolve and pin the then-current completed manifests.

## Run and verify

Preview locally (no GPU allocation):

```bash
uv run --locked python -m pretraining_experiments.seed_and_schedules.launch --dry-run
```

Bounded check: one A10, ten WSD updates at the full 4096/8/4 memory workload.
It trains two updates, restores the checkpoint, finishes update 10, publishes
all four charts, and checks their uploaded scalar histories/chart tables.

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.seed_and_schedules.launch \
  --smoke --wandb-entity dkbsingh7-iit-delhi
```

Inspect `tracking-check.json` and open the linked run's `charts/train_lr`,
`charts/train_loss`, `charts/val_sql_loss`, and `charts/val_english_loss`.
The API check proves chart data exists; browser rendering still needs inspection.
The full-workload smoke run and its tracking recheck are listed in the
[experiment sources](seed_and_schedules/takeaways.md#sources).
The CPU coordinator verifies fresh uploaded history with bounded backoff after
GPU work finishes. To recheck an existing suite, set `SUITE_ID` to its printed ID:

```bash
uv run --locked modal run --env main --profile sanika-05 \
  -m pretraining_experiments.seed_and_schedules.launch \
  --verify-suite "$SUITE_ID"
```

A full sweep is a separate command:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.seed_and_schedules.launch \
  --wandb-entity dkbsingh7-iit-delhi
```

Each submission prints a durable Modal call ID. New runs use the W&B project
`sqlgen-seed-and-schedules` by default. The output directory under
`/vol/checkpoints/pretraining_experiments/seed_and_schedules/<suite>/` contains
a pinned `plan.json`, per-case `settings.json`, `metrics.jsonl`, `result.json`,
`latest.pt`, and snapshots at updates 100, 500 and 800. The smoke equivalents are 1, 5 and 8.
Restoring includes optimizer, CPU/CUDA/sample RNG, dataset identity, and metric
history, with unchanged schedule configuration. This implementation chooses
the screenshot's cosine-tail alternative rather than a mid-run decay branch.

Native metrics use one optimizer-update x-axis. Explicit charts contain the
actual sparse evaluation coordinates. Final scalar summaries use `final/*`,
so summary-only values do not look like empty validation histories. The suite
comparison overlays all five curves using saved full histories rather than
W&B's sampled history API.

After all five complete, the CPU coordinator writes `analysis.json` and
`takeaways.md`, including individual cosine losses, mean, sample standard
deviation (n=3), and paired seed-1 schedule differences. Twice the deviation
is only a screening heuristic. Download these measured outputs before updating
the repository takeaways; do not fill a result table from predictions.

## Correctness checks

```bash
uv run --locked python -m unittest discover -s tests
```

Tests cover WSD phase boundaries, fixed validation across training seeds,
unchanged experiment settings, exact CPU checkpoint replay, real offline W&B
chart serialization, missing uploaded-history rejection, controlled analysis,
and compatibility with earlier launch commands. No test submits training GPUs.

## Warmup duration follow-up

The seed/schedule experiment also has a prepared [300-update warmup comparison](seed_and_schedules/warmup_takeaways.md).
It reuses the completed 100-update baselines and adds only two seed-1 runs:
cosine and WSD linear. It has not been launched. WSD uses 300 warmup, 500 stable,
and 200 decay updates, preserving the 1000-update / 131.072M-token budget.
