# Pretraining experiments

Read an experiment's question and prediction before launching it. Its takeaway
file separates measurements from explanations and uncertainty. Model code stays
in `model/`, corpus preparation/sampling in `data/`, and shared experiment code
in this package. Python 3.13.15 and the committed `uv.lock` define the environment.

| Experiment | Code / execution guide | Analytical takeaways |
| --- | --- | --- |
| Sequence packing | [sequence_packing/launch.py](sequence_packing/launch.py), [commands](../packing_experiment/README.md) | [Packing](sequence_packing/takeaways.md) |
| DDP correctness | [distributed_training/experiment_plan.py](../distributed_training/experiment_plan.py), [commands](../distributed_training/README.md) | [DDP correctness](../distributed_training/ddp_correctness_takeaways.md) |
| Gradient synchronization | [distributed_training/ddp_core.py](../distributed_training/ddp_core.py), `launch --part b` | [Gradient sync](../distributed_training/gradient_sync_takeaways.md) |
| Sharding and activation memory | [distributed_training/fsdp_checkpointing.py](../distributed_training/fsdp_checkpointing.py), [commands](../distributed_training/memory-and-sharding.md) | [Memory/sharding](../distributed_training/memory_takeaways.md) |
| Seed variation | [seed_and_schedules/seed_noise.py](seed_and_schedules/seed_noise.py) | [Seed noise](seed_and_schedules/seed_noise_takeaways.md) |
| Memory/resume check | [Shared trainer](../train.py), `--smoke` | [Measured smoke check](seed_and_schedules/smoke_takeaways.md) |
| Learning-rate schedules | [seed_and_schedules/launch.py](seed_and_schedules/launch.py) | [Schedules](seed_and_schedules/schedules_takeaways.md) |

The shared single-GPU loop lives outside this package in [`train.py`](../train.py).
Experiments supply settings and launch that loop. Shared support code lives in
[`training/`](../training/): configuration/data loading, schedules, checkpointing,
dataset identity, and W&B tracking. The distributed experiments retain their
rank-specific collective loop in `../distributed_training/ddp_worker.py`.

```text
train.py                         # shared single-GPU training loop
training/                        # shared support, no experiment launchers
pretraining_experiments/
  sequence_packing/
    launch.py
    check.py
    takeaways.md
  seed_and_schedules/
    launch.py
    seed_noise.py
    analysis.py
    seed_noise_takeaways.md
    schedules_takeaways.md
    smoke_takeaways.md
distributed_training/            # earlier DDP/FSDP experiments
  launch.py
  ddp_worker.py
  ...
```

Old module paths remain small compatibility imports; they contain no duplicate
training loop. Local reorganization does not replace the uploaded source in an
already running Modal job.

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
The [measured smoke check](seed_and_schedules/smoke_takeaways.md) records the first full-workload run and the verified upload fix.
The CPU coordinator verifies fresh uploaded history with bounded backoff after
GPU work finishes. To recheck an existing suite without retraining:

```bash
uv run --locked modal run --env main --profile sanika-05 \
  -m pretraining_experiments.seed_and_schedules.launch \
  --verify-suite day07-smoke-073af7c11d79
```

A full sweep is a separate command:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.seed_and_schedules.launch \
  --wandb-entity dkbsingh7-iit-delhi
```

Each submission prints a durable Modal call ID. Outputs are under
`/vol/checkpoints/pretraining_experiments/day07/<suite>/`: a pinned `plan.json`,
per-case `settings.json`, `metrics.jsonl`, `result.json`, `latest.pt`, and retained
snapshots at updates 100, 500 and 800. The smoke equivalents are 1, 5 and 8.
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
