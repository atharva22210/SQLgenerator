# Optimizer, batch size, and peak learning rate

Compare final held-out SQL loss after tuning three peak LRs for each optimizer at each batch size. Each of the 18 runs consumes exactly 32,768,000 packed training input tokens. The quarter budget limits how long we train on the shared sampling stream. Sampling still uses the full published corpus.

| Optimizer | Candidate peak LRs | Parameters updated |
| --- | --- | --- |
| AdamW | 0.00015, 0.0006, 0.0024 | All weights, betas (0.9, 0.95) |
| Momentum SGD | 0.005, 0.02, 0.08 | All weights, momentum 0.9, no Nesterov |
| Muon + AdamW | 0.005, 0.02, 0.08 | PyTorch 2.10 Muon on hidden matrices, momentum 0.95 with Nesterov. AdamW on tied embedding/output weights and norm vectors, fixed peak 0.0006 |

The grids are starting hypotheses. Weight decay is zero throughout to remove the difference between coupled and decoupled decay. The historical AdamW run used weight decay 0.1, so the within-sweep AdamW runs provide the baseline. All scheduled groups end at 10% of their own peak LR.

| Batch in tokens | Microbatches per update (4 × 4096 each) | Warmup / hold / linear decay in updates | End-of-hold / final validation updates |
| --- | ---: | --- | --- |
| 131,072 | 8 | 25 / 175 / 50 | 200 / 250 |
| 262,144 | 16 | 12 / 88 / 25 | 100 / 125 |

Both runs draw 2,000 microbatches in the same order, starting from model and sampler seed 1. The trainer validates at each fifth of the token budget using 128 fixed batches per corpus and validation seed 12345. The [earlier WSD comparison](../seed_and_schedules/takeaways.md) motivates the 10/70/20 split.

## Run

From the repository root, use Python 3.13.15 and the locked project environment. Preview all configs locally.

```sh
.venv/bin/python -m pretraining_experiments.optimizer_batch_lr.plan
```

With the team's Modal profile, published corpus on the sqlgen-data volume, and wandb-secret configured, run the bounded GPU smoke first. Replace ENTITY with the W&B entity. Detached mode lets the CPU coordinator finish after the terminal exits.

```sh
.venv/bin/modal run --detach -m pretraining_experiments.optimizer_batch_lr.launch --wandb-entity ENTITY --smoke
```

The smoke uses the middle LR and larger batch for each optimizer, runs two updates on one A10 at a time, validates the final update, and checks the uploaded W&B scalar history and chart tables. Inspect all three rows in comparison.json, the linked charts, and peak GPU memory before submitting the sweep.

```sh
.venv/bin/modal run --detach -m pretraining_experiments.optimizer_batch_lr.launch --wandb-entity ENTITY --launch
```

The full sweep uses one A10 at a time for 589,824,000 total training tokens plus validation. The printed suite path on the volume contains plan.json with the dataset identity, per-case history and checkpoints, and comparison.json with SQL loss at the end of the hold and decay, training-loop seconds, throughput, and W&B links. Training-loop time excludes evaluation, checkpoint writes, and tracking overhead. Compare the lowest finite final SQL loss within each optimizer and batch. An LR-grid winner at an endpoint calls for testing the neighboring range.
