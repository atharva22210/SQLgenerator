# Selecting QK normalization, weight decay, and EMA for Muon

Which combination gives the lowest held-out SQL loss after 32.768M input tokens? Train **four models** at seed 1. Select QK normalization at zero decay first, then compare three decay coefficients with the selected architecture. Track raw and EMA weights in each trajectory.

The reference is Muon at peak LR 0.01 on hidden matrices, companion AdamW at 0.0006 on the tied embedding/output matrix and norm vectors, and 32,768 input tokens per update. The earlier raw, QK-off, zero-decay run finished at SQL loss 0.8126. Repeat this case because the historical run lacks the evolving EMA weights and per-head attention history. The question here concerns the selected LR. The gradient rise observed at 0.04 belongs to a separate investigation. [Reference run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/e493969ba08141cab55a7cb084c3f510)

## Run order and controls

| Run | QK norm from initialization | Weight decay | Comparison |
| --- | --- | ---: | --- |
| 1 | Off | 0 | Instrumented reference and first EMA comparison |
| 2 | On | 0 | Effect of QK norm alone |
| 3 | Selected from runs 1 and 2 | 0.01 | Mild decay versus the selected zero-decay run |
| 4 | Same selection | 0.1 | Stronger decay versus zero and mild decay |

Choose QK using final **raw** SQL validation loss so EMA stays out of the architecture decision. Record any disagreement with the attention diagnostics. Keep QK off on a tie. Start runs 3 and 4 from the same seed and initialization as the selected zero-decay case. Starting from its trained checkpoint would change the training budget. This design leaves the QK-by-decay interaction at nonzero decay unmeasured.

Reuse the winning case's architecture, published dataset identity, seed, microbatch order, clipping at 1, and 100/700/200-update WSD schedule. Every case consumes 1,000 updates. Match the initial values of all shared weights across the QK pair. The added per-head normalization scales start at one. Apply the same decay coefficient to the Muon matrices and companion embedding group, each at its own scheduled LR. Keep all normalization vectors at zero decay. Pin validation seed 12345 and the same 128 validation batches per corpus. [Baseline plan](../optimizer_batch_lr/plan.py) · [Optimizer groups](../../training/optimizers.py)

## What to measure

**Attention.** Freeze one small probe of SQL and English documents with recorded IDs, tokens, masks, and query positions. At initialization and every 25 updates, evaluate it with the raw weights and no gradient. For each layer and query head, log query and key L2 norms before and after optional normalization, and the maximum and 99th percentile of valid scaled pre-softmax scores after RoPE. Match each query head to its actual grouped KV head. For softmax probabilities over allowed keys, log median and fifth-percentile normalized entropy per head. Compute entropy per query as −Σ p log p / log(number of allowed keys), omitting queries with only one allowed key. Use the same sampled query rows each time and include all their valid keys. This keeps cost bounded and respects causal and document boundaries. Label maxima as probe maxima. Keep the normal training attention kernel intact.

Plot scores and entropy against tokens, retaining head and layer identity so that a change in one head remains visible. Also plot pre-clip gradient norm and the fraction of updates clipped in each interval. Our prediction is that QK norm will moderate sustained score growth **if it occurs**. Rising upper-tail scores and falling entropy in the same heads across successive observations would motivate that explanation. A single high score or a head that consistently attends to one token is weak evidence. Score control, optimization stability, and SQL loss are three distinct outcomes.

**EMA.** Initialize one FP32 shadow copy from the initial weights. After each successful optimizer step, average each unique parameter once, preserving the tied weights. Choose a half-life of 1,638,400 input tokens (50 updates). The old shadow's coefficient per update is 2^(-1/50), about 0.9862. After 50 updates the contribution from the starting shadow halves. This is one chosen timescale, so the decision concerns enabling EMA at this half-life. The shadow is evaluation-only. Save and restore it alongside the step on resume. Evaluate raw and EMA on identical documents at least at update 800 (end of hold) and 1,000 (end of cooldown), restoring raw weights before further training. We predict averaging may help more before cooldown. Record paired SQL and English losses even when EMA loses.

## Checkpoints and paired evaluation

For each of the four runs, retain numbered, resumable snapshots at updates **100, 200, 400, 600, 800, and 1,000**. These cover the end of warmup, every scheduled validation, the end of the hold, and the final model. Save raw weights, EMA weights and half-life, optimizer state, sampler and RNG state, configuration, dataset identity, step and token count together. Keep earlier milestones when updating the latest resume file. Load the saved model configuration and either the raw `model` state or the `ema_model` state for SFT initialization. Start SFT with a fresh optimizer.

Update the shadow online with `ema = beta * ema + (1 - beta) * raw_after_step`. Every training forward pass uses raw weights. At each scheduled validation, score the same documents twice under evaluation mode, once with each set of weights. Preserve raw weights, optimizer state and training RNG across validation. Each retained snapshot can reproduce both evaluations later. Intermediate updates contribute to the running average without requiring a file for each step.

## Selection and resource budget

Select QK and decay using final raw SQL loss throughout. Record EMA for observation only and defer choosing inference weights. Keep raw and EMA diagnostics at all settings. Treat a near-tie as provisional. Report hold-end loss, final English loss, and attention diagnostics alongside the ranking. If scores look healthier but SQL gets worse, report that tradeoff. If results are close, retain the simpler reference provisionally or repeat the reference and contender at a second shared seed. When the only difference is EMA, one additional trajectory supplies both evaluations. Single-seed differences cannot establish a reproducible winner.

Use one A10 sequentially. Four full cases total 131,072,000 training tokens. The prior sweep's reported worker cost of about $6.31 for 24 cases implies about $1.05 for four at the old average. Attention probes and paired evaluation add unmeasured overhead. **$4 is the ceiling for all new work combined**, including smoke, startup, monitoring compute, and any repeats. Before training, verify enforceable remaining budget for that amount. Removing a workspace limit alone does not enforce an experiment limit.

Start with a bounded smoke on QK-on with decay and EMA. Verify masked probe scores against a dense reference, normalized entropy bounds, EMA checkpoint resume, raw weight restoration, peak memory, and actual uploaded W&B histories and chart tables. Use the smoke and first full case to measure billed cost before committing to later cases. If the remaining cap cannot cover a case with overhead, stop and report the observed comparisons. The second-seed check is optional and shares the same ceiling.

## Status

See the experiment takeaways for measured status. Run the bounded resume smoke before the full comparison. Ansh's SGD study provides the separate loss-slice and LR-branching exercise. [SGD landscape results](https://github.com/atharva22210/SQLgenerator/blob/82e1884/pretraining_experiments/loss_landscape/takeaways.md)

## Running the experiment

Use the locked project environment from the repository root.

```sh
.venv/bin/modal run --detach -m pretraining_experiments.muon_qk_decay_ema.launch --entity ENTITY --smoke
.venv/bin/modal run --detach -m pretraining_experiments.muon_qk_decay_ema.launch --entity ENTITY --launch
```

The CPU coordinator pins the original winning run's dataset identity. Each completed case publishes paired validation curves and a per-head W&B table. The Volume keeps full probe tokens, the sampled causal prefixes, step-indexed per-head metrics, and numbered checkpoints under the printed suite ID in `/vol/checkpoints/pretraining_experiments/muon_qk_decay_ema`. The completion record names the chosen checkpoint and evaluation weights.
