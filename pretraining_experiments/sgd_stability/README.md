# SGD batch and LR stability screen

Question: do smaller batches and a wider SGD LR bracket reveal persistent
fixed-example loss fluctuations, and which saved states could support later
controlled LR branches? This is an experiment, not a promise of oscillation.

| Batch tokens/update | Microbatch × context | Accumulation | Updates | Warmup | Peak LRs |
| ---: | --- | ---: | ---: | ---: | --- |
| 16,384 | 4 × 4096 | 1 | 2,000 | 200 | 0.03, 0.1, 0.3 |
| 32,768 | 4 × 4096 | 2 | 1,000 | 100 | 0.03, 0.1, 0.3 |

All six start fresh from seed 1, consume 32,768,000 input tokens (196,608,000
for the suite), and use the same 2,000 hardware microbatches. No earlier model
checkpoint is used as an initialization. Context remains 4096. The same model,
masked packing, BF16/FP32 setup, data identity checks and fixed 128-document
validation per corpus are reused from the AdamW screen. Validation occurs at
matching token coordinates every 2,097,152 tokens plus the final budget.

## Optimizer and controls

Use **plain SGD without momentum**, with gradient clipping **1.0** and
**decoupled weight decay 0.1** on matrices; norms receive no weight decay. These
are the user's chosen controls. The step is:

```text
clipped_gradient = clip(accumulated_target_weighted_gradient, max_norm=1)
parameter = (1 - learning_rate * decay) * parameter - learning_rate * clipped_gradient
```

`training.optimizers.DecoupledSGD` deliberately uses the same decay placement as
AdamW. Ordinary PyTorch SGD's weight_decay is coupled L2 regularization; changing
that silently would weaken the optimizer comparison. The existing momentum-SGD
implementation remains available to its original experiment and is unchanged.

Keep the earlier **cosine schedule**, with 10% warmup and a final LR of 10% of
peak, rather than changing optimizer and schedule together. SGD's numerical LR
scale differs from AdamW's; this bracket is exploratory. Training may be slow,
stable, oscillatory or divergent. No automatic retries at different LRs or
additional LR-range extensions are submitted.

This gives an initial optimizer comparison at the shared batch **32,768** only.
There is no AdamW run at 16,384 in the completed quarter-budget screen. Compare
best tested recipes with their actual LR ranges, not equal numerical LR or
claims of an optimum. Decoupled decay has the same rule and coefficient, but its
total strength still depends on the LR sum and number of updates. One seed and
boundary winners do not establish a universal optimizer ranking.

## Observe stability without confusing it with changing examples

Each update records raw training loss, LR, pre-clipping accumulated gradient norm,
and whether clipping activated. An independent sampler constructs a fixed probe
of one standalone English and one standalone SQL **training** document. Their
actual token IDs, target masks, positions and attention layout are saved in
`fixed_probe.pt` with the seed/interval recipe and pinned dataset identity.

The same probe is evaluated **after every update**, without backward passes or
advancing the training sampler. Model mode and Torch RNG are restored. Per-step
sampling avoids aliasing an alternating pattern by measuring only every 8th
update. The probe's token-weighted loss is a diagnostic, not a new held-out set.
One small probe cannot establish behavior over the full training objective.

W&B has explicit training, SQL validation, English validation, LR and fixed-probe
loss charts, all against input tokens. `train/clipped` is a native scalar. The
comparison uses at most 9,000 training/probe rows per chart (below the 10,000-row
artifact-table limit); sparse validation remains sparse. The CPU coordinator
verifies uploaded scalar coordinates and chart-table row counts after workers
exit. Browser rendering is a separate UI check. No online dummy W&B run is made.

Retaining clipping can suppress the instability we hope to investigate; report
that rather than forcing an expected pattern. With clipping, decay, stochastic
batches and changing LR, the full-batch, unclipped quadratic GD threshold
`2 / largest_Hessian_eigenvalue` is not a plug-in diagnostic. A loss wiggle alone
is not an edge-of-stability finding. See the scope of the original
[full-batch GD study](https://arxiv.org/abs/2103.00065).

## Checkpoints and future work

The shared engine saves full model/optimizer state, training and architecture
settings, sampler and CPU/CUDA RNG, token count, data identity and committed
metrics. Retained milestones are update 1, warmup end, quarter, midpoint and
finish: **1/200/500/1000/2000** for batch 16,384 and **1/100/250/500/1000** for
batch 32,768. Apart from update 1, these align by input-token exposure.

Each retained milestone also contains `update_direction`: actual CPU FP32
parameter differences from the immediately preceding update, including clipping,
decay and floating-point rounding. Directions are keyed by parameter name; do not
assume they are gradients. They describe the arriving update at the saved state.
Directions are computed only at milestones to avoid cloning all weights every
step. `latest.pt` is refreshed at validation points for recovery and may have no
direction; use retained `step-*.pt` files for directional probes.

The suite plan indexes checkpoint paths. Preserve the pinned corpus/tokenizer
and `fixed_probe.pt` to replay future batches and observations. Later original-LR
versus reduced-LR branches must start from one common checkpoint with identical
optimizer and RNG states. Ordinary resume still requires unchanged settings and
probe recipe; explicit changed-LR branching support and Hessian probes are future
work, not silently launched by this screen. A nonfinite run stops and retains
its last saved valid state; OOM/auth/data errors propagate as infrastructure errors.

## Commands and resources

Preview without allocation:

```bash
uv run --locked python -m pretraining_experiments.sgd_stability.launch
```

Explicitly launch the six cases, at most two independent A10 workers at a time:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.sgd_stability.launch \
  --launch --wandb-entity dkbsingh7-iit-delhi
```

Project: `sqlgen-sgd-stability`. Volume artifacts:
`/vol/checkpoints/pretraining_experiments/sgd_stability/<suite>/<case>/`.
Code, guide and `takeaways.md` live together in this folder; the optimization loop
remains shared in `training/trainer.py`. No existing experiment is moved or overwritten.

Prior six-run SDK lifetimes were about 69 aggregate worker-minutes. Per-update
probe forwards and five retained state/direction snapshots add overhead; allow
roughly **$2–$4** for this sweep as a planning estimate, not a cap or invoice.
Divergence may shorten a case. The nominal budget excludes validation/probe
forward passes. There are no extra GPU smoke runs; local real-SDK tests precede
submission and the first real pair is checked during startup.
