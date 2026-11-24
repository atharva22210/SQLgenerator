# Loss landscape, curvature and matched-checkpoint branching

This follows the completed `sgd_stability` experiment. It does not retrain the
six-run grid. Keep code and the analytical `takeaways.md` in this subject folder;
reuse `training/trainer.py` for optimization.

**Gradient clipping remains 1.0.** The user's clarification was to omit clipping
frequency, not remove clipping. This report does not aggregate that frequency;
the existing trainer may retain its raw per-update activation flag.

## Predeclared investigation

1. **CPU: quantify historical validation fluctuations.** Use all six source SGD
   runs. Fit a quadratic trend against normalized input-token exposure and report
   the sample SD (ddof=1) of residuals. Show linear-detrend sensitivity too. Exclude
   warmup for the primary window (15 points); use the late half as a second window
   (9 points). Trend mismatch and sparse sampling limit interpretation. This is
   not seed uncertainty or a statistical oscillation detector.
2. **Four directional slices, one batch.** Fix batch 16,384. For peak LR 0.1 and
   0.3, load retained updates 500 (8.192M tokens) and 2000 (32.768M). Normalize the
   saved actual arriving update vector to whole-parameter norm 1. Evaluate alpha
   = 0, ±0.01, ±0.02, ±0.04, ±0.08 and calculate central directional curvature at
   epsilon 0.01 and 0.02. Restore the checkpoint weights after every slice. The
   directions differ across states; none is a top-Hessian direction by definition.
3. **Two 200-update branches.** Fork the same LR-0.3/update-500 checkpoint. Hold
   LR constant at 0.3 in the control arm and 0.1 in the intervention. These are
   new schedules; the source's current cosine LR was approximately 0.282, so the
   control is a peak-LR hold rather than exact continuation of the old schedule.
   Keep SGD, clipping 1.0, decoupled decay 0.1, batch 16,384, context 4096 and loss
   accounting unchanged. Restore source weights, optimizer, sampler and CPU/CUDA
   RNG in each arm. Hash actual input IDs, targets, positions and attention
   boundaries on every successful update; compare hashes before drawing paired
   conclusions. Reuse the exact saved fixed probe, observed every 2 updates plus
   update 1/final. Full SQL/English validation occurs every 50 updates.
4. **Bounded Hessian estimates.** At the same four retained states, run 12 exact
   autograd Hessian-vector products in power iteration. No dense Hessian is built.
   Report the signed dominant estimate and relative residual, not just a number.
   Power iteration finds the largest-magnitude eigenvalue: if it is negative,
   leave `lambda_max_estimate` absent instead of mislabeling it largest-positive.
   Even a positive estimate is provisional if the residual has not converged.

## Precision and objective

The fixed training probe has 1,255 English and 77 SQL prediction targets, and is
therefore 94.22% English-weighted. Preserve that objective to connect with the
recorded fluctuations, while acknowledging its small/unbalanced scope. This is
not the full training-corpus Hessian or held-out validation curvature.

Loss slices and Hessian products use FP32 parameters, TF32 disabled, and MATH
scaled-dot-product attention. Each saved probe row is one standalone document;
trim only padding and retain every real token, position and target. This avoids
both BF16 finite-difference noise and the FlashAttention backend's unsupported
second derivatives. Training remains on its usual BF16/FlashAttention path.
CPU tests verify equal reference loss and exact second derivatives on a tiny
Transformer, plus known positive/negative quadratic Hessians.

The data-loss Hessian excludes weight decay and the derivative of clipping.
Report `actual_checkpoint_lr × positive_curvature_estimate` as a reference plot,
using the LR actually applied at that checkpoint rather than its nominal peak.
The textbook `eta*lambda = 2` threshold applies to unclipped full-batch gradient
descent on a local quadratic; our clipped, decayed, stochastic process does not
obey that rule directly. Neither fluctuations, one directional slice, nor a
large reference product alone proves an edge of stability. See
[the original full-batch GD study](https://arxiv.org/abs/2103.00065).

Branching tests LR sensitivity, but LR also changes per-step shrinkage:
0.97 at LR 0.3 versus 0.99 at LR 0.1 on decayed matrices. A causal claim about
only the gradient-step mechanism requires separately controlling that factor.
A smooth but stalled branch is not automatically a better branch; compare raw
loss level, improvement and validation as well as fluctuation amplitude.
Every-2-update probes may miss period-2 behavior; retain raw training history too.

## Persistence and resource bounds

Source root (read-only):
`/vol/checkpoints/pretraining_experiments/sgd_stability/sgd-stability-87b9a2c0cec8`.
New outputs:
`/vol/checkpoints/pretraining_experiments/loss_landscape/<suite>/`.

The suite saves the exact plan, historical fluctuation scores, one JSON per
checkpoint, curvature iterations, branch settings/results, future-input hashes,
paired-input verification and summary. Each branch gets a new W&B ID and isolated
checkpoints; original source histories/weights are not overwritten. Ordinary
resume remains strict; `restore_branch` explicitly whitelists schedule/length
changes. Branch provenance and hash prefixes survive checkpoint resume.

Only 400 new optimizer updates are requested: 6,553,600 training input tokens
across the two arms. Validation/probe forwards and four × twelve HVP iterations
are additional work. Use one A10 sequentially for the four diagnostics, then two
A10 workers for the branches. Diagnostic GPU timeout is 30 minutes; each branch
is limited to 20 minutes. These are upper bounds, not expected durations or a
spending cap. The CPU detrending allocates no GPU. No AdamW reruns or automatic
LR extensions are launched.

## Commands

```bash
# Preview without submitting compute
uv run --locked python -m pretraining_experiments.loss_landscape.launch

# Explicit bounded investigation
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.loss_landscape.launch \
  --launch --wandb-entity dkbsingh7-iit-delhi
```

W&B project: `sqlgen-loss-landscape`. It contains two branch training runs and
one comparison run with detrended residuals, aligned loss slices, Hessian
estimates and branch curves. The CPU coordinator verifies completed branch
histories and chart tables; browser rendering remains a UI check. Offline tests
serialize the actual chart objects before submission. No online dummy run or
extra training sweep is created.

## Executed signed-eigenvalue refinement

The early peak-0.1 checkpoint had a negative dominant eigenvalue, so the completed
investigation used one extra 12-step symmetric Lanczos projection plus a residual
HVP to estimate the largest algebraic positive curvature. Original signed power
estimates remain in the record. No training was repeated.

The explicit correction/finalization commands are:

```bash
uv run --locked modal run --env main --profile sanika-05 \
  -m pretraining_experiments.loss_landscape.refine --suite landscape-bdb9235e3e0e
uv run --locked modal run --env main --profile sanika-05 \
  -m pretraining_experiments.loss_landscape.finish \
  --suite landscape-bdb9235e3e0e --wandb-entity dkbsingh7-iit-delhi --republish
```

`finish` without `--republish` only verifies already-uploaded charts and exports
evidence on CPU. Use the explicitly named `hessian/refined_estimates` and
`hessian/refined_lr_times_curvature` panels for the correction; the initial
power-only charts remain historical evidence. Bounded backoff handles delayed
Forge/W&B metadata indexing after SDK finalization.
