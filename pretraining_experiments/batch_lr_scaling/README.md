# Batch size and learning-rate scaling

Question: at the same training input-token budget, how does batch size affect the
best tested learning rate, stability and validation loss? This is the curriculum's
Experiment 8. Code and takeaways are organized by subject, not by day.

## Fixed design

Every completed case consumes **131,072,000 non-padding input tokens**, matching
the preceding schedule experiment. The planned 16 cases total 2,097,152,000 input
token slots, plus validation. This is not a promise that all cases fit a particular
credit balance. Review the selected grid and available compute before launching.

| Tokens per optimizer update | Microbatch | Context | Accumulation | Updates | Warmup updates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 4 | 4096 | 1 | 8000 | 800 |
| 65,536 | 4 | 4096 | 4 | 2000 | 200 |
| 262,144 | 4 | 4096 | 16 | 500 | 50 |
| 524,288 | 4 | 4096 | 32 | 250 | 25 |

For every row, test peak LRs **3e-4, 1e-3, 3e-3, 6e-3**. Use cosine decay with
10% warmup and minimum LR 0.1 × that case's peak. Unlike a fixed warmup-step
count, these settings spend the same 13,107,200 input tokens warming up.
All phase lengths here are integers; no rounding is required.

All cases use the same model (6 layers, width 512, FFN 1408, 8 query/4 KV heads,
QK normalization off), normal initialization std 0.02, AdamW betas (0.9,0.95),
weight decay 0.1, clip norm 1, BF16 compute/FP32 weights, and masked packing.
Training seed is 1; validation seed is 12345. The coordinator resolves the latest
completed English/SQL manifests once, and each worker verifies that pinned
identity. The vocabulary/tokenizer checks are shared with the previous experiment.

Each case consumes 8000 microbatches of 4×4096 slots. The Python sampling RNG is
independent of optimization and validation, so changing accumulation groups the
same intended microbatch stream into different updates. It does not require more
GPUs. GPU arithmetic is not assumed bit-identical; same-seed drift was observed
in the preceding experiment and limits strong causal claims about small gaps.

## What is measured?

The batch size is measured in input tokens per optimizer update. Every accumulated
loss is weighted by the number of valid prediction targets. Actual predicted
counts are logged as `data/predicted_tokens`; next-token shifts and document
boundary masks make them smaller than the input-token budget. Different training
seeds are not swept here. The 0.5 SQL document-sampling probability does not imply
50% SQL tokens; the actual token fraction is logged.

Evaluate the same 128 standalone documents per corpus every **8,388,608 input
tokens**, plus the final budget. This yields 16 validation observations per corpus
for every case. In update units, the intervals are 512,128,32,16 respectively.
Evaluating every 50 updates instead would compare different token exposure and
charge large-batch cases for far more evaluation per training token.

A nonfinite gradient norm or training loss raises `NumericalDivergence`. That case
is recorded as `diverged`, including the failed update, and excluded from final
loss ranking. Its last valid checkpoint remains available. CUDA OOMs, missing
corpora, W&B errors, and other runtime exceptions propagate instead of being
silently called an unstable LR. Large but finite loss is not automatically
classified as divergence. No LR or batch is automatically reduced.

## Files and shared engine

- `plan.py`: pure grid, batch arithmetic, proportional warmup, token-aligned validation.
- `launch.py`: explicit submission, bounded concurrency and persistent results.
- `analysis.py`: matched-budget checks, best-tested LR selection and plots.
- `takeaways.md`: question, predictions, results status, limitations and next test.
- [`../../train.py`](../../train.py): the single shared GPU optimization loop.

## Preview now, launch later

Local preview needs no Modal connection and allocates no GPU:

```bash
uv run --locked python -m pretraining_experiments.batch_lr_scaling.launch
```

Preview a selected subset using the exact numeric batch sizes/LRs:

```bash
uv run --locked python -m pretraining_experiments.batch_lr_scaling.launch \
  --batch-sizes 16384,65536 --learning-rates 0.0003,0.001
```

**Future execution command** (not run as part of implementation):

```bash
uv run --locked modal run --detach --env main --profile sanika-05 \
  -m pretraining_experiments.batch_lr_scaling.launch \
  --launch --wandb-entity dkbsingh7-iit-delhi --max-parallel 2
```

The Modal entrypoint also defaults to preview unless `--launch` is supplied.
The same subset flags work on the Modal command. `--max-parallel` accepts 1–5,
with two independent A10 workers at a time by default. Lower concurrency limits
simultaneous GPU use, not aggregate GPU cost. There is no automatic smoke run.
Use local correctness tests before deciding whether any GPU preflight is needed.

The coordinator remains detached from the laptop. Each case receives an isolated
directory under `/vol/checkpoints/pretraining_experiments/batch_lr_scaling/<suite>`.
The plan, dataset identity, raw metrics, settings, result and latest checkpoint
are saved. Warmup-end snapshots are retained. Full-state checkpoint restoration
reuses the optimizer, sampler and CPU/CUDA RNG states; it cannot change LR/batch
configuration while pretending to resume the same experiment.

## W&B and outputs

Per-case native curves use `tokens_seen` as the x-axis. Counter metrics are hidden
from automatic panel generation; summary values are not treated as line histories.
This does not change panels already saved in an older W&B workspace.
The CPU coordinator verifies fresh uploaded histories after GPU work finishes.

The comparison run includes:

1. Final SQL loss versus peak LR, one line per batch size.
2. Best tested SQL loss versus batch size at the same token budget.
3. SQL/English validation, training loss and LR versus non-padding tokens.
4. A full numeric table including diverged cases; their losses remain absent,
   rather than being replaced with a made-up large number or zero.

Training/LR overview charts sample up to 401 points per case to avoid silently
exceeding W&B table limits across all 16 runs. All sparse validation observations
and original raw scalar histories are retained. Final results use the final
matched-budget checkpoint, not each run's best checkpoint at unequal budgets.

`analysis.json` and `takeaways.md` record the measured grid and the best tested
LR for each batch. A selected subset is labeled by its actual rows; it does not
stand in for a completed full sweep. A winner at an LR-grid edge is a reason to
extend the grid later, not proof of an optimum or an exact critical batch size.
One seed and a reused sampled validation set do not establish significance or
SQL execution accuracy. Gradient noise and curvature are not directly measured.

## Local correctness checks

```bash
uv run --locked python -m unittest discover -s tests -p 'test_batch_lr_scaling.py'
uv run --locked python -m unittest discover -s tests
```

Tests verify batch/budget arithmetic, proportional warmup, token-aligned LR and
validation, default no-submission behavior, subset validation, divergence handling,
rejection of mismatched data/settings/budgets, and token-based chart table limits.
The tests do not launch Modal functions, GPUs, or online W&B runs.
