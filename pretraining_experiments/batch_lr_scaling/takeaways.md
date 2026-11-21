# Batch size and learning rate: experiment record

## Question

At 131.072M training input tokens per case, can a larger effective batch and a
higher learning rate match or improve the loss of more frequent smaller updates?

## Prediction

The best tested LR may move upward as batch size increases. At sufficiently large
batches, fewer updates might harm token efficiency even after LR tuning. The
16-run grid tests these ideas; neither a U-shaped curve nor a critical-batch knee
is guaranteed. We have not recorded a numerical prediction for the knee.

## Controlled settings

Context 4096; microbatch 4; accumulation 1,4,16,32. Effective batches are
16,384 / 65,536 / 262,144 / 524,288 input tokens. Updates are 8000 / 2000 /
500 / 250. Every batch size tests peak LR 3e-4,1e-3,3e-3,6e-3, cosine decay,
10% warmup and minimum LR 10% of peak. The same model, updated dataset identity,
training seed 1, validation seed 12345 and 128-document validation samples are
held fixed. See [the guide](README.md) for the complete specification.

## Measured results

**Not run.** Implementation and local tests provide no training results. No GPU
smoke test, training sweep or online W&B run was launched for this experiment.
Fill the result table from completed saved histories, including failed cases and
the actual subset tested, rather than using predictions as findings.

## Explanation to test

Larger accumulated batches average more targets before an optimizer step, but
also reduce the number of updates at a fixed token budget. Higher LR might offset
some of that reduction; it can also make training unstable. We vary both factors
in a grid rather than impose a square-root or linear LR scaling rule. The shared
trainer preserves target-weighted accumulation across masked microbatches.

## Limitations

One seed per grid cell; four candidate LRs; sampled and reused validation; no
SQL execution evaluation. Best tested LR is not the true optimum. A loss increase
at a large batch can reflect an inadequate LR grid, optimization settings, or
finite-budget effects. A grid alone does not measure gradient noise or curvature,
and cannot identify an exact universal critical batch size. LR-dependent AdamW
shrinkage and its moment updates also change across the tested configurations.

## Next experiment

After reviewing budget and collecting the selected runs, compare only completed
cases at equal token exposure. Extend an LR boundary if necessary, then replicate
promising batch/LR comparisons with more seeds. Choose additional experiments
from measured uncertainty, not from an expected diagram in the curriculum.
