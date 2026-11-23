# Optimizer and batch-size sweep

## Question

At equal training-token exposure, does doubling the batch change which optimizer achieves the lowest held-out SQL loss after a three-point peak-LR search?

## Prediction

Unmeasured prediction: at least one optimizer will benefit from the extra updates at the smaller batch. Retuning its LR may reduce that gap.

## Controlled settings

See the [experiment design](README.md) and [case generator](plan.py) for optimizer groups, candidate LRs, fixed seeds, validation, and WSD phase lengths. AdamW at B with peak LR 0.0006 is the reference case.

## Measured results

Pending GPU execution. Record the suite ID, dataset identity, per-case W&B links, hold-end and final SQL loss, and training throughput from comparison.json here. Record divergence alongside completed cases.

## Explanation

Pending measurements. Compare rankings at 80% and 100% of the token budget to see whether cooldown changes the ordering.

## Limitations

A single seed and short training horizon limit generalization. Muon results include its companion AdamW. A best LR on the grid boundary leaves the location of the optimum open.

## Next experiment

Expand a boundary LR first if needed, then repeat the leading configurations across seeds before selecting a training default.
