# Optimizer and batch-size sweep

## Question

At equal training-token exposure, does doubling the batch change which optimizer achieves the lowest held-out SQL loss after a three-point peak-LR search?

## Prediction

Before the runs, we predicted that at least one optimizer will benefit from the extra updates at the smaller batch. Retuning its LR may reduce that gap.

## Controlled settings

See the [experiment design](README.md) and [case generator](plan.py) for optimizer groups, candidate LRs, fixed seeds, validation, and WSD phase lengths. AdamW at batch 32,768 with peak LR 0.001 is the reference case.

## Measured results

Smoke suite `smoke-09b8d1ed0977` completed on 2026-11-25. Each optimizer completed two 65,536-token updates at its middle LR. The coordinator verified the uploaded scalar histories and all four chart tables for each run. Peak reserved GPU memory was 3.6465 GiB in each case. [W&B project](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay)

All 24 cases in suite `sweep-cdc968400e2b` completed 32,768,000 input tokens on seed 1. W&B scalar histories and per-run chart tables passed verification for every case.

| Optimizer | Batch tokens | Peak LR | Hold-end SQL loss | Final SQL loss | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| adamw | 32,768 | 0.001 | 1.1321 | 0.9264 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/4faf589f37b44d619988e6733e00bb6b) |
| adamw | 65,536 | 0.001 | 1.4270 | 1.2144 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/605ec0680d6947a1890a06249884c691) |
| adamw | 32,768 | 0.0003 | 1.4388 | 1.2623 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/061eed01aa654502ad8e4a7ae47277da) |
| adamw | 65,536 | 0.0003 | 1.7382 | 1.5761 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/f77db9d563ee4433ad4a16fb9e24c572) |
| adamw | 32,768 | 0.003 | 1.5185 | 1.2874 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/f875a81171de41c3bd1e73ed9098b76d) |
| adamw | 65,536 | 0.003 | 1.5917 | 1.3597 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/8acb38d4db7e421e95df1b91be79bb37) |
| muon | 32,768 | 0.02 | 0.9761 | 0.8408 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/d0a6b729476f45f89d059ad533ee28e9) |
| muon | 65,536 | 0.02 | 1.1246 | 0.9564 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/3319efac2d074a9ba7a5720ab2f9c309) |
| muon | 32,768 | 0.01 | 0.9488 | 0.8126 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/e493969ba08141cab55a7cb084c3f510) |
| muon | 65,536 | 0.01 | 1.1263 | 0.9533 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/90379bf80ef0430db73fab71cadcba72) |
| muon | 32,768 | 0.04 | 1.4593 | 1.3085 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/dddb9d7b272f45b19ce6b42a6c36bc23) |
| muon | 65,536 | 0.04 | 1.3579 | 1.2026 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/59dbcf98a485454597f0b39c2a3c52b9) |
| momentum_sgd | 32,768 | 0.01 | 3.6509 | 3.4633 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/a6a3f39affe74e178b015e6a524e191c) |
| momentum_sgd | 65,536 | 0.01 | 4.1471 | 4.0150 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/e65321b628974d2ab1e23a362f4d95bf) |
| momentum_sgd | 32,768 | 0.003 | 4.5432 | 4.4212 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/39f61ce3c94946508520137d73ef3e3f) |
| momentum_sgd | 65,536 | 0.003 | 4.9715 | 4.8292 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/ffbc9d7144e44444b9c4edf5e58fbe2a) |
| momentum_sgd | 32,768 | 0.03 | 2.7446 | 2.5679 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/c7be8f8324774d9bb546a6a511d56576) |
| momentum_sgd | 65,536 | 0.03 | 3.3143 | 3.1292 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/564b680845d44f1e90056e7826bd88d5) |
| sgd | 32,768 | 0.1 | 3.7538 | 3.5296 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/1ebcb065d5714af1ac5ec6dc6d0f5363) |
| sgd | 65,536 | 0.1 | 4.2817 | 4.0802 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/03f5ccd07e374e48a5f76b7892820e4a) |
| sgd | 32,768 | 0.03 | 4.6595 | 4.4812 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/bef31bdcba5f4f009c5610d8b5a2bc3c) |
| sgd | 65,536 | 0.03 | 5.3929 | 5.1277 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/a54436c226f84d50aa5a231e0465f909) |
| sgd | 32,768 | 0.3 | 2.9503 | 2.7085 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/5e32d81396024cf2aed2dcc2e5169f17) |
| sgd | 65,536 | 0.3 | 3.6920 | 3.4758 | [Run](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-optimizer-batch-lr-wsd-zero-decay/runs/5653b0173d6449efa19bcc266aa43a6f) |

## Explanation

After choosing the best tested LR within each optimizer, all four reached lower final SQL loss at 32,768 tokens per update. This agrees with the prediction that additional updates could help at fixed token exposure. Muon at 0.01 led at both batches. AdamW preferred 0.001 at both. The two SGD variants each preferred their highest tested rate.

## Limitations

A single seed and short training horizon limit generalization. Muon results include its companion AdamW. A best LR on the grid boundary leaves the location of the optimum open.

## Next experiment

Expand a boundary LR first if needed, then repeat the leading configurations across seeds before selecting a training default.
