# Muon QK normalization and weight decay

## Question and prediction

Does normalizing queries and keys improve raw held-out SQL loss at zero decay? Which decay then works best with the selected setting? We expected QK normalization could moderate growing attention scores, if present, and EMA might be more useful before cooldown. Selection used final raw SQL loss.

## Controlled settings

Suite `sweep-4d990b7e0456` used seed 1, Muon peak 0.01, companion AdamW peak 0.0006, and 32,768 tokens per update. Each model processed 32,768,000 tokens. See the [design](README.md) for validation, EMA, probes, and checkpoint settings.

## Measured results

| QK norm | Decay | Raw SQL | Raw English | EMA SQL | EMA English | Sources |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| False | 0 | 0.81724 | 4.84481 | 0.82203 | 4.85616 | [Training](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/eeeef22b96054715a8dae3e12a90cabe) · [Attention and paired validation](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/julg5a8h) |
| True | 0 | 0.77368 | 4.74400 | 0.78075 | 4.75701 | [Training](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/7df0ac7693364225ad41799f7db2632e) · [Attention and paired validation](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/4atgmjjw) |
| True | 0.01 | 0.77329 | 4.74277 | 0.78065 | 4.75546 | [Training](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/cc4baa903ccc48f6b632608b213deddf) · [Attention and paired validation](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/ud9yrua3) |
| True | 0.1 | 0.75436 | 4.71652 | 0.75878 | 4.72436 | [Training](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/6f3339a8dae94c85971f23a45931b63a) · [Attention and paired validation](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-muon-qk-decay-ema/runs/0znlppkw) |

## Explanation and limitations

QK normalization lowered both final losses at zero decay. With QK enabled, decay 0.01 produced nearly the same loss as zero, while 0.1 had the lowest final losses tested. EMA finished above the raw SQL and English losses in all four runs. Measurements cover a single seed and horizon. The validation losses alone cannot establish an attention-logit explosion or its cause.

## Next experiment

Hold QK on and decay at 0.1 during the SQL–English mixture comparison. Define mixture targets by realized token share and track both validation losses.
