# Seed variation: measured findings

## Question and prediction

How much do final losses change across training seeds? We expected some variation,
but recorded no numerical prediction before launch. The seeds change initial
weights and training-document sampling together; they do not isolate either cause.

## Controlled settings

Cosine seeds 1, 2 and 3; identical 34.48M model, audited dataset, 4096 context,
masked packing, microbatch 8, accumulation 4, and 1000 updates. Every run recorded
131,072,000 non-padding input tokens. Validation uses the same 128 sampled
standalone documents per corpus with seed 12345, evaluated every 50 updates.

## Measured results

| Seed | Final SQL loss | Final English loss |
| --- | ---: | ---: |
| 1 | 0.796816 | 4.708972 |
| 2 | 0.812063 | 4.753372 |
| 3 | 0.807651 | 4.713550 |
| Mean | 0.805510 | 4.725298 |
| Sample std, n=3 | 0.007846 | 0.024420 |
| Descriptive 2 × std | 0.015692 | 0.048840 |

All final losses were also the best observed validation losses in their runs.

## Explanation

The measured SQL spread is modest relative to the 0.129–0.133 loss advantage
observed for WSD. Seed 1 was the best final cosine outcome in both corpora, so
the schedule comparison was not against a weak baseline seed. This motivates
replicating WSD, but does not establish a statistically significant treatment effect.

## Limitations

Three samples give an uncertain variance estimate. This is sample standard
deviation across runs, not a token-level standard error. The 20 evaluations
reuse one validation sample and are not independent test sets. Same-seed runs
also diverged during identical warmup schedules, so GPU numerical variation may
contribute. The threshold 2 × std is only a screening heuristic.

## Next experiment

Repeat WSD linear for seeds 2 and 3 if budget allows, and compare paired losses
against the already measured cosine runs. First check initial-state/batch hashes
and same-seed GPU reproducibility. Larger validation samples can be evaluated
from retained checkpoints without retraining the model.

See the [full analysis and evidence](takeaways.md), including the exact manifests,
individual W&B links, saved validation observations, and reproducibility caveats.
