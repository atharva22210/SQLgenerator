# Learning-rate schedules: measured findings

## Question and prediction

At equal training tokens, does WSD improve final loss over cosine? The motivating
prediction was a noisier/higher stable-phase loss followed by a final decay drop.
The observations only support part of that story: WSD improved during decay, but
was already ahead of cosine before the tail began.

## Controlled settings

All three schedules use training seed 1, the same fixed validation seed/sample,
131.072M non-padding input tokens, 1000 updates, 4096 context, and the same model
and optimizer. Warmup lasts 100 updates. Cosine then decays throughout the run;
WSD stays at 6e-4 through update 800 and decays to 6e-5 over updates 801–1000.

## Measured results

| Schedule | Final SQL loss | Final English loss |
| --- | ---: | ---: |
| Cosine | 0.796816 | 4.708972 |
| WSD linear | 0.667811 | 4.403726 |
| WSD cosine | 0.663367 | 4.406684 |

Against cosine, WSD linear reduces SQL cross-entropy by 0.129005 (16.19%) and
English by 0.305247 (6.48%). WSD cosine reduces them by 0.133449 (16.75%) and
0.302289 (6.42%). These are loss reductions, not SQL accuracy improvements.

## Explanation

At update 800, about 67–68% of the eventual SQL advantage and 62% of the English
advantage already existed. Sustaining the higher LR may have helped progress at
this budget; the curve alone cannot prove that mechanism. During the last 200
updates WSD lowered SQL loss by about 0.089 versus cosine's 0.046, and English
by about 0.204 versus 0.089. The final phase added to an existing advantage.

Tail choice is unresolved: WSD cosine is lower by 0.004445 on SQL, while WSD
linear is lower by 0.002958 on English. Those gaps are small relative to cosine's
seed spread and WSD's own spread has not been measured.

## Limitations

Same-seed training losses diverge on update 2, before schedules differ; matching
seeds did not yield identical GPU trajectories. The two WSD trajectories also
differ before their tails diverge. Kernel arithmetic is a plausible cause, not
an isolated diagnosis. Only one seed per WSD tail was measured. These findings
favor replication of WSD, not a universal schedule ranking or a significance claim.
Schedule changes also alter integrated LR and LR-scaled weight-decay exposure.

## Next experiment

Verify same-seed reproducibility, then prioritize WSD linear seeds 2 and 3.
Avoid paying for another full tail comparison solely to resolve the tiny current
gap. Keep cosine as the baseline in the planned batch/LR experiment to isolate
that question. Retained checkpoints do not by themselves test early-decay branching.

See the [full phase-by-phase analysis](takeaways.md) and its recorded curves,
numeric validation observations, stability measurements and source links.
