# Day 7: memory, resume, and tracking check

## Question

Does the requested 4096-context workload fit on one A10 after optimizer state
exists, resume a saved update, and publish complete W&B histories/chart tables?

## Prediction

The existing BF16 compute, unpadded attention and chunked vocabulary loss should
keep this 34.48M-parameter model within A10 memory. Checkpoint restore should
continue the schedule/data stream, and each curve should retain all ten updates.
This is a correctness prediction, not a prediction of the best LR schedule.

## Controlled settings

Run on 2026-11-22: one A10, 34,478,592 parameters, 4096 context, microbatch 8,
accumulation 4, masked packing, ten WSD-linear updates (warmup 1, stable 7,
decay 2), evaluation every update with one document per corpus. Train seed 1;
validation seed 12345. The first invocation stopped after update 2 and the
second restored that checkpoint and completed updates 3–10.

Data: flat `/vol/bpe_tokenization/pretrain_data`, SQL corpus
`ff43d7752a904b649fad5b2d2dbb4df3`, 32768-token BPE, manifest identity
`914bb1a0fe4adfc52f474a5f4dccea636b16242ffdc0b2cfa14a02adab54501d`.

## Measured results

| Check | Observation |
| --- | --- |
| Completed updates | 10; checkpoint restored after 2 |
| Peak allocated GPU memory across both invocations | 6.466 GiB |
| Peak reserved GPU memory | 6.748 GiB |
| Non-padding training tokens | 1,310,720 |
| Reported throughput, updates 3–10 | approximately 72.6–72.8k useful tokens/sec |
| WSD learning rates | 6e-4 through update 8; 3.3e-4 at 9; 6e-5 at 10 |
| Uploaded scalar coordinates | all updates 1–10 verified |
| Uploaded explicit charts | LR, training loss, SQL loss, English loss; ten rows each |

The W&B smoke run (`004edebb74da49cdb7630d29e8bb0266`) was subsequently
deleted at the user's request. Its archived Modal metrics/checks remain available.
The persistent outputs are under
`/vol/checkpoints/pretraining_experiments/day07/day07-smoke-073af7c11d79/`.
A CPU-only recheck wrote `tracking-checks.json` and the case's
`tracking-check.json` after the initial verification failure below.

## Explanation

The workload fit after AdamW state had been allocated. The shared loop kept
BF16 compute, FP32 parameters/state, target-weighted accumulation, variable-length
attention, and chunked loss. Reported throughput measures a completed training
update; it excludes validation, IO, logging and startup scans.

The first verification attempt reported only updates 1–7 even though saved
metrics and W&B's final summary reached 10. A fresh uncached API query recovered
all ten updates and all chart tables. Verification now bypasses history caches,
retries indexing delays with bounded backoff, ignores null fields on chart-only
rows, and checks table row counts. It runs on CPU after GPU work completes, and
can be repeated independently of training. The corrected CPU recheck passed.

## Limitations

This is one short correctness run. The single-document evaluations do not
measure a schedule winner or seed variation. GPU resume was exercised; exact
uninterrupted-versus-resumed equality was checked separately in CPU tests.
The API checks history coordinates and chart data, not visual browser rendering.
Throughput from eight resumed updates is not a stable full-sweep runtime estimate.

## Next experiment

Inspect the linked chart panels, then run the five planned 1000-update cases
with 128-document validation samples. Keep the pinned build and fixed validation
seed. Use completed results to estimate seed variation and compare schedules.
