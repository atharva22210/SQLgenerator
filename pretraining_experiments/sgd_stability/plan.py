"""Six bounded stability probes with the preceding screen's token budget."""
from dataclasses import asdict
from training.config import TrainConfig

TOKEN_BUDGET = 32_768_000
BATCHES = (16_384, 32_768)
LEARNING_RATES = (.03, .1, .3)
EVAL_TOKENS = 2_097_152
PROBE_SEED = 23456


def experiment_plan() -> list[dict]:
    """Keep context, clipping, decay and token-based cosine fixed across six cases.

    SGD's LR scale differs from AdamW's. This exploratory bracket is not an
    asserted stability boundary. All cases start fresh from seed 1 and consume
    the same 2,000 hardware microbatches; no automatic LR-range extension runs.
    """
    cases = []
    for batch in BATCHES:
        steps = TOKEN_BUDGET // batch
        for lr in LEARNING_RATES:
            config = TrainConfig(
                optimizer='sgd', mode='masked_pack', seq_len=4096, micro_batch_size=4,
                grad_accum_steps=batch // 16384, max_steps=steps, warmup_steps=steps // 10,
                peak_lr=lr, min_lr=lr * .1, lr_schedule='cosine',
                weight_decay=.1, grad_clip=1., seed=1, validation_seed=12345,
                eval_every=EVAL_TOKENS // batch, val_batches=128)
            cases.append({'label': f'sgd-batch-{batch}-lr-{lr:g}', 'batch_tokens': batch,
                          'token_budget': TOKEN_BUDGET, 'train': asdict(config)})
    return cases
