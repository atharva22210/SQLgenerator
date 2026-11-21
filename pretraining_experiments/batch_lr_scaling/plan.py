"""Pure experiment design; importing or printing a plan allocates no hardware."""
from dataclasses import asdict
from training.config import TrainConfig

TOKEN_BUDGET = 131_072_000
MICROBATCH = 4
CONTEXT = 4096
BATCHES = (16_384, 65_536, 262_144, 524_288)
LEARNING_RATES = (3e-4, 1e-3, 3e-3, 6e-3)
EVAL_TOKENS = 8_388_608


def experiment_plan(batch_sizes=BATCHES, learning_rates=LEARNING_RATES) -> list[dict]:
    """Return cases with matching total slots, validation tokens, and microbatch stream.

    Batch size means tokens per optimizer update, not per GPU forward. Every case
    consumes 8000 microbatches of 4×4096 slots; accumulation only changes how they
    are grouped into updates. The seed-1 sampler is independent of the optimizer.
    Warmup is 10% of each run's update count, not a common number of updates.
    """
    if not batch_sizes or not learning_rates or len(set(batch_sizes)) != len(batch_sizes) or len(set(learning_rates)) != len(learning_rates):
        raise ValueError('Select nonempty, unique batches and learning rates')
    if any(batch not in BATCHES for batch in batch_sizes) or any(lr not in LEARNING_RATES for lr in learning_rates):
        raise ValueError('Selection must be part of the agreed batch/LR grid')
    cases = []
    for batch in batch_sizes:
        steps = TOKEN_BUDGET // batch
        for lr in learning_rates:
            config = TrainConfig(mode='masked_pack', seq_len=CONTEXT, micro_batch_size=MICROBATCH,
                                 grad_accum_steps=batch // (MICROBATCH * CONTEXT), max_steps=steps,
                                 warmup_steps=steps // 10, lr_schedule='cosine', peak_lr=lr, min_lr=lr * .1,
                                 eval_every=EVAL_TOKENS // batch, val_batches=128, seed=1, validation_seed=12345)
            cases.append({'label': f'batch-{batch}-lr-{lr:g}', 'batch_tokens': batch,
                          'token_budget': TOKEN_BUDGET, 'train': asdict(config)})
    return cases
