"""Day 7 question: how much does final loss vary across training seeds?"""
from dataclasses import asdict
from training.config import TrainConfig


def cosine_cases(base: TrainConfig) -> list[dict]:
    """Return three controlled cases; ONLY the training seed changes.

    The caller supplies a shared validation_seed. Each case independently
    constructs its model and data RNG from its training seed. These three runs
    estimate combined initialization/sampling variability, not either in isolation.
    """
    if base.validation_seed is None:
        raise ValueError('Seed comparisons require one fixed validation_seed')
    return [{'label': f'cosine-seed-{seed}',
             'train': {**asdict(base), 'lr_schedule': 'cosine', 'seed': seed}}
            for seed in (1, 2, 3)]
