"""Bounded follow-up to the completed SGD stability screen."""
from dataclasses import asdict
from pathlib import Path
from training.config import TrainConfig

SOURCE_SUITE = 'sgd-stability-87b9a2c0cec8'
SOURCE_ROOT = Path('/vol/checkpoints/pretraining_experiments/sgd_stability') / SOURCE_SUITE
BATCH_TOKENS = 16384
BRANCH_UPDATES = 200
BRANCH_EVERY = 2
HESSIAN_STEPS = 12
EPSILONS = (.01, .02)
ALPHAS = (-.08, -.04, -.02, -.01, 0., .01, .02, .04, .08)


def checkpoint_plan() -> list[dict]:
    """Compare one batch at two original peak LRs and equal early/late exposure."""
    return [{'label': f'lr-{lr:g}-{phase}', 'peak_lr': lr, 'phase': phase, 'update': update,
             'checkpoint': str(SOURCE_ROOT / f'sgd-batch-16384-lr-{lr:g}' / f'step-{update:06d}.pt')}
            for lr in (.1, .3) for phase, update in (('early', 500), ('late', 2000))]


def branch_plan(source_config: dict) -> list[dict]:
    """Fork the same noisy checkpoint into two constant-LR, 200-update arms.

    Only schedule, horizon and evaluation interval change. Clipping and decay
    remain 1.0 and 0.1. Original SGD has no momentum/adaptive state, but restore
    the full optimizer state and all RNG states rather than assuming that.
    """
    if (source_config.get('optimizer') != 'sgd' or source_config.get('peak_lr') != .3
            or source_config.get('micro_batch_size') * source_config.get('seq_len') * source_config.get('grad_accum_steps') != BATCH_TOKENS
            or source_config.get('grad_clip') != 1. or source_config.get('weight_decay') != .1):
        raise ValueError('Expected the retained clipped SGD 0.3/batch-16384 source')
    source = SOURCE_ROOT / 'sgd-batch-16384-lr-0.3' / 'step-000500.pt'
    cases = []
    for lr in (.3, .1):
        config = {**source_config, 'peak_lr': lr, 'min_lr': lr, 'warmup_steps': 0,
                  'max_steps': BRANCH_UPDATES, 'eval_every': 50}
        cases.append({'label': f'constant-lr-{lr:g}', 'checkpoint': str(source),
                      'train': asdict(TrainConfig(**config))})
    return cases
