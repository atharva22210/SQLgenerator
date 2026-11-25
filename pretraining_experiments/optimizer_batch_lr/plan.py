"""Preview the full fixed-token experiment locally with python -m."""
from dataclasses import asdict
import json

from training.config import TrainConfig

TOKEN_BUDGET = 32_768_000
BATCH_TOKENS = (32_768, 65_536)
# Insertion order is the execution priority when funding is limited.
# Rates bracket a candidate for each optimizer at both batch sizes.
PEAK_RATES = {'adamw': (0.0003, 0.001, 0.003),
              'muon': (0.01, 0.02, 0.04),
              'momentum_sgd': (0.003, 0.01, 0.03),
              'sgd': (0.03, 0.1, 0.3)}


def experiment_plan():
    """Return labeled trainer configs; both batches consume 2,000 identical microbatches."""
    cases = []
    for optimizer, rates in PEAK_RATES.items():
        for peak in (rates[1], rates[0], rates[2]):
            for batch in BATCH_TOKENS:
                steps = TOKEN_BUDGET // batch
                config = TrainConfig(
                    mode='masked_pack', optimizer=optimizer, peak_lr=peak, min_lr=peak / 10,
                    weight_decay=0.0, grad_accum_steps=batch // (4 * 4096),
                    micro_batch_size=4, seq_len=4096, seed=1, validation_seed=12345,
                    val_batches=128, max_steps=steps, warmup_steps=steps // 10,
                    lr_schedule='wsd', wsd_stable_frac=0.7, wsd_decay_frac=0.2,
                    wsd_decay_type='linear', eval_every=steps // 5)
                cases.append({'label': f'{optimizer}-batch-{batch}-lr-{peak:g}',
                              'train': asdict(config), 'batch_tokens': batch})
    return cases


if __name__ == '__main__':
    print(json.dumps(experiment_plan(), indent=2))
