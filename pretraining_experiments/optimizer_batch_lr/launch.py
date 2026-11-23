"""Single-GPU workers and CPU coordination for the optimizer sweep."""
import json
from pathlib import Path
import uuid

import modal

from training.config import DATA_DIR, TrainConfig, image, volume, wandb_secret
from training.errors import NumericalDivergence
from training.schedules import wsd_boundaries
from .plan import BATCH_TOKENS, PEAK_RATES, experiment_plan

app = modal.App('sqlgen-optimizer-batch-lr')
ROOT = Path('/vol/checkpoints/pretraining_experiments/optimizer_batch_lr')
PROJECT = 'sqlgen-optimizer-batch-lr'


@app.function(image=image, gpu='A10', cpu=4, memory=16384, timeout=86400,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def train_case(case, suite, dataset, entity, smoke):
    """Train one labeled config against the pinned corpus and return its history."""
    from training.trainer import run_training
    config = TrainConfig(**case['train'])
    output = ROOT / suite / case['label']
    try:
        result = run_training(config, entity, PROJECT, case['label'], output_dir=output,
                              group=suite, expected_dataset=dataset, x_axis='tokens_seen',
                              stop_after_update=2 if smoke else None)
    except NumericalDivergence as error:
        if smoke:
            raise
        result = {'run': case['label'], 'status': 'diverged', 'update': error.update,
                  'reason': str(error)}
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        volume.commit()
    return result


@app.function(image=image, timeout=86400, volumes={'/vol': volume}, secrets=[wandb_secret])
def run_suite(entity: str, suite: str, smoke: bool):
    """Run cases sequentially and persist one comparison row per finished worker."""
    from training.dataset import dataset_identity
    from training.tracking import verify_uploaded_result
    volume.reload()
    dataset = dataset_identity(DATA_DIR)
    cases = experiment_plan()
    if smoke:
        # Use both optimizer states for Muon and the largest accumulation setting.
        cases = [case for case in cases if case['batch_tokens'] == max(BATCH_TOKENS)
                 and case['train']['peak_lr'] == PEAK_RATES[case['train']['optimizer']][1]]
    root = ROOT / suite
    root.mkdir(parents=True, exist_ok=True)
    (root / 'plan.json').write_text(json.dumps({'cases': cases, 'dataset': dataset,
                                               'smoke': smoke}, indent=2))
    volume.commit()
    rows = []
    for case in cases:
        result = train_case.remote(case, suite, dataset, entity, smoke)
        volume.reload()
        row = {'run': case['label'], 'optimizer': case['train']['optimizer'],
               'batch_tokens': case['batch_tokens'], 'peak_lr': case['train']['peak_lr']}
        if result.get('status') == 'diverged':
            row.update(result)
        else:
            row.update(verify_uploaded_result(result))
            history = result['history']
            final = history[-1]
            row.update(status='smoke' if smoke else 'completed',
                       final_sql_loss=final['val/sql_loss'], tokens_seen=final['tokens_seen'])
            if not smoke:
                _, hold_end = wsd_boundaries(TrainConfig(**case['train']))
                row['hold_end_sql_loss'] = history[hold_end - 1]['val/sql_loss']
            seconds = sum(case['batch_tokens']
                          / r['perf/raw_tokens_per_sec'] for r in history)
            row['training_seconds'] = seconds
            row['tokens_per_training_second'] = final['tokens_seen'] / seconds
        rows.append(row)
        (root / 'comparison.json').write_text(json.dumps(rows, indent=2))
        volume.commit()
    return rows


@app.local_entrypoint()
def main(wandb_entity: str, smoke: bool = False, launch: bool = False):
    """Submit either the bounded three-worker smoke suite or all 18 full runs."""
    if smoke == launch or not wandb_entity:
        raise ValueError('Supply --wandb-entity and exactly one of --smoke or --launch')
    suite = ('smoke-' if smoke else 'sweep-') + uuid.uuid4().hex[:12]
    call = run_suite.spawn(wandb_entity, suite, smoke)
    print(f'{call.object_id}  {ROOT / suite}/comparison.json')
