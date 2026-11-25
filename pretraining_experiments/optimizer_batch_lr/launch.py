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
PROJECT = 'sqlgen-optimizer-batch-lr-wsd-zero-decay'


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
def run_suite(entity: str, suite: str, smoke: bool, optimizer: str):
    """Run cases sequentially and persist one comparison row per finished worker."""
    from training.dataset import dataset_identity
    from training.tracking import verify_uploaded_result
    volume.reload()
    dataset = dataset_identity(DATA_DIR)
    cases = experiment_plan()
    if optimizer:
        cases = [case for case in cases if case['train']['optimizer'] == optimizer]
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
            row['peak_reserved_gib'] = final['perf/peak_reserved_gib']
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
    if not smoke:
        publish_comparison(entity, suite, rows)
    return rows


def publish_comparison(entity, suite, rows):
    """Plot final SQL loss against peak LR using only complete equal-budget runs."""
    import wandb

    with wandb.init(entity=entity, project=PROJECT, group=suite, name=f'{suite}-comparison',
                    job_type='comparison') as run:
        for optimizer in PEAK_RATES:
            curves = []
            labels = []
            for batch in BATCH_TOKENS:
                selected = [row for row in rows if row['optimizer'] == optimizer
                            and row['batch_tokens'] == batch and row['status'] == 'completed']
                if selected:
                    curves.append(sorted(selected, key=lambda row: row['peak_lr']))
                    labels.append(f'{batch:,} tokens/update')
            if curves:
                chart = wandb.plot.line_series(
                    xs=[[row['peak_lr'] for row in curve] for curve in curves],
                    ys=[[row['final_sql_loss'] for row in curve] for curve in curves],
                    keys=labels, title=f'{optimizer}: final SQL validation loss', xname='Peak LR')
                run.log({f'comparison/{optimizer}': chart})


@app.local_entrypoint()
def main(wandb_entity: str, smoke: bool = False, launch: bool = False, optimizer: str = ""):
    """Submit a two-update smoke or full runs, optionally for one optimizer."""
    if smoke == launch or not wandb_entity:
        raise ValueError('Supply --wandb-entity and exactly one of --smoke or --launch')
    if optimizer and optimizer not in PEAK_RATES:
        raise ValueError(f"Unknown optimizer {optimizer}")
    suite = ('smoke-' if smoke else 'sweep-') + uuid.uuid4().hex[:12]
    call = run_suite.spawn(wandb_entity, suite, smoke, optimizer)
    print(f'{call.object_id}  {ROOT / suite}/comparison.json')
