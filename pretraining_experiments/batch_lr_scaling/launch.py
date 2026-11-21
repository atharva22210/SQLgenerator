"""Preview or explicitly launch a fixed-token batch-size × learning-rate sweep.

Importing this file and the default local preview never submit training. The
shared top-level train.py remains the only single-GPU optimization loop.
"""
import argparse
import json
from pathlib import Path
import uuid

import modal
from training.config import DATA_DIR, TrainConfig, image, volume, wandb_secret
from training.errors import NumericalDivergence
from .plan import BATCHES, LEARNING_RATES, TOKEN_BUDGET, experiment_plan

app = modal.App('sqlgen-batch-lr-scaling')
OUTPUT_ROOT = Path('/vol/checkpoints/pretraining_experiments/batch_lr_scaling')
PROJECT = 'sqlgen-batch-lr-scaling'


def execute_case(case: dict, suite: str, dataset: dict, entity: str, project: str) -> dict:
    """Run one configuration; only explicit numerical divergence is recoverable.

    CUDA OOMs, missing data, authentication errors and coding errors propagate.
    Treating every exception as an unstable LR would hide infrastructure bugs.
    A failed case keeps its last valid checkpoint and the exact failure update.
    """
    from train import run_training
    output = OUTPUT_ROOT / suite / case['label']
    output.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(**case['train'])
    try:
        result = run_training(config, entity, project, case['label'], output_dir=output,
                              group=suite, title=case['label'], expected_dataset=dataset,
                              milestone_updates=(config.warmup_steps,), x_axis='tokens_seen')
        result['status'] = 'completed'
    except NumericalDivergence as error:
        history_path = output / 'metrics.jsonl'
        history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
        result = {'run': case['label'], 'status': 'diverged', 'completed': False,
                  'config': case['train'], 'dataset_identity': dataset,
                  'failed_update': error.update, 'reason': str(error), 'history': history,
                  'wandb_url': None}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=image, gpu='A10', cpu=4, memory=16_384, timeout=86_400,
              startup_timeout=600, scaledown_window=2, max_containers=5,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def train_case(case: dict, suite: str, dataset: dict, entity: str, project: str) -> dict:
    """One independent model on one A10; batch growth uses accumulation only."""
    return execute_case(case, suite, dataset, entity, project)


@app.function(image=image, cpu=2, memory=4096, timeout=86_400,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def run_suite(cases: list[dict], suite: str, entity: str, project: str, max_parallel: int) -> dict:
    """Pin the data once, bound concurrent GPU calls, then analyze on CPU.

    Cases can be a selected subset of the 16-cell grid. Persist every completed
    chunk so a later infrastructure failure does not erase earlier evidence.
    Concurrency changes wall time, not the total training-token budget or price
    of an individual GPU-second. No GPU is reserved for chart verification.
    """
    import wandb
    from training.dataset import dataset_identity
    from training.tracking import verify_uploaded_result
    from .analysis import analyze, publish, report_markdown
    if not 1 <= max_parallel <= 5:
        raise ValueError('max_parallel must be between 1 and 5')
    volume.reload()
    dataset = dataset_identity(DATA_DIR)
    root = OUTPUT_ROOT / suite
    root.mkdir(parents=True, exist_ok=True)
    (root / 'plan.json').write_text(json.dumps({'cases': cases, 'dataset': dataset,
                                               'max_parallel': max_parallel}, indent=2))
    volume.commit()
    results = []
    for start in range(0, len(cases), max_parallel):
        chunk = cases[start:start + max_parallel]
        results.extend(train_case.starmap([(case, suite, dataset, entity, project) for case in chunk]))
        volume.reload()
        (root / 'progress.json').write_text(json.dumps([{k: v for k, v in r.items() if k != 'history'} for r in results], indent=2))
        volume.commit()
    # Save the numeric analysis before online verification; a transient W&B
    # outage must not hide already completed training results.
    measured = analyze(cases, results)
    (root / 'analysis.json').write_text(json.dumps(measured, indent=2))
    (root / 'takeaways.md').write_text(report_markdown(measured))
    volume.commit()
    for result in results:
        if result['status'] == 'completed':
            check = verify_uploaded_result(result)
            (root / result['run'] / 'tracking-check.json').write_text(json.dumps(check, indent=2))
    with wandb.init(entity=entity, project=project, group=suite, job_type='comparison',
                    name='Batch size and LR comparison', config={'dataset_identity': dataset, 'suite': suite}) as run:
        analysis = publish(wandb, run, cases, results)
        comparison_url = run.url
    (root / 'analysis.json').write_text(json.dumps(analysis, indent=2))
    (root / 'takeaways.md').write_text(report_markdown(analysis))
    summary = {'suite': suite, 'comparison_url': comparison_url, 'cases': len(cases),
               'completed': sum(r['status'] == 'completed' for r in results), 'dataset_identity': dataset}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


def select_cases(batch_sizes: str = '', learning_rates: str = '') -> list[dict]:
    """Parse optional comma-separated subsets; do not silently invent grid cells."""
    batches = tuple(int(x.strip()) for x in batch_sizes.split(',')) if batch_sizes else BATCHES
    rates = tuple(float(x.strip()) for x in learning_rates.split(',')) if learning_rates else LEARNING_RATES
    return experiment_plan(batches, rates)


@app.local_entrypoint()
def main(wandb_entity: str = '', wandb_project: str = PROJECT, launch: bool = False,
         batch_sizes: str = '', learning_rates: str = '', max_parallel: int = 2) -> None:
    """Default to preview; --launch explicitly submits the selected sweep."""
    cases = select_cases(batch_sizes, learning_rates)
    if not 1 <= max_parallel <= 5:
        raise ValueError('max_parallel must be between 1 and 5')
    print(json.dumps({'cases': cases, 'runs': len(cases), 'total_training_token_slots': len(cases) * TOKEN_BUDGET,
                      'max_parallel_a10s': min(len(cases), max_parallel)}, indent=2))
    if not launch:
        return
    if not wandb_entity:
        raise ValueError('Supply --wandb-entity before launching')
    suite = f'batch-lr-{uuid.uuid4().hex[:12]}'
    call = run_suite.spawn(cases, suite, wandb_entity, wandb_project, max_parallel)
    print(json.dumps({'suite': suite, 'call_id': call.object_id,
                      'results': str(OUTPUT_ROOT / suite / 'summary.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local-only preview; use Modal --launch to submit later.')
    parser.add_argument('--batch-sizes', default='')
    parser.add_argument('--learning-rates', default='')
    args = parser.parse_args()
    main(batch_sizes=args.batch_sizes, learning_rates=args.learning_rates)
