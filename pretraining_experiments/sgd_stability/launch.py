"""Explicit six-case submission; shared trainer, bounded GPUs, CPU-only analysis."""
import argparse
import json
from pathlib import Path
import uuid
import modal
from training.config import DATA_DIR, TrainConfig, image, volume, wandb_secret
from training.errors import NumericalDivergence
from .plan import PROBE_SEED, TOKEN_BUDGET, experiment_plan

app = modal.App('sqlgen-sgd-stability')
ROOT = Path('/vol/checkpoints/pretraining_experiments/sgd_stability')
PROJECT = 'sqlgen-sgd-stability'


def milestones(config: TrainConfig) -> tuple[int, ...]:
    """Retain early and later states, even if a high LR fails before the finish."""
    return tuple(sorted({1, config.warmup_steps, config.max_steps // 4,
                         config.max_steps // 2, config.max_steps}))


def execute_case(case, suite, dataset, entity):
    """Run one fresh case; only nonfinite numerical failures are recoverable."""
    from training.trainer import run_training
    output = ROOT / suite / case['label']
    output.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(**case['train'])
    try:
        result = run_training(config, entity, PROJECT, case['label'], output_dir=output,
                              group=suite, expected_dataset=dataset, x_axis='tokens_seen',
                              milestone_updates=milestones(config), probe_every=1, probe_seed=PROBE_SEED)
        result['status'] = 'completed'
    except NumericalDivergence as error:
        path = output / 'metrics.jsonl'
        history = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        result = {'run': case['label'], 'status': 'diverged', 'completed': False,
                  'config': case['train'], 'dataset_identity': dataset, 'history': history,
                  'failed_update': error.update, 'reason': str(error), 'wandb_url': None}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=image, gpu='A10', cpu=4, memory=16384, timeout=86400,
              max_containers=2, scaledown_window=2,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def train_case(case, suite, dataset, entity):
    """One independent model per GPU; effective batch grows by accumulation."""
    return execute_case(case, suite, dataset, entity)


@app.function(image=image, cpu=2, memory=4096, timeout=86400,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def run_suite(suite, entity):
    """Pin the current dataset once, run three pairs, verify charts without GPUs."""
    import wandb
    from training.dataset import dataset_identity
    from training.tracking import verify_uploaded_result
    from .analysis import analyze, publish, report_markdown
    volume.reload()
    dataset = dataset_identity(DATA_DIR)
    cases = experiment_plan()
    root = ROOT / suite
    root.mkdir(parents=True, exist_ok=True)
    plan = {'cases': cases, 'dataset': dataset, 'probe_recipe': {'every': 1, 'seed': PROBE_SEED},
            'prediction': 'The wider SGD LR bracket may expose persistent fluctuations; this is not guaranteed.',
            'limits': 'One seed; clipping and decay retained; cosine LR; no direct Hessian measurement.',
            'checkpoints': {c['label']: {str(u):str(root / c['label'] / f'step-{u:06d}.pt')
                                       for u in milestones(TrainConfig(**c['train']))} for c in cases}}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))
    volume.commit()
    results = []
    for start in range(0, len(cases), 2):
        results.extend(train_case.starmap([(case, suite, dataset, entity) for case in cases[start:start+2]]))
        volume.reload()
        (root / 'progress.json').write_text(json.dumps([{k:v for k,v in r.items() if k != 'history'} for r in results], indent=2))
        volume.commit()
    analysis = analyze(results)
    (root / 'analysis.json').write_text(json.dumps(analysis, indent=2))
    (root / 'takeaways.md').write_text(report_markdown(analysis))
    volume.commit()
    for result in results:
        if result['status'] == 'completed':
            check = verify_uploaded_result(result)
            (root / result['run'] / 'tracking-check.json').write_text(json.dumps(check, indent=2))
    with wandb.init(entity=entity, project=PROJECT, group=suite, job_type='comparison',
                    name='SGD batch and LR stability', config={'dataset_identity': dataset, 'suite': suite}) as run:
        publish(wandb, run, results)
        url = run.url
    summary = {'suite': suite, 'comparison_url': url, 'cases': 6,
               'completed': sum(r['status'] == 'completed' for r in results), 'dataset_identity': dataset}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.local_entrypoint()
def main(wandb_entity: str = '', launch: bool = False):
    """Preview by default. --launch allocates at most two A10 workers."""
    cases = experiment_plan()
    print(json.dumps({'cases': cases, 'runs': len(cases), 'total_training_tokens': len(cases)*TOKEN_BUDGET,
                      'max_parallel_a10s': 2}, indent=2))
    if not launch:
        return
    if not wandb_entity:
        raise ValueError('Supply --wandb-entity before launch')
    suite = f'sgd-stability-{uuid.uuid4().hex[:12]}'
    call = run_suite.spawn(suite, wandb_entity)
    print(json.dumps({'suite': suite, 'call_id': call.object_id, 'results': str(ROOT / suite)}))


if __name__ == '__main__':
    argparse.ArgumentParser(description='Local preview; use Modal --launch to submit.').parse_args()
    main()
