"""Compare seed variation and cosine versus warmup–stable–decay schedules.

Each full case uses one A10. The CPU coordinator pins the current published
Modal dataset and launches five independent workers. --smoke runs ONE ten-update
WSD case, including a real checkpoint/resume, and verifies uploaded W&B data.
"""
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import modal
from training.config import DATA_DIR, TrainConfig, image, volume, wandb_secret
from . import EXPERIMENT_NAME
from .seed_noise import cosine_cases

SUITE_PREFIX = 'seed-and-schedules'
PROJECT = f'sqlgen-{SUITE_PREFIX}'
app = modal.App(PROJECT)
OUTPUT_ROOT = Path('/vol/checkpoints/pretraining_experiments/seed_and_schedules')
ARCHIVED_OUTPUT_ROOT = OUTPUT_ROOT.with_name('day07')


def suite_directory(suite: str) -> Path:
    """Resolve a printed suite ID to its checkpoint and result directory.

    Older IDs encode the archived directory name. Keep that lookup so the
    verification command can still read saved histories in place.
    """
    archived_prefix = f'{ARCHIVED_OUTPUT_ROOT.name}-'
    if not suite.startswith((f'{SUITE_PREFIX}-', archived_prefix)) or any(
        char not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for char in suite
    ):
        raise ValueError('Expected a printed seed and schedule suite ID')
    root = ARCHIVED_OUTPUT_ROOT if suite.startswith(archived_prefix) else OUTPUT_ROOT
    return root / suite


def experiment_plan(*, max_steps: int = 1000, smoke: bool = False) -> list[dict]:
    """Return JSON-ready cases with 4096 context and unchanged 8 × 4 batching.

    All runs use the same validation seed. WSD tails and cosine seed 1 start
    from identical model/data states. A smoke case changes duration/evaluation
    only; it keeps the full memory workload. No automatic batch downsizing.
    """
    if smoke:
        max_steps = 10
    if max_steps < 10 or max_steps % 10:
        raise ValueError('max_steps must be a positive multiple of 10')
    base = TrainConfig(mode='masked_pack', seq_len=4096, micro_batch_size=8,
                       grad_accum_steps=4, max_steps=max_steps, warmup_steps=max_steps // 10,
                       eval_every=1 if smoke else 50, val_batches=1 if smoke else 128,
                       seed=1, validation_seed=12345)
    cases = cosine_cases(base)
    cases += [{'label': f'wsd-{tail}-seed-1',
               'train': {**asdict(base), 'lr_schedule': 'wsd', 'wsd_decay_type': tail}}
              for tail in ('linear', 'cosine')]
    return cases[3:4] if smoke else cases


def _train_case(case: dict, suite: str, dataset: dict, entity: str, project: str, smoke: bool) -> dict:
    """Train one pinned case; the smoke path proves resume without changing LR."""
    from training.trainer import run_training as _run
    config = TrainConfig(**case['train'])
    output = suite_directory(suite) / case['label']
    options = dict(output_dir=output, group=suite, title=f'{EXPERIMENT_NAME} · {case["label"]}',
                   expected_dataset=dataset,
                   milestone_updates=(config.warmup_steps, config.max_steps // 2,
                                      config.max_steps * 8 // 10))
    if smoke:
        _run(config, entity, project, case['label'], stop_after_update=2, **options)
    result = _run(config, entity, project, case['label'], **options)
    return result


@app.function(image=image, gpu='A10', cpu=4, memory=16_384, timeout=86_400,
              startup_timeout=600, scaledown_window=2, max_containers=5,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def train_case(case: dict, suite: str, dataset: dict, entity: str, project: str) -> dict:
    """Execute one full case; five cases can occupy separate A10s concurrently."""
    return _train_case(case, suite, dataset, entity, project, False)


@app.function(image=image, gpu='A10', cpu=4, memory=16_384, timeout=900,
              startup_timeout=600, scaledown_window=2, max_containers=1,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def smoke_case(case: dict, suite: str, dataset: dict, entity: str, project: str) -> dict:
    """Bound correctness/resume/upload verification to fifteen minutes on one A10."""
    return _train_case(case, suite, dataset, entity, project, True)


@app.function(image=image, cpu=2, memory=4096, timeout=86_400,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def run_suite(cases: list[dict], suite: str, entity: str, project: str, smoke: bool,
              analyze_results: bool = True) -> dict:
    """Pin one completed dataset, launch cases in parallel, and retain evidence."""
    from training.dataset import dataset_identity
    volume.reload()
    dataset = dataset_identity(DATA_DIR)
    root = suite_directory(suite)
    warmup = all('baseline_label' in case for case in cases)
    baselines = []
    if warmup:
        from .warmup_comparison import BASELINE_SUITE, load_baselines
        baselines = load_baselines(suite_directory(BASELINE_SUITE), cases, dataset)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'plan.json').write_text(json.dumps({'cases': cases, 'dataset': dataset}, indent=2))
    volume.commit()
    if smoke:
        results = [smoke_case.remote(cases[0], suite, dataset, entity, project)]
    else:
        results = list(train_case.starmap([(case, suite, dataset, entity, project) for case in cases]))
    volume.reload()
    if not analyze_results:
        # Keep only the requested training runs in W&B. The CPU parent merely
        # owns their detached lifetime and records their links; no comparison
        # run, report generation or API verification job follows.
        summary = {'suite': suite, 'dataset_identity': dataset,
                   'runs': [{'run': result['run'], 'wandb_url': result['wandb_url'],
                             'updates': result['updates']} for result in results]}
        (root / 'summary.json').write_text(json.dumps(summary, indent=2))
        volume.commit()
        return summary
    from training.tracking import verify_uploaded_result
    for result in results:
        check = verify_uploaded_result(result)
        (root / result['run'] / 'tracking-check.json').write_text(json.dumps(check, indent=2))
        result['tracking_check'] = check
    summary = {'suite': suite, 'dataset_identity': dataset,
               'runs': [{key: value for key, value in result.items() if key != 'history'} for result in results]}
    if not smoke:
        import wandb
        if warmup:
            from .warmup_comparison import publish_comparison, report_markdown
        else:
            from .analysis import publish_comparison, report_markdown
        with wandb.init(entity=entity, project=project, group=suite,
                        name=f'{EXPERIMENT_NAME} · comparison', job_type='comparison',
                        config={'dataset_identity': dataset, 'suite': suite}) as run:
            analysis = (publish_comparison(wandb, run, cases, baselines, results) if warmup
                        else publish_comparison(wandb, run, results))
            summary['comparison_url'] = run.url
        (root / 'analysis.json').write_text(json.dumps(analysis, indent=2))
        (root / 'takeaways.md').write_text(report_markdown(analysis))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.function(image=image, cpu=1, memory=4096, timeout=600,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def recheck_tracking(suite: str) -> dict:
    """Verify an existing suite's saved results without allocating any GPU."""
    from training.tracking import verify_uploaded_result
    root = suite_directory(suite)
    volume.reload()
    paths = sorted(root.glob('*/result.json'))
    if not paths:
        raise FileNotFoundError(f'No completed case results under {root}')
    checks = {}
    for path in paths:
        result = json.loads(path.read_text())
        checks[result['run']] = verify_uploaded_result(result)
        (path.parent / 'tracking-check.json').write_text(json.dumps(checks[result['run']], indent=2))
    (root / 'tracking-checks.json').write_text(json.dumps(checks, indent=2))
    volume.commit()
    return checks


@app.local_entrypoint()
def main(wandb_entity: str = '', wandb_project: str = PROJECT, smoke: bool = False,
         max_steps: int = 1000, dry_run: bool = False, verify_suite: str = '',
         warmup_comparison: bool = False, analyze_results: bool = True) -> None:
    """Print a reviewable plan or spawn a durable coordinator; never wait locally.

    Use --smoke first and inspect tracking-check.json/the linked charts. A full
    default suite uses 5 × 131,072,000 training token slots; this does not include
    validation, startup, checkpoint IO, or document-boundary masked targets.
    """
    if verify_suite:
        print(json.dumps(recheck_tracking.remote(verify_suite), indent=2))
        return
    if warmup_comparison:
        if smoke or max_steps != 1000:
            raise ValueError('Warmup comparison requires 1000 full updates and no smoke run')
        from .warmup_comparison import experiment_plan as warmup_plan
        cases = warmup_plan()
    else:
        cases = experiment_plan(max_steps=max_steps, smoke=smoke)
    print(json.dumps({'cases': cases, 'token_slots_per_case': cases[0]['train']['max_steps'] * 8 * 4 * 4096,
                      'parallel_a10s': len(cases)}, indent=2))
    if dry_run:
        return
    if not wandb_entity:
        raise ValueError('Supply --wandb-entity for experiment tracking')
    suite = f'{SUITE_PREFIX}-{"smoke" if smoke else "full"}-{uuid.uuid4().hex[:12]}'
    if warmup_comparison:
        suite = f'{SUITE_PREFIX}-warmup300-{uuid.uuid4().hex[:12]}'
    call = run_suite.spawn(cases, suite, wandb_entity, wandb_project, smoke, analyze_results)
    print(json.dumps({'suite': suite, 'call_id': call.object_id,
                      'results': str(suite_directory(suite) / 'summary.json')}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preview the seed and schedule experiment locally.')
    parser.add_argument('--dry-run', action='store_true', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--warmup-comparison', action='store_true')
    options = parser.parse_args()
    main(dry_run=options.dry_run, smoke=options.smoke, warmup_comparison=options.warmup_comparison)
