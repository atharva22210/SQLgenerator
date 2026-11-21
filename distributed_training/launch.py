"""Distributed training experiments on one machine with two A10s.

The single-GPU A1 reference uses only GPU 0 of this allocation; GPU 1 stays
idle for that case. This keeps the host fixed but both GPUs are billed.
Start with the two-update commands in distributed_training/README.md.

Process ownership:
    laptop: CLI builds a plan and .spawn() submits one cloud invocation
    Modal parent: allocates two A10s and runs cases sequentially
    torchrun: creates one worker per GPU used by the current case
    rank 0 worker: logs global metrics and writes this case's final snapshot
    Modal parent: commits completed files and creates suite comparison charts

Training data and model code are shared with the packing experiment. The cloud image fixes dependencies via the existing uv lock.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

import modal
from data.grouped_corpus import resolve_grouped_sql
from training_common import DATA_DIR, image as base_image, volume, wandb_secret
from distributed_training.experiment_plan import experiment_plan

app = modal.App('sqlgen-distributed-training')
# Dependency installation and code shipping are different: the locked image
# supplies PyTorch etc.; these additions make our Python modules importable
# under /root for the subprocesses launched by torchrun.
image = base_image.add_local_python_source('distributed_training')


# Stable labels remain the CLI selectors and output paths. These titles are
# shared by the W&B run list, comparison legends, and results tables.
CASE_TITLES = {
    'A1-single-reference': 'A1 · Single-GPU reference',
    'A2-ddp-distinct-draws': 'A2 · Two-GPU DDP, distinct data',
    'A3-ddp-duplicate-draws': 'A3 · Two-GPU DDP, duplicate data',
    'B1-sync-every-microbatch': 'B1 · DDP, sync every microbatch',
    'B2-sync-last-microbatch': 'B2 · DDP, sync on final microbatch',
    'C3-ddp-baseline': 'C3 · DDP baseline',
    'C3-fix1-ddp-checkpoint': 'C3 · DDP + activation checkpointing',
    'C3-fix2-shard-grad-op': 'C3 · FSDP gradient and optimizer sharding (ZeRO-2)',
    'C3-fix3-full-shard': 'C3 · FSDP full sharding (ZeRO-3)',
    'C3-fix4-full-shard-checkpoint': 'C3 · FSDP full sharding + activation checkpointing',
}


def case_title(case: dict) -> str:
    """Describe a planned case, including its checkpoint setting when enabled."""
    title = CASE_TITLES[case['label']]
    if case.get('checkpoint_blocks'):
        return f'{title} ({case["checkpoint_blocks"]}/{case["model"]["n_layers"]} blocks)'
    return title


def compare_final_weights(first: Path, second: Path) -> dict:
    """Compare A1/A2 weights; BF16/reduction ordering need not be bit-identical.

    Args:
        first: A1 final.pt snapshot, holding the single-process reference.
        second: A2 final.pt snapshot, holding the distributed result.
    Returns:
        Maximum absolute parameter difference and ||A1-A2||_2 / ||A1||_2.
        These summarize final drift; they do not assert a long BF16 trajectory
        should be bit-identical or replace the one-update gradient tests.

    Reports absolute maximum difference and relative L2 distance. Full final
    state comparison diagnoses drift; the tests check one-update math tightly.
    """
    import torch
    left = torch.load(first, map_location='cpu', weights_only=False)['model']
    right = torch.load(second, map_location='cpu', weights_only=False)['model']
    # Accumulate scalar reductions one parameter at a time in CPU float64.
    # Avoid concatenating a second huge flattened model merely to compare it.
    # These snapshots are our own trusted experiment outputs (optimizer and
    # settings are included, hence weights_only=False).
    square_error = square_reference = largest = 0.0
    for name, value in left.items():
        difference = value.double() - right[name].double()
        largest = max(largest, difference.abs().max().item())
        square_error += difference.square().sum().item()
        square_reference += value.double().square().sum().item()
    return {'max_absolute_weight_difference': largest,
            'relative_l2_weight_difference': (square_error / max(square_reference, 1e-30)) ** 0.5}


# Modal memory=32768 is HOST memory in MiB, not GPU VRAM. gpu='A10:2'
# reserves two actual devices. volumes maps the persistent volume to /vol;
# secrets injects credentials into the container without putting keys in code.
# timeout bounds execution; startup_timeout separately bounds container startup.
@app.function(image=image, gpu='A10:2', cpu=8, memory=32768,
              timeout=86400, startup_timeout=600, scaledown_window=2,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def run_suite(cases: list[dict], suite_name: str, wandb_entity: str, wandb_project: str) -> dict:
    """Execute suite cases sequentially, with torchrun handling worker failures.

    cases: Plan from experiment_plan; all workers read the same settings file.
    suite_name: Unique storage directory for this submission.
    wandb_entity/project: Destination for rank-0 metrics and combined charts.
    Only the parent commits the Volume, after every worker has closed its files.
    Return value is a JSON-compatible suite summary. A subprocess failure raises
    immediately, except for an explicit CUDA OOM recorded in a Part C case.
    The Part C suite records that outcome and proceeds to the next case.
    Already committed earlier cases remain available if a later case fails.
    """
    import wandb
    volume.reload()
    # Bind the full suite to one completed split even if another build
    # publishes during the experiment.
    requested = cases[0].get('grouped_sql_dir')
    grouped_sql_dir, _ = resolve_grouped_sql(DATA_DIR, Path(requested) if requested else None)
    # Cheap kernel/collective gate; abort before any curriculum run on failure.
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node=2', '-m', 'distributed_training.cuda_preflight'], check=True)
    part_c = cases[0].get('part_c', False)
    topics = {'A': 'DDP correctness', 'B': 'DDP gradient synchronization', 'C': 'Sharding and memory'}
    purpose = ' + '.join(dict.fromkeys(topics[case['label'][0]] for case in cases))
    # A timestamp distinguishes repeated launches in the workspace. Use the
    # stable corpus ID in config and a short prefix for scanning the run list.
    started = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    first_train = cases[0]['train']
    group = (f'{purpose} · Grouped SQL {grouped_sql_dir.name[:8]} · '
             f'{first_train["seq_len"]:,} tokens · {started}')
    tracking_config = {'suite_id': suite_name, 'grouped_sql_dir': str(grouped_sql_dir),
                       'corpus_id': grouped_sql_dir.name}
    results = []
    # One enclosing remote function makes sequencing durable after the laptop
    # disconnects. Spawning each case separately without dependencies would
    # allow them to overlap and invalidate the intended sequential comparison.
    for case in cases:
        settings = {**case, **tracking_config, 'run_name': case_title(case),
                    'group': group,
                    'wandb_entity': wandb_entity, 'wandb_project': wandb_project,
                    'output_dir': f'/vol/checkpoints/distributed_training/{suite_name}/{case["label"]}'}
        # All ranks read this same small JSON. Reusing its temporary filename
        # is safe because the previous torchrun subprocess has finished before
        # the next case overwrites it. Checkpoints use unique persistent paths.
        settings_path = Path('/tmp/ddp-settings.json')
        settings_path.write_text(json.dumps(settings))
        # torchrun supplies RANK, LOCAL_RANK and WORLD_SIZE to each process.
        # --standalone creates a local rendezvous; NCCL handles CUDA collectives.
        process = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                                  f'--nproc_per_node={case["world_size"]}',
                                  '-m', 'distributed_training.ddp_worker', '--settings', str(settings_path)])
        output = Path(settings['output_dir'])
        if process.returncode:
            oom = {path.stem: json.loads(path.read_text()) for path in output.glob('oom-rank-*.json')}
            if not part_c or not oom:
                process.check_returncode()
            result = {'run_name': settings['run_name'], 'oom': True, 'oom_details': oom,
                      'parameter_count': next(iter(oom.values()))['parameter_count']}
            (output / 'result.json').write_text(json.dumps(result, indent=2))

        # Worker processes have exited, so their final writes are complete.
        # Publish this case before starting the next, not only at suite end.
        volume.commit()
        result = json.loads((output / 'result.json').read_text())
        results.append(result)
    comparison = {}
    root = Path(f'/vol/checkpoints/distributed_training/{suite_name}')
    if cases[0]['label'].startswith('A1'):
        comparison = compare_final_weights(root / cases[0]['label'] / 'final.pt',
                                           root / cases[1]['label'] / 'final.pt')
    comparison_url = None
    if len(cases) > 1:
        with wandb.init(entity=wandb_entity, project=wandb_project, group=group,
                        name=f'{purpose} · Compare training methods',
                        job_type='comparison', config=tracking_config) as run:
            if part_c:
                log_part_c_comparison(wandb, run, cases, results)
            else:
                log_ab_comparison(wandb, run, cases, results)
            run.summary.update(comparison)
            comparison_url = run.url
    summary = {'runs': results, 'weight_comparison': comparison, 'wandb_url': comparison_url or results[0].get('wandb_url'),
               'grouped_sql_dir': str(grouped_sql_dir), 'wandb_group': group}
    (root / 'comparison.json').write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


def log_ab_comparison(wandb, run, cases: list[dict], results: list[dict]) -> None:
    # Compare the final FULL filtered SQL pass, not the smaller
    # periodic sample. Throughput excludes validation/IO and initial warmup
    # updates; its scope is documented in the worker and execution guide.
    table = wandb.Table(columns=['case', 'sql_loss', 'steady_non_pad_tokens_per_sec'])
    for case, result in zip(cases, results):
        metrics = result['last_metrics']
        table.add_data(case_title(case), metrics['val/sql_loss_filtered_all'],
                       result['steady_global_non_pad_tokens_per_sec'])
    curves = [result['validation_curve'] for result in results]
    validation_plot = wandb.plot.line_series(
        xs=[[point['update'] for point in curve] for curve in curves],
        ys=[[point['loss'] for point in curve] for curve in curves],
        keys=[case_title(case) for case in cases],
        title='Sampled held-out SQL validation loss',
        xname='Optimizer update',
    )
    run.log({'comparison/sql_loss_curve': validation_plot,
             'comparison/results': table,
             'comparison/sql_loss': wandb.plot.bar(table, 'case', 'sql_loss', title='Standalone filtered SQL validation'),
             'comparison/throughput': wandb.plot.bar(table, 'case', 'steady_non_pad_tokens_per_sec', title='Steady global useful tokens/sec')})


def log_part_c_comparison(wandb, run, cases: list[dict], results: list[dict]) -> None:
    """Plot sampled loss curves and per-rank GPU activity in the suite run."""
    table = wandb.Table(columns=['case', 'oom', 'params', 'predicted_ddp_state_gib',
                                 'gpu0_peak_gib', 'gpu1_peak_gib', 'steady_non_pad_tokens_per_sec'])
    completed = [(case_title(case), result) for case, result in zip(cases, results) if not result['oom']]
    plots = {}
    for case, result in zip(cases, results):
        peaks = [value / 1024**3 if value is not None else None
                 for value in result.get('rank_peak_allocated_bytes', [None, None])]
        params = result.get('parameter_count')
        table.add_data(case_title(case), result['oom'], params, params * 16 / 1024**3 if params else None,
                       *(peaks if not result['oom'] else [None, None]),
                       result.get('steady_global_non_pad_tokens_per_sec'))

    plots['comparison/results'] = table
    if completed:
        for source, label in [('train_curve', 'Training loss'), ('validation_curve', 'Sampled SQL loss')]:
            curves = [result[source] for _, result in completed]
            plots[f'comparison/{source}'] = wandb.plot.line_series(
                xs=[[point['update'] for point in curve] for curve in curves],
                ys=[[point['loss'] for point in curve] for curve in curves],
                keys=[name for name, _ in completed], title=label, xname='Optimizer update')
        for metric, title in (
            ('nccl_kernel_ms', 'Time in NCCL communication kernels by GPU (ms)'),
            ('other_kernel_ms', 'Time in other GPU kernels by GPU (ms)'),
            ('overlap_ms', 'Communication and other kernel overlap by GPU (ms)'),
        ):
            profile_table = wandb.Table(columns=['case and GPU', 'milliseconds'])
            for name, result in completed:
                for rank, profile in enumerate(result['profile_by_rank']):
                    profile_table.add_data(f'{name} GPU {rank}', profile[metric])
            plots[f'comparison/{metric}'] = wandb.plot.bar(
                profile_table, 'case and GPU', 'milliseconds', title=title)
    run.log(plots)


@app.local_entrypoint()
def main(part: str = 'a', max_steps: int = 0, micro_batch_size: int = 0,
         seq_len: int = 4096, sql_fraction: float = 0.5, wandb_entity: str = '',
         wandb_project: str = 'sqlgen-distributed-training', grouped_sql_dir: str = '',
         eval_every: int = 0, val_batches: int = 128,
         checkpoint_blocks: int = 12, accumulation: int = 2, case: str = '') -> None:
    """Spawn A/B, their combined suite, or the Part C memory experiment.

    Args:
        part: 'a' includes A1/A2/A3, 'b' includes B1/B2, 'ab' is A1/A2/B1/B2.
        max_steps: Optimizer updates per case; 0 keeps A=300 and B=100.
        micro_batch_size: Sequences per GPU per microstep; 0 keeps A=8/B=4.
        seq_len: Fixed token slots per row (4096 normally, 64 for small checks).
        sql_fraction: SQL document sampling probability during TRAINING.
            Validation remains filtered SQL-only regardless of this value.
        wandb_entity/wandb_project: Tracking destination, not credentials.
        grouped_sql_dir: Pin a completed build. Default picks the latest Grouped SQL corpus.
        eval_every: Optimizer updates between sampled validation points.
        val_batches: Documents per fixed validation sample.
        checkpoint_blocks: Part C block count for both checkpointed cases.
        accumulation: Part C microbatches per GPU per optimizer update.
        case: Optional exact Part C label, for rerunning one case by itself.
    Returns:
        None; prints a durable Modal function-call ID and result directory.

    max_steps/micro_batch_size=0 select curriculum defaults. Use --detach for
    long suites. No run-name reuse or automatic resume: each call gets a UUID.
    """
    if not wandb_entity:
        raise ValueError('Supply --wandb-entity for experiment tracking')
    eval_every = eval_every or (5 if part == 'c' else 25)
    cases = experiment_plan(part, max_steps, micro_batch_size, seq_len, sql_fraction,
                            eval_every=eval_every, val_batches=val_batches,
                            checkpoint_blocks=checkpoint_blocks, accumulation=accumulation)
    if case:
        if part != 'c':
            raise ValueError('--case requires --part c')
        matching = [planned for planned in cases if planned['label'] == case]
        if not matching:
            labels = ', '.join(planned['label'] for planned in cases)
            raise ValueError(f'Unknown Part C case {case!r}. Choose one of {labels}')
        cases = matching
    if grouped_sql_dir:
        directory = Path(grouped_sql_dir)
        if directory.parent != DATA_DIR / 'grouped_sql' or len(directory.name) != 32 or any(
            char not in '0123456789abcdef' for char in directory.name
        ):
            raise ValueError('Expected a completed corpus under the configured training build grouped_sql directory')
        cases = [{**planned, 'grouped_sql_dir': str(directory)} for planned in cases]
    suite_labels = {'a': 'ddp-correctness', 'b': 'ddp-gradient-sync', 'ab': 'ddp-correctness-and-sync',
                    'c': 'sharding-and-memory'}
    suite_name = f'{suite_labels[part]}-{uuid.uuid4().hex[:12]}'
    print(json.dumps({'suite': suite_name, 'cases': cases}, indent=2), flush=True)
    # spawn returns a handle immediately. remote() would wait for the return
    # value; call.get() could retrieve it later, but we deliberately do not wait
    # here. --detach keeps this submitted suite alive after CLI disconnection.
    call = run_suite.spawn(cases, suite_name, wandb_entity, wandb_project)
    print(json.dumps({'suite': suite_name, 'call_id': call.object_id,
                      'results': f'/vol/checkpoints/distributed_training/{suite_name}/comparison.json'}), flush=True)
