"""Day 6 Parts A/B: one clearly labeled suite on one machine with two A10s.

The single-GPU A1 reference uses only GPU 0 of this allocation; GPU 1 stays
idle for that case. This keeps the host fixed but both GPUs are billed.
Start with the two-update commands in docs/day6-ddp-parts-ab.md.
"""
import json
from pathlib import Path
import subprocess
import sys
import uuid

import modal
from train_pretrain import ROOT, image as base_image, volume, wandb_secret
from distributed_training.experiment_plan import experiment_plan

app = modal.App('sqlgen-day6-ddp-parts-ab')
image = (base_image.add_local_file(ROOT / 'train_pretrain.py', '/root/train_pretrain.py')
         .add_local_dir(ROOT / 'distributed_training', '/root/distributed_training'))


def compare_final_weights(first: Path, second: Path) -> dict:
    """Compare A1/A2 weights; BF16/reduction ordering need not be bit-identical.

    Reports absolute maximum difference and relative L2 distance. Full final
    state comparison diagnoses drift; the tests check one-update math tightly.
    """
    import torch
    left = torch.load(first, map_location='cpu', weights_only=False)['model']
    right = torch.load(second, map_location='cpu', weights_only=False)['model']
    square_error = square_reference = largest = 0.0
    for name, value in left.items():
        difference = value.double() - right[name].double()
        largest = max(largest, difference.abs().max().item())
        square_error += difference.square().sum().item()
        square_reference += value.double().square().sum().item()
    return {'max_absolute_weight_difference': largest,
            'relative_l2_weight_difference': (square_error / max(square_reference, 1e-30)) ** 0.5}


@app.function(image=image, gpu='A10:2', cpu=8, memory=32768,
              timeout=86400, startup_timeout=600, scaledown_window=2,
              volumes={'/vol': volume}, secrets=[wandb_secret])
def run_suite(cases: list[dict], suite_name: str, wandb_entity: str, wandb_project: str) -> dict:
    """Execute suite cases sequentially, with torchrun handling worker failures.

    cases: Plan from experiment_plan; all workers read the same settings file.
    suite_name: Unique directory/group, so experiments cannot overwrite one another.
    wandb_entity/project: Destination for rank-0 metrics and combined charts.
    Only the parent commits the Volume, after every worker has closed its files.
    """
    import wandb
    volume.reload()
    # Cheap kernel/collective gate; abort before any curriculum run on failure.
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node=2', '-m', 'distributed_training.cuda_preflight'], check=True)
    results = []
    for case in cases:
        settings = {**case, 'run_name': f'{suite_name}-{case["label"]}', 'group': suite_name,
                    'wandb_entity': wandb_entity, 'wandb_project': wandb_project,
                    'output_dir': f'/vol/checkpoints/day6/{suite_name}/{case["label"]}'}
        settings_path = Path('/tmp/ddp-settings.json')
        settings_path.write_text(json.dumps(settings))
        # torchrun supplies RANK, LOCAL_RANK and WORLD_SIZE to each process.
        # --standalone creates a local rendezvous; NCCL handles CUDA collectives.
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        f'--nproc_per_node={case["world_size"]}',
                        '-m', 'distributed_training.ddp_worker', '--settings', str(settings_path)], check=True)
        volume.commit()
        result = json.loads((Path(settings['output_dir']) / 'result.json').read_text())
        results.append(result)
    comparison = {}
    root = Path(f'/vol/checkpoints/day6/{suite_name}')
    if cases[0]['label'].startswith('A1'):
        comparison = compare_final_weights(root / cases[0]['label'] / 'final.pt',
                                           root / cases[1]['label'] / 'final.pt')
    with wandb.init(entity=wandb_entity, project=wandb_project, group=suite_name,
                    name=f'{suite_name}-comparison', job_type='comparison') as run:
        table = wandb.Table(columns=['case', 'sql_loss', 'steady_non_pad_tokens_per_sec'])
        for case, result in zip(cases, results):
            metrics = result['last_metrics']
            table.add_data(case['label'], metrics['val/sql_loss_filtered_all'],
                           result['steady_global_non_pad_tokens_per_sec'])
        run.log({'comparison/results': table,
                 'comparison/sql_loss': wandb.plot.bar(table, 'case', 'sql_loss', title='Standalone filtered SQL validation'),
                 'comparison/throughput': wandb.plot.bar(table, 'case', 'steady_non_pad_tokens_per_sec', title='Steady global useful tokens/sec')})
        run.summary.update(comparison)
        summary = {'runs': results, 'weight_comparison': comparison, 'wandb_url': run.url}
        (root / 'comparison.json').write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.local_entrypoint()
def main(part: str = 'a', max_steps: int = 0, micro_batch_size: int = 0,
         seq_len: int = 2048, sql_fraction: float = 0.5, wandb_entity: str = '',
         wandb_project: str = 'sqlgen-day6') -> None:
    """Spawn A, B, or the A1/A2/B1/B2 combined suite (part=ab); arguments are documented in experiment_plan and the guide.

    max_steps/micro_batch_size=0 select curriculum defaults. Use --detach for
    long suites. No run-name reuse or automatic resume: each call gets a UUID.
    """
    if not wandb_entity:
        raise ValueError('Supply --wandb-entity for experiment tracking')
    cases = experiment_plan(part, max_steps, micro_batch_size, seq_len, sql_fraction)
    suite_name = f'part-{part}-{uuid.uuid4().hex[:12]}'
    print(json.dumps({'suite': suite_name, 'cases': cases}, indent=2), flush=True)
    call = run_suite.spawn(cases, suite_name, wandb_entity, wandb_project)
    print(json.dumps({'suite': suite_name, 'call_id': call.object_id,
                      'results': f'/vol/checkpoints/day6/{suite_name}/comparison.json'}), flush=True)
