"""Sequential paired validation for the four-case Muon follow-up."""
from dataclasses import replace
import json
from pathlib import Path
import uuid

import modal

from training.config import TrainConfig, image, volume, wandb_secret
from training.trainer import run_training
from training.tracking import verify_uploaded_result
from .probe import AttentionProbe

app = modal.App('sqlgen-muon-qk-decay-ema')
PROJECT = 'sqlgen-muon-qk-decay-ema'
ROOT = Path('/vol/checkpoints/pretraining_experiments/muon_qk_decay_ema')
BASELINE = Path('/vol/checkpoints/pretraining_experiments/optimizer_batch_lr/sweep-cdc968400e2b/muon-batch-32768-lr-0.01/settings.json')
MILESTONES = (100, 200, 400, 600, 800, 1000)


@app.function(image=image, gpu='A10', cpu=4, memory=16384, timeout=1800,
              max_containers=1, scaledown_window=2, volumes={'/vol': volume}, secrets=[wandb_secret])
def train(entity, suite, cfg, dataset, smoke):
    """Train one fresh case, or resume the smoke after its first saved update."""
    config = TrainConfig(**cfg)
    name = f'qk-{int(config.use_qk_norm)}-decay-{config.weight_decay:g}'
    directory = ROOT / suite / name
    args = dict(output_dir=directory, group=suite, expected_dataset=dataset,
                x_axis='tokens_seen', probe_factory=AttentionProbe,
                milestone_updates=(1, 2) if smoke else MILESTONES)
    if smoke:
        run_training(config, entity, PROJECT, name, stop_after_update=1, **args)
    result = run_training(config, entity, PROJECT, name,
                          stop_after_update=2 if smoke else None, **args)
    for step in args['milestone_updates']:
        if not (directory / f'step-{step:06d}.pt').is_file():
            raise RuntimeError(f'Missing checkpoint at update {step}')
    return result


@app.function(image=image, timeout=10800, volumes={'/vol': volume}, secrets=[wandb_secret])
def suite_run(entity, suite, smoke):
    """Use raw zero-decay loss to select QK, then fix evaluation weights for decay."""
    import wandb
    volume.reload()
    settings = json.loads(BASELINE.read_text())
    baseline = TrainConfig(**{k: settings[k] for k in TrainConfig.__dataclass_fields__ if k in settings})
    baseline = replace(baseline, ema_half_life_tokens=1_638_400)
    rows = []
    directory = ROOT / suite
    directory.mkdir(parents=True, exist_ok=True)
    volume.commit()

    def case(qk, decay):
        config = replace(baseline, use_qk_norm=qk, weight_decay=decay)
        result = train.remote(entity, suite, vars(config), settings['dataset_identity'], smoke)
        volume.reload()
        row = verify_uploaded_result(result)
        history = result['history']
        final = history[-1]
        # Confirm the new paired and attention metrics reached the server too.
        api_run = wandb.Api().run(result['wandb_path'])
        for key in ('val/ema_sql_loss', 'val/ema_english_loss', 'probe/max_score', 'probe/min_p05_entropy'):
            received = float(api_run.summary[key])
            if abs(received - final[key]) > 1e-5 * max(1, abs(final[key])):
                raise RuntimeError(f'Unexpected uploaded {key} for {result["run"]}')
        probe_files = sorted((directory / result['run'] / 'probe').glob('step-*.json'))
        with wandb.init(entity=entity, project=PROJECT, group=suite,
                        name=f"{result['run']}-diagnostics", job_type='diagnostics') as diag:
            table = wandb.Table(columns=['update', 'tokens', 'document', 'layer', 'head',
                                         'max_score', 'p99_score', 'median_entropy', 'p05_entropy',
                                         'q_before', 'q_after', 'k_before', 'k_after'])
            for path in probe_files:
                update = int(path.stem.split('-')[1])
                for head in json.loads(path.read_text()):
                    table.add_data(update, update * 32768, *(head[k] for k in table.columns[2:]))
            diag.log({'attention_heads': table})
            for corpus in ('sql', 'english'):
                evaluations = [r for r in history if f'val/{corpus}_loss' in r]
                chart = wandb.plot.line_series(
                    xs=[r['tokens_seen'] for r in evaluations],
                    ys=[[r[f'val/{prefix}{corpus}_loss'] for r in evaluations]
                        for prefix in ('', 'ema_')], keys=['raw', 'ema'],
                    title=f'{corpus} validation loss', xname='Training tokens')
                diag.log({f'comparison/{corpus}': chart})
            row['diagnostics_url'] = diag.url
        row.update(name=result['run'], qk=qk, decay=decay, raw=final['val/sql_loss'],
                   ema=final['val/ema_sql_loss'], english=final['val/english_loss'],
                   ema_english=final['val/ema_english_loss'], tokens=final['tokens_seen'],
                   peak_gib=final['perf/peak_reserved_gib'], checkpoint=result['checkpoint'])
        rows.append(row)
        (directory / 'comparison.json').write_text(json.dumps(rows, indent=2))
        volume.commit()
        return row

    if smoke:
        case(True, .01)
        return rows
    case(False, 0.)
    case(True, 0.)
    selected = min(rows, key=lambda row: row['raw'])
    view = 'raw'  # EMA is observational throughout this small-model screen.
    for decay in (.01, .1):
        case(selected['qk'], decay)
    best = min((row for row in rows if row['qk'] == selected['qk']), key=lambda row: row[view])
    decision = dict(evaluation=view, winner=best, rows=rows)
    (directory / 'decision.json').write_text(json.dumps(decision, indent=2))
    volume.commit()
    with wandb.init(entity=entity, project=PROJECT, group=suite, name=f'{suite}-comparison',
                    job_type='comparison') as run:
        run.log({'final_sql': wandb.plot.line_series(
            xs=[row['decay'] for row in rows if row['qk'] == selected['qk']],
            ys=[[row[key] for row in rows if row['qk'] == selected['qk']] for key in ('raw','ema')],
            keys=['raw','ema'], title='Final SQL loss at selected QK setting', xname='Weight decay')})
    return decision


@app.local_entrypoint()
def main(entity: str, smoke: bool = False, launch: bool = False):
    """Submit one bounded verification or a sequential four-run comparison."""
    if smoke == launch:
        raise ValueError('Choose exactly one of --smoke and --launch')
    suite = ('smoke-' if smoke else 'sweep-') + uuid.uuid4().hex[:12]
    print(suite, suite_run.spawn(entity, suite, smoke).object_id)
