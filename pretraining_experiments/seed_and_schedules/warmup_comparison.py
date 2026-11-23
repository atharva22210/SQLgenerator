"""A controlled 100-versus-300-update warmup follow-up; no training on import."""
from dataclasses import asdict
import json
import math
import statistics
from pathlib import Path

from training.config import TrainConfig
from training.tracking import curve_payload

BASELINE_SUITE = 'day07-full-2a6757d322c4'


def experiment_plan() -> list[dict]:
    """Prepare only two fresh seed-1 cases; completed 100-update runs are reused.

    At 1000 updates, WSD must use stable_frac=.5 with warmup=300 and decay=.2.
    Keeping .7 would create 1200 updates and violate the fixed training budget.
    Longer warmup tests stability; it does not guarantee safety or a higher loss.
    """
    base = TrainConfig(mode='masked_pack', seq_len=4096, micro_batch_size=8,
                       grad_accum_steps=4, max_steps=1000, warmup_steps=300,
                       eval_every=50, val_batches=128, seed=1, validation_seed=12345,
                       wsd_stable_frac=.5, wsd_decay_frac=.2)
    return [{'label': f'{name}-warmup300-seed-1', 'baseline_label': f'{name}-seed-1',
             'train': {**asdict(base), 'lr_schedule': schedule}}
            for name, schedule in (('cosine', 'cosine'), ('wsd-linear', 'wsd'))]


def load_baselines(directory: Path, cases: list[dict], dataset: dict) -> list[dict]:
    """Reject incompatible archived settings/data BEFORE spending GPU compute."""
    results = []
    for case in cases:
        result = json.loads((directory / case['baseline_label'] / 'result.json').read_text())
        expected = {**case['train'], 'warmup_steps': 100, 'wsd_stable_frac': .7}
        if result['config'] != expected or result['dataset_identity'] != dataset or not result['completed']:
            raise ValueError('Warmup comparison requires the original completed settings and dataset')
        results.append(result)
    return results


def stable_region_gaps(cosine: dict, wsd: dict, warmup: int) -> dict:
    """Measure WSD-minus-cosine during stable updates only, excluding the tail.

    Validation uses matched fixed documents; training gaps use changing sampled
    batches and must be interpreted with more caution. Serial points within one
    seed are not independent replications. Preserve the warmup-end gap as a
    reference for drift that existed before the schedules separated.
    """
    left = {row['optimizer_update']: row for row in cosine['history']}
    right = {row['optimizer_update']: row for row in wsd['history']}
    metrics = {}
    for metric in ('train/loss', 'val/sql_loss', 'val/english_loss'):
        points = [{'update': update, 'gap': right[update][metric] - left[update][metric]}
                  for update in range(warmup + 1, 801)
                  if metric in left[update] and metric in right[update]]
        values = [point['gap'] for point in points]
        metrics[metric] = {'points': points, 'mean_gap': statistics.mean(values),
                           'above': sum(value > 0 for value in values),
                           'below': sum(value < 0 for value in values),
                           'equal': sum(value == 0 for value in values),
                           'observations': len(values),
                           'warmup_end_gap': right[warmup][metric] - left[warmup][metric]}
    return metrics


def publish_comparison(wandb, run, cases: list[dict], baselines: list[dict], results: list[dict]) -> dict:
    """Compare complete matched-budget runs; do not assert a preferred curve shape."""
    if cases != experiment_plan() or len(baselines) != 2 or len(results) != 2:
        raise ValueError('Expected both 100-update baselines and both 300-update cases')
    records = []
    for case, baseline, result in zip(cases, baselines, results, strict=True):
        expected_base = {**case['train'], 'warmup_steps': 100, 'wsd_stable_frac': .7}
        for actual, config, label in ((baseline, expected_base, case['baseline_label']),
                                      (result, case['train'], case['label'])):
            history = actual['history']
            if (actual['run'] != label or actual['config'] != config or not actual['completed']
                    or actual['dataset_identity'] != baselines[0]['dataset_identity']
                    or [r['optimizer_update'] for r in history] != list(range(1,1001))
                    or history[-1]['perf/cumulative_non_pad_tokens'] != 131072000):
                raise ValueError('Settings, data, complete history and token budgets must match')
            for metric in ('val/sql_loss', 'val/english_loss'):
                if [r['optimizer_update'] for r in history if metric in r] != list(range(50,1001,50)):
                    raise ValueError('Validation coordinates differ')
            if any(not math.isfinite(v) for row in history for v in row.values()):
                raise ValueError('Nonfinite measured history')
            records.append({'run': label, 'warmup_steps': config['warmup_steps'],
                            'sql_loss': history[-1]['val/sql_loss'],
                            'english_loss': history[-1]['val/english_loss'],
                            'clipped_fraction': sum(r['train/grad_norm'] > 1 for r in history)/1000,
                            'wandb_url': actual['wandb_url']})
    ordered = [r for pair in zip(baselines,results) for r in pair]
    plots = curve_payload(wandb, [r['history'] for r in ordered], [r['run'] for r in ordered])
    plots['charts/gradient_norm'] = wandb.plot.line_series(
        xs=[[r['optimizer_update'] for r in result['history']] for result in ordered],
        ys=[[r['train/grad_norm'] for r in result['history']] for result in ordered],
        keys=[r['run'] for r in ordered], title='Pre-clipping gradient norm', xname='Optimizer update')
    stable_gaps = {'100': stable_region_gaps(baselines[0], baselines[1], 100),
                   '300': stable_region_gaps(results[0], results[1], 300)}
    for metric in ('train/loss', 'val/sql_loss', 'val/english_loss'):
        series = [stable_gaps[warmup][metric]['points'] for warmup in ('100', '300')]
        plots[f'charts/stable_gap_{metric.replace("/", "_")}'] = wandb.plot.line_series(
            xs=[[p['update'] for p in points] for points in series],
            ys=[[p['gap'] for p in points] for points in series],
            keys=['Warmup 100', 'Warmup 300'],
            title=f'{metric}: WSD minus cosine (positive = WSD higher)', xname='Optimizer update')
    run.log(plots)
    return {'records': records, 'stable_region_gaps': stable_gaps, 'baseline_suite': BASELINE_SUITE,
            'dataset_identity': baselines[0]['dataset_identity']}


def report_markdown(analysis: dict) -> str:
    """Write measured findings without inventing replication or significance."""
    lines = ['# Warmup duration comparison', '', '## Question and prediction', '',
             'Is WSD loss above cosine during its stable phase with 300-update warmup, '
             'and does that differ from the completed 100-update comparison? Either outcome is valid.', '',
             '## Controlled settings', '', '1000 updates, 131.072M input tokens, 4096 context, batch 8 × 4, '
             'same data, seed 1 and validation seed 12345. Only warmup and its required stable-phase '
             'duration change. WSD decay stays at updates 801–1000.', '', '## Measured results', '',
             '| Run | Warmup | SQL loss | English loss | Clipped update fraction |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in analysis['records']:
        lines.append(f'| {r["run"]} | {r["warmup_steps"]} | {r["sql_loss"]:.6f} | '
                     f'{r["english_loss"]:.6f} | {r["clipped_fraction"]:.3f} |')
    lines += ['', '## Stable-region ordering', '',
              'Gap = WSD loss minus cosine loss. Positive means WSD is higher. '
              'Use updates 301–800 for warmup 300, and 101–800 for the archived warmup 100. '
              'The warmup endpoint and decay region are excluded from the mean.', '',
              '| Warmup | Metric | Mean gap | Above / observations | Below | Equal | Gap at warmup end |',
              '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for warmup, metrics in analysis['stable_region_gaps'].items():
        for metric, values in metrics.items():
            lines.append(f'| {warmup} | {metric} | {values["mean_gap"]:+.6f} | '
                         f'{values["above"]}/{values["observations"]} | {values["below"]} | '
                         f'{values["equal"]} | {values["warmup_end_gap"]:+.6f} |')
    lines += ['', 'The new validation comparison has ten observations per corpus, at updates '
              '350,400,…,800. A positive mean need not mean WSD is higher everywhere; '
              'read the pointwise gaps and above/below counts too. These counts are '
              'descriptive, not ten independent seeds or significance tests. A gap '
              'already present at update 300 cannot be attributed to a schedule change '
              'that begins afterward. Training gaps remain sensitive to batch composition.']
    lines += ['', '## Explanation', '', 'Compare each schedule with its own shorter-warmup baseline. '
              'Inspect early gradient spikes alongside final fixed-sample loss. Lower clipping alone '
              'does not establish better learning; loss need not follow the LR shape.', '',
              '## Limitations', '', 'One seed per new configuration; same-seed GPU drift was observed '
              'in the original suite. No automatic significance claim or guaranteed benefit from longer '
              'warmup. The archived baseline includes checkpoint replay, so wall-clock cost is not a '
              'controlled outcome. Validation loss is not SQL execution accuracy.', '',
              '## Next experiment', '', 'If the new setting is promising, replicate with matched seeds '
              'before adopting it. Do not retune simply to force the predicted curve shape.', '', '## Evidence', '']
    lines += [f'- [{r["run"]}]({r["wandb_url"]})' for r in analysis['records']]
    return '\n'.join(lines)+'\n'
