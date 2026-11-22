"""Validate seed and schedule results and render a measured report."""
import argparse
import json
import math
from pathlib import Path
import statistics

from pretraining_experiments.seed_and_schedules import EXPERIMENT_NAME
from training.tracking import curve_payload


def analyze(results: list[dict]) -> dict:
    """Compare final losses at a matched budget, retaining all individual seeds.

    Input: five result dictionaries from the shared trainer (config/history/data
    identity included). Output: JSON-compatible sample statistics and paired
    seed-1 schedule differences. Positive improvement means lower WSD loss.
    Twice the sample deviation is a screening heuristic, not a significance test.
    """
    if len(results) != 5 or any(not result['completed'] for result in results):
        raise ValueError('Analysis requires five completed runs')
    reference = results[0]
    invariant = {key: value for key, value in reference['config'].items()
                 if key not in {'seed', 'lr_schedule', 'wsd_decay_type'}}
    if invariant['validation_seed'] is None:
        raise ValueError('Seed comparison needs a shared validation_seed')
    by_case = {}
    for result in results:
        config, history = result['config'], result['history']
        actual = {key: value for key, value in config.items()
                  if key not in {'seed', 'lr_schedule', 'wsd_decay_type'}}
        if actual != invariant or result['dataset_identity'] != reference['dataset_identity']:
            raise ValueError('Cases have mismatched settings or data')
        if [row['optimizer_update'] for row in history] != list(range(1, config['max_steps'] + 1)):
            raise ValueError('Training history is missing or duplicates updates')
        expected_evals = [step for step in range(1, config['max_steps'] + 1)
                          if step % config['eval_every'] == 0 or step == config['max_steps']]
        for metric in ('val/sql_loss', 'val/english_loss'):
            if [row['optimizer_update'] for row in history if metric in row] != expected_evals:
                raise ValueError('Cases must use the same validation updates')
            if not all(math.isfinite(row[metric]) for row in history if metric in row):
                raise ValueError('Validation contains nonfinite values')
        key = (f'cosine-seed-{config["seed"]}' if config['lr_schedule'] == 'cosine'
               else f'wsd-{config["wsd_decay_type"]}-seed-{config["seed"]}')
        if key in by_case:
            raise ValueError('Duplicate seed and schedule case')
        by_case[key] = result
    expected = {*(f'cosine-seed-{seed}' for seed in (1, 2, 3)),
                'wsd-linear-seed-1', 'wsd-cosine-seed-1'}
    if set(by_case) != expected:
        raise ValueError('Expected cosine seeds 1/2/3 and both WSD tails with seed 1')
    metrics = {}
    for metric in ('val/sql_loss', 'val/english_loss'):
        values = [by_case[f'cosine-seed-{seed}']['history'][-1][metric] for seed in (1, 2, 3)]
        std = statistics.stdev(values)
        baseline = values[0]
        metrics[metric] = {'cosine_seeds': dict(zip(('1', '2', '3'), values)),
                          'n': 3, 'mean': statistics.mean(values), 'sample_std': std,
                          'screening_threshold_2std': 2 * std,
                          'paired_seed_1_improvement': {
                              tail: baseline - by_case[f'wsd-{tail}-seed-1']['history'][-1][metric]
                              for tail in ('linear', 'cosine')}}
    return {'metrics': metrics, 'dataset_identity': reference['dataset_identity'],
            'controlled_settings': invariant,
            'runs': [{'name': key, 'wandb_url': result['wandb_url'],
                      'non_pad_tokens': result['history'][-1]['perf/cumulative_non_pad_tokens']}
                     for key, result in by_case.items()]}


def report_markdown(analysis: dict) -> str:
    """Render measured values with the question, uncertainty, and next test."""
    settings = analysis['controlled_settings']
    lines = [f'# {EXPERIMENT_NAME}', '', '## Question', '',
             'How large is seed variation, and does WSD improve loss relative to cosine at the same budget?', '',
             '## Prediction', '',
             'WSD may remain higher during the stable phase and fall during decay. '
             'This is a hypothesis, not a promised outcome.', '', '## Controlled settings', '',
             f'Context {settings["seq_len"]}; masked packing; batch {settings["micro_batch_size"]} × '
             f'{settings["grad_accum_steps"]} accumulation; {settings["max_steps"]} updates; '
             f'validation seed {settings["validation_seed"]}; {settings["val_batches"]} documents per corpus.',
             f'Published dataset manifest SHA256: `{analysis["dataset_identity"]["manifest_sha256"]}`.', '',
             '## Measured results', '', '| Metric | Seed 1 | Seed 2 | Seed 3 | Mean | Sample std | 2 × std |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, stats in analysis['metrics'].items():
        values = [*stats['cosine_seeds'].values(), stats['mean'], stats['sample_std'], stats['screening_threshold_2std']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines += ['', '## Explanation', '',
              'Positive paired improvement below means WSD had lower final loss than cosine with the same seed. '
              'A sharp fall aligned with the decay phase would be consistent with a schedule effect; '
              'the curves are needed to assess that timing. Final loss alone does not establish its cause.', '']
    for name, stats in analysis['metrics'].items():
        for tail, delta in stats['paired_seed_1_improvement'].items():
            lines.append(f'- {name}, {tail} WSD: paired improvement {delta:+.6f}; '
                         f'larger than 2 × seed std: {delta > stats["screening_threshold_2std"]}.')
    lines += ['', '## Limitations', '',
              'Only three cosine seeds and one seed per WSD tail were measured. The 2 × std threshold is '
              'a screening heuristic, not statistical significance. Validation is a fixed document sample, '
              'not the entire held-out corpus or SQL execution accuracy. Compare the recorded token counts '
              'as well as optimizer updates. These seeds mix initialization and data-order variation.', '',
              '## Next experiment', '',
              'Repeat both WSD tails with seeds 2 and 3, then compare paired differences across seeds. '
              'Keep validation documents fixed; separately test larger validation samples if the ranking is close.', '',
              '## Evidence', '']
    lines += [f'- [{run["name"]}]({run["wandb_url"]}); non-padding tokens: {run["non_pad_tokens"]:,}.'
              for run in analysis['runs']]
    return '\n'.join(lines) + '\n'


def publish_comparison(wandb, run, results: list[dict]) -> dict:
    """Publish explicit overlays and a numeric table, using saved full histories."""
    analysis = analyze(results)
    plots = curve_payload(wandb, [result['history'] for result in results], [result['run'] for result in results])
    rows = [[result['run'], result['history'][-1]['val/sql_loss'],
             result['history'][-1]['val/english_loss']] for result in results]
    plots['comparison/results'] = wandb.Table(columns=['case', 'sql_loss', 'english_loss'], data=rows)
    run.log(plots)
    for metric, stats in analysis['metrics'].items():
        name = metric.removeprefix('val/')
        run.summary.update({f'seed_noise/{name}_mean': stats['mean'],
                            f'seed_noise/{name}_sample_std': stats['sample_std']})
    return analysis


def main() -> None:
    """Analyze downloaded result.json files; does not train or contact W&B."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs=5, help='Five result.json files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze([json.loads(path.read_text()) for path in args.results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_markdown(analysis))
    args.output.with_suffix('.json').write_text(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
