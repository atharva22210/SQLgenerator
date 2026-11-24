"""Validate matched-budget results before ranking learning-rate choices."""
import json
import math
from pathlib import Path

from .plan import TOKEN_BUDGET, experiment_plan


def analyze(cases: list[dict], results: list[dict]) -> dict:
    """Return numeric records and the best tested SQL LR per batch, excluding failures.

    Require exact plan coverage, common data, complete finite histories and equal
    final token exposure. A diverged run is a recorded failure, not a high invented
    loss or a completed run at a smaller budget. This does not estimate seed variance.
    """
    if len(cases) != len(results) or not cases:
        raise ValueError('Expected one result per planned case')
    canonical = {case['label']: case for case in experiment_plan()}
    if len({case['label'] for case in cases}) != len(cases) or any(canonical.get(c['label']) != c for c in cases):
        raise ValueError('Cases differ from the agreed grid')
    by_name = {r['run']: r for r in results}
    if len(by_name) != len(results) or set(by_name) != {c['label'] for c in cases}:
        raise ValueError('Missing, duplicate or unexpected case results')
    dataset = results[0]['dataset_identity']
    rows = []
    for case in cases:
        result = by_name[case['label']]
        if result['dataset_identity'] != dataset or result['config'] != case['train']:
            raise ValueError('Data or settings differ from the pinned plan')
        row = {'case': case['label'], 'batch_tokens': case['batch_tokens'],
               'peak_lr': case['train']['peak_lr'], 'status': result['status'],
               'sql_loss': None, 'english_loss': None, 'wandb_url': result.get('wandb_url')}
        if result['status'] == 'completed':
            history = result['history']; config = case['train']
            if not result['completed'] or [r['optimizer_update'] for r in history] != list(range(1, config['max_steps'] + 1)):
                raise ValueError('Completed case lacks its full update history')
            expected = [step * case['batch_tokens'] for step in range(1, config['max_steps'] + 1)
                        if step % config['eval_every'] == 0 or step == config['max_steps']]
            for metric in ('val/sql_loss', 'val/english_loss'):
                evaluated = [r for r in history if metric in r]
                if [r['tokens_seen'] for r in evaluated] != expected:
                    raise ValueError('Validation token coordinates do not match')
            if any(not math.isfinite(v) for r in history for v in r.values()):
                raise ValueError('Nonfinite metrics in a completed case')
            final = history[-1]
            if final['tokens_seen'] != TOKEN_BUDGET or final['perf/cumulative_non_pad_tokens'] != TOKEN_BUDGET:
                raise ValueError('Completed cases must reach the same non-padding token budget')
            row.update(sql_loss=final['val/sql_loss'], english_loss=final['val/english_loss'])
        elif result['status'] != 'diverged':
            raise ValueError('Unknown status; infrastructure errors must not masquerade as divergence')
        rows.append(row)
    best = []
    for batch in sorted({r['batch_tokens'] for r in rows}):
        valid = [r for r in rows if r['batch_tokens'] == batch and r['status'] == 'completed']
        best.append(min(valid, key=lambda r: r['sql_loss']) if valid else {'batch_tokens': batch, 'status': 'no_completed_case'})
    return {'token_budget': TOKEN_BUDGET, 'dataset_identity': dataset, 'records': rows, 'best_tested_sql': best}


def report_markdown(summary: dict) -> str:
    """Build an honest measured table; leave untested conclusions as questions."""
    lines = ['# Batch size and learning rate: measured takeaways', '', '## Question', '',
             'At equal training input tokens, how does batch size affect the best tested LR and validation loss?', '',
             '## Prediction', '', 'Larger batches may benefit from higher LR; at some size they may lose token efficiency. '
             'Neither a shift nor an exact critical batch size is guaranteed.', '', '## Controlled settings', '',
             f'{summary["token_budget"]:,} non-padding tokens per completed case; context 4096; microbatch 4; '
             'cosine with 10% warmup; seed 1; fixed validation seed 12345 and 128 documents per corpus. '
             'The pinned dataset identity is in analysis.json.', '', '## Measured results', '',
             '| Batch tokens | Peak LR | Status | Final SQL loss | Final English loss |',
             '| ---: | ---: | --- | ---: | ---: |']
    for r in summary['records']:
        sql = '—' if r['sql_loss'] is None else f'{r["sql_loss"]:.6f}'
        english = '—' if r['english_loss'] is None else f'{r["english_loss"]:.6f}'
        lines.append(f'| {r["batch_tokens"]:,} | {r["peak_lr"]:g} | {r["status"]} | {sql} | {english} |')
    lines += ['', '## Explanation', '', 'Best tested LR by batch (selected on SQL validation):', '']
    for r in summary['best_tested_sql']:
        lines.append(f'- Batch {r["batch_tokens"]:,}: ' +
                     (f'LR {r["peak_lr"]:g}, SQL loss {r["sql_loss"]:.6f}.' if r['status'] == 'completed' else 'no completed case.'))
    lines += ['', 'Compare the best-loss curve across batches and the LR curves within each batch. '
              'An interior minimum is more informative than a winner at a grid boundary. '
              'A boundary winner may simply mean the LR search range is too narrow.', '',
              '## Limitations', '', 'One seed per case; a limited LR grid; one reused validation sample. '
              'The best tested point is not the true optimum. Any apparent critical-batch knee is a region '
              'to investigate, not an exact universal constant. Gradient accumulation changes update '
              'frequency without adding GPUs. Actual predicted-target counts differ from input tokens. '
              'If only a subset ran, missing grid points cannot support a full-grid conclusion. '
              'Loss is not SQL execution accuracy.', '', '## Next experiment', '',
              'Replicate promising comparisons across seeds and refine LR around the best tested values. '
              'If a winner is at a boundary, extend the LR range before concluding batch size is harmful.', '',
              '## Evidence', '']
    lines += [f'- [{r["case"]}]({r["wandb_url"]})' for r in summary['records'] if r['wandb_url']]
    return '\n'.join(lines) + '\n'


def publish(wandb, run, cases: list[dict], results: list[dict]) -> dict:
    """Plot LR/batch tradeoffs and sparse full validation curves on a token axis.

    Each case has <=1000 raw points. Bound overview tables so future larger grids cannot exceed
    W&B's default table size: sample up to 401 points/case for overview plots.
    All validation points and original numeric histories remain unchanged.
    """
    summary = analyze(cases, results)
    records = summary['records']
    plots = {'comparison/results': wandb.Table(columns=['case', 'batch_tokens', 'peak_lr', 'status', 'sql_loss', 'english_loss'],
              data=[[r[k] for k in ('case', 'batch_tokens', 'peak_lr', 'status', 'sql_loss', 'english_loss')] for r in records])}
    groups = [[r for r in records if r['batch_tokens'] == batch and r['status'] == 'completed']
              for batch in sorted({r['batch_tokens'] for r in records})]
    groups = [sorted(g, key=lambda r: r['peak_lr']) for g in groups if g]
    if groups:
        plots['comparison/sql_loss_vs_peak_lr'] = wandb.plot.line_series(
            xs=[[r['peak_lr'] for r in g] for g in groups], ys=[[r['sql_loss'] for r in g] for g in groups],
            keys=[f'{g[0]["batch_tokens"]:,} tokens/update' for g in groups], title='Final SQL loss versus peak LR', xname='Peak learning rate')
        best = [r for r in summary['best_tested_sql'] if r['status'] == 'completed']
        plots['comparison/best_sql_loss_vs_batch'] = wandb.plot.line_series(
            xs=[r['batch_tokens'] for r in best], ys=[[r['sql_loss'] for r in best]],
            keys=['Best tested SQL LR'], title='Best tested SQL loss at equal token budget', xname='Tokens per optimizer update')
    completed = [r for r in results if r['status'] == 'completed']
    for metric in ('train/lr', 'train/loss', 'val/sql_loss', 'val/english_loss'):
        series = []
        for result in completed:
            rows = [r for r in result['history'] if metric in r]
            stride = max(1, math.ceil(len(rows) / 400))
            selected = rows[::stride]
            if rows and selected[-1] != rows[-1]: selected.append(rows[-1])
            series.append(selected)
        if series:
            plots[f'comparison/{metric.replace("/", "_")}_vs_tokens'] = wandb.plot.line_series(
                xs=[[r['tokens_seen'] for r in rows] for rows in series],
                ys=[[r[metric] for r in rows] for rows in series], keys=[r['run'] for r in completed],
                title=f'{metric} versus training tokens (overview)', xname='Non-padding training tokens')
    run.log(plots)
    return summary


def main() -> None:
    """Recompute measured analysis from downloaded JSON files, entirely locally."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(json.loads(args.plan.read_text())['cases'],
                      [json.loads(path.read_text()) for path in args.results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_markdown(summary))
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
