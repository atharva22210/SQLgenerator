"""Validate recorded results; finite fluctuations are not automatically called EoS."""
import math
from statistics import mean
from .plan import EVAL_TOKENS, TOKEN_BUDGET, experiment_plan


def analyze(results: list[dict]) -> dict:
    """Require exact six-case coverage before comparing equal-budget final loss.

    Report fixed-probe upward steps as a descriptive count only. Without Hessian
    or controlled LR branches, that count does not diagnose an edge of stability.
    Diverged cases retain their shorter history and never enter final-loss ranks.
    """
    cases = {c['label']: c for c in experiment_plan()}
    if len(results) != len(cases) or {r['run'] for r in results} != set(cases):
        raise ValueError('Expected exactly the six planned SGD cases')
    identity = results[0]['dataset_identity']
    records = []
    for r in results:
        case = cases[r['run']]
        if r['config'] != case['train'] or r['dataset_identity'] != identity:
            raise ValueError('Run settings or data differ from the pinned plan')
        h = r['history']
        if any(not math.isfinite(v) for row in h for v in row.values()):
            raise ValueError('Committed history contains nonfinite metrics')
        if [row['optimizer_update'] for row in h] != list(range(1, len(h) + 1)):
            raise ValueError('Committed updates must be contiguous')
        if r['status'] == 'completed':
            if (len(h) != case['train']['max_steps'] or h[-1]['tokens_seen'] != TOKEN_BUDGET
                    or not r['completed'] or any(row['tokens_seen'] != (i+1)*case['batch_tokens'] for i,row in enumerate(h))):
                raise ValueError('Completed case did not reach its exact budget')
            expected = list(range(EVAL_TOKENS, TOKEN_BUDGET, EVAL_TOKENS)) + [TOKEN_BUDGET]
            for metric in ('val/sql_loss', 'val/english_loss'):
                if [row['tokens_seen'] for row in h if metric in row] != expected:
                    raise ValueError('Validation coordinates differ')
            if any('probe/fixed_loss' not in row or 'train/clipped' not in row for row in h):
                raise ValueError('Missing per-update stability diagnostics')
        elif r['status'] != 'diverged':
            raise ValueError('Unexpected status; do not hide infrastructure failures')
        probe = [row['probe/fixed_loss'] for row in h if 'probe/fixed_loss' in row]
        records.append({
            'run': r['run'], 'batch_tokens': case['batch_tokens'], 'peak_lr': case['train']['peak_lr'],
            'status': r['status'], 'updates': len(h), 'tokens_seen': h[-1]['tokens_seen'] if h else 0,
            'sql_loss': h[-1].get('val/sql_loss') if r['status'] == 'completed' else None,
            'english_loss': h[-1].get('val/english_loss') if r['status'] == 'completed' else None,
            'clipped_fraction': mean(row['train/clipped'] for row in h) if h else None,
            'probe_upward_steps': sum(b > a for a,b in zip(probe,probe[1:])),
            'probe_observations': len(probe), 'wandb_url': r.get('wandb_url'),
            'failed_update': r.get('failed_update')})
    return {'dataset_identity': identity, 'token_budget': TOKEN_BUDGET, 'records': records,
            'interpretation': 'Fixed-probe fluctuations and divergence are observations, not proof of edge of stability.'}


def report_markdown(summary: dict) -> str:
    """Write measured rows beside this experiment; keep explanations provisional."""
    lines = ['# SGD stability: measured screen', '', '## Question', '',
             'Do smaller batches and a wider SGD LR bracket reveal stable versus unstable trajectories?', '',
             '## Prediction', '', 'Higher LR may produce persistent fixed-probe fluctuations or divergence; neither is guaranteed.', '',
             '## Controlled settings', '',
             '32.768M tokens, context 4096, plain SGD with decoupled weight decay 0.1, gradient clipping 1.0, '
             'cosine/10% warmup, seed 1. See plan.json for data identity and settings.', '',
             '## Measured results', '', '| Run | Status | SQL loss | English loss | Clipped fraction | Probe upward steps |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summary['records']:
        def fmt(value): return '—' if value is None else f'{value:.6f}'
        lines.append(f"| {row['run']} | {row['status']} | {fmt(row['sql_loss'])} | {fmt(row['english_loss'])} | {fmt(row['clipped_fraction'])} | {row['probe_upward_steps']} |")
    lines += ['', '## Explanation', '', summary['interpretation'], '',
              '## Limitations', '', 'One seed, changing cosine LR, small fixed probe and clipping. No Hessian spectrum '
              'or LR-reduction branch has been measured. SGD LRs differ from the AdamW grid; compare best tested recipes, '
              'not equal numeric LR. Only batch 32768 overlaps the earlier AdamW screen.', '',
              '## Next experiment', '', 'Inspect sustained fixed-probe patterns before selecting a saved state for matched-future-batch '
              'LR branches. If every case is stable or every case diverges, the bracket needs revision; no automatic extension is authorized.']
    lines += ['', '## Sources', ''] + [f"- [{r['run']}]({r['wandb_url']})" for r in summary['records'] if r['wandb_url']]
    return '\n'.join(lines) + '\n'


def publish(wandb, run, results: list[dict]) -> dict:
    """Overlay complete finite histories with explicit token coordinates."""
    from training.tracking import curve_payload
    summary = analyze(results)
    completed = [r for r in results if r['status'] == 'completed']
    if completed:
        run.log(curve_payload(wandb, [r['history'] for r in completed],
                              [r['run'] for r in completed], x_key='tokens_seen'))
    run.log({'comparison/results': wandb.Table(columns=['run','status','sql_loss','english_loss','clipped_fraction'],
             data=[[row[k] for k in ('run','status','sql_loss','english_loss','clipped_fraction')] for row in summary['records']])})
    return summary
