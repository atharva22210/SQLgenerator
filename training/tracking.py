"""Scalar histories and explicit charts with a single optimizer-update axis."""
import json
import math
import time

CURVES = {"train/lr": "Learning rate", "train/loss": "Training loss",
          "val/sql_loss": "SQL validation loss", "val/english_loss": "English validation loss"}


def define_metrics(run, step_metric: str = 'optimizer_update') -> None:
    """Define native scalar axes; charts/tables never masquerade as scalars."""
    run.define_metric('optimizer_update', hidden=True)
    run.define_metric('completed_updates', hidden=True)
    if step_metric != 'optimizer_update':
        run.define_metric(step_metric, hidden=True)
    for namespace in ('train', 'val', 'data', 'perf', 'probe'):
        run.define_metric(f'{namespace}/*', step_metric=step_metric)


def log_update(run, update: int, metrics: dict, history: list[dict], output) -> None:
    """Write ONE complete numeric row per update to W&B and the local JSONL.

    Use an explicit x-axis column rather than W&B's internal step argument:
    internal steps also advance when chart objects are logged. Sparse validation
    stays sparse; absent evaluation values are never replaced with zeroes.
    """
    if history and update <= history[-1]['optimizer_update']:
        raise ValueError('Optimizer updates must increase within a training attempt')
    if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in metrics.values()):
        raise ValueError('Training metrics must be finite numeric scalars')
    row = {'optimizer_update': update, **metrics}
    run.log(row)
    history.append(row)
    output.write(json.dumps(row) + '\n')


def curve_payload(wandb, histories: list[list[dict]], names: list[str], *,
                  x_key: str = 'optimizer_update') -> dict:
    """Build each chart from real (update,value) pairs, one series per run."""
    plots = {}
    for metric, title in CURVES.items():
        series = [[row for row in rows if metric in row] for rows in histories]
        if any(not rows for rows in series):
            raise ValueError(f'Missing history for {metric}')
        plots[f'charts/{metric.replace("/", "_")}'] = wandb.plot.line_series(
            xs=[[row[x_key] for row in rows] for rows in series],
            ys=[[row[metric] for row in rows] for rows in series],
            keys=names, title=title, xname='Optimizer update' if x_key == 'optimizer_update' else 'Non-padding training tokens')
    if all(any('probe/fixed_loss' in row for row in rows) for rows in histories):
        series = [[row for row in rows if 'probe/fixed_loss' in row] for rows in histories]
        plots['charts/fixed_probe_loss'] = wandb.plot.line_series(
            xs=[[row[x_key] for row in rows] for rows in series],
            ys=[[row['probe/fixed_loss'] for row in rows] for rows in series],
            keys=names, title='Fixed training-batch loss (post-update)',
            xname='Optimizer update' if x_key == 'optimizer_update' else 'Non-padding training tokens')
    return plots


def verify_history(api, run_path: str, expected_updates: list[int], expected_evaluations: list[int]) -> dict:
    """Check uploaded scalar rows AND chart tables, using the public W&B API.

    No sampled run.history(): scan_history returns all records. This
    checks data and chart specifications; it cannot assert a browser rendered UI.
    """
    run = api.run(run_path)
    # The first query after SDK finalization can precede server indexing. A
    # cached partial history would make every retry repeat the same false error.
    rows = list(run.scan_history(use_cache=False))
    counts = {}
    for metric in CURVES:
        # The API expands absent columns to None, including chart-only rows.
        # Retain only actual scalar observations; missing expected updates still
        # fail the exact coordinate check below.
        matching = [row for row in rows if row.get(metric) is not None
                    and row.get('optimizer_update') is not None]
        actual = sorted({int(row['optimizer_update']) for row in matching})
        expected = expected_evaluations if metric.startswith('val/') else expected_updates
        if actual != expected or any(not math.isfinite(row[metric]) for row in matching):
            raise RuntimeError(f'Uploaded history incomplete for {metric}: {actual}')
        counts[f'charts/{metric.replace("/", "_")}_table'] = len(expected)
    for key, expected_count in counts.items():
        table = run.summary.get(key)
        if table is None or table.get('nrows') != expected_count or table.get('ncols') != 3:
            raise RuntimeError(f'Uploaded chart table missing or incomplete: {key}')
    every = run.config.get('probe_recipe', {}).get('every', 0)
    if every:
        expected = [u for u in expected_updates if u == 1 or u % every == 0 or u == expected_updates[-1]]
        matching = [row for row in rows if row.get('probe/fixed_loss') is not None]
        if (sorted({int(row['optimizer_update']) for row in matching}) != expected
                or any(not math.isfinite(row['probe/fixed_loss']) for row in matching)):
            raise RuntimeError('Uploaded fixed-probe history incomplete')
        table = run.summary.get('charts/fixed_probe_loss_table')
        if table is None or table.get('nrows') != len(expected):
            raise RuntimeError('Uploaded fixed-probe chart table incomplete')
    return {'scalar_history_verified': True, 'chart_tables_verified': True, 'wandb_url': run.url}


def verify_uploaded_result(result: dict) -> dict:
    """Recheck fresh server data with bounded backoff; never rerun training.

    Call from the CPU coordinator after GPU workers exit. Saved numeric history
    is the expected contract; a transient indexing delay should consume CPU
    verification time rather than another training attempt or an idle GPU.
    """
    import wandb
    history = result['history']
    updates = [row['optimizer_update'] for row in history]
    evaluations = [row['optimizer_update'] for row in history if 'val/sql_loss' in row]
    delays = (0, 2, 4, 8, 16, 30)
    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            return verify_history(wandb.Api(timeout=30), result['wandb_path'], updates, evaluations)
        except (RuntimeError, wandb.errors.Error):
            if attempt == len(delays) - 1:
                raise
    raise AssertionError('Unreachable verification retry state')
