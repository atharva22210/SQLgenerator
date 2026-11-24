"""Measured fluctuation scores and paired-branch checks; no clipping-frequency metric."""
import math
import numpy as np
from .numerics import detrended_score


def fluctuation_scores(results: list[dict]) -> list[dict]:
    """Detrend full post-warmup and late windows on fixed validation coordinates.

    Quadratic detrending is predeclared; linear results expose sensitivity to
    that choice. Only 15 post-warmup points and 9 late points exist, so neither
    residual SD is an uncertainty/significance or a pure oscillation estimate.
    """
    records = []
    for result in results:
        if not result['completed']:
            raise ValueError('Need completed source runs')
        config = result['config']
        batch = config['micro_batch_size'] * config['seq_len'] * config['grad_accum_steps']
        budget = config['max_steps'] * batch
        for metric in ('val/sql_loss', 'val/english_loss'):
            for window, start in (('post_warmup', .1), ('late_half', .5)):
                rows = [row for row in result['history'] if metric in row and row['tokens_seen'] > start * budget]
                tokens = [row['tokens_seen'] for row in rows]
                losses = [row[metric] for row in rows]
                quadratic = detrended_score(tokens, losses, 2)
                linear = detrended_score(tokens, losses, 1)
                records.append({'run': result['run'], 'batch_tokens': batch, 'peak_lr': config['peak_lr'],
                                'metric': metric, 'window': window, 'final_loss': result['history'][-1][metric],
                                'tokens': tokens, 'losses': losses, 'quadratic': quadratic, 'linear': linear})
    return records


def branch_summary(results: list[dict]) -> dict:
    """Validate a common origin and hash-identical future examples before scoring.

    Evaluate only the common recorded probe prefix if one branch diverges. Raw
    losses and detrended SD are both preserved; fewer fluctuations alone do not
    imply better quality, and differences of two arms do not estimate seed noise.
    """
    if len(results) != 2:
        raise ValueError('Expected two branch arms')
    first, second = results
    origins = [{k:v for k,v in r['branch_origin'].items() if k != 'initial_probe_loss'} for r in results]
    if origins[0] != origins[1] or first['dataset_identity'] != second['dataset_identity']:
        raise ValueError('Branches do not share the same saved origin/data')
    a = first['config']; b = second['config']
    if {k for k in a if a[k] != b[k]} != {'peak_lr', 'min_lr'}:
        raise ValueError('Only constant LR may differ between branch arms')
    common = min(len(r['history']) for r in results)
    for r in results:
        if r['status'] not in {'completed','diverged'}:
            raise ValueError('Unexpected branch status')
        if r['status']=='completed' and len(r['history']) != r['config']['max_steps']:
            raise ValueError('Completed branch lacks its planned updates')
        if len(r['sample_hashes']) != len(r['history']):
            raise ValueError('Missing actual future-input fingerprints')
        if [row['optimizer_update'] for row in r['history']] != list(range(1,len(r['history'])+1)):
            raise ValueError('Branch histories must be contiguous')
        if any(not math.isfinite(v) for row in r['history'] for v in row.values()):
            raise ValueError('Recorded branch history is nonfinite')
    if first['sample_hashes'][:common] != second['sample_hashes'][:common]:
        raise ValueError('Branches consumed different future training inputs')
    if not math.isclose(first['initial_probe_loss'],second['initial_probe_loss'],abs_tol=.005,rel_tol=0):
        raise ValueError('Initial branch probe losses disagree')
    common_updates = sorted(set(row['optimizer_update'] for row in first['history'][:common] if 'probe/fixed_loss' in row)
                            & set(row['optimizer_update'] for row in second['history'][:common] if 'probe/fixed_loss' in row))
    records = []
    for r in results:
        by_update = {row['optimizer_update']: row for row in r['history']}
        losses = [by_update[u]['probe/fixed_loss'] for u in common_updates]
        changes = np.diff(losses)
        score = detrended_score(common_updates,losses,2) if len(losses)>3 else None
        records.append({'run': r['run'], 'lr': r['config']['peak_lr'], 'status': r['status'],
                        'successful_updates': len(r['history']), 'initial_probe_loss':r['initial_probe_loss'],
                        'common_final_probe_loss':losses[-1] if losses else None,
                        'probe_rmssd':float(np.sqrt(np.mean(changes**2))) if len(changes) else None,
                        'detrended_probe_sample_std':score['residual_sample_std'] if score else None,
                        'final_sql_loss':r['history'][-1].get('val/sql_loss') if r['history'] else None,
                        'final_english_loss':r['history'][-1].get('val/english_loss') if r['history'] else None,
                        'wandb_url':r.get('wandb_url')})
    return {'common_origin':origins[0],'matched_successful_updates':common,
            'actual_future_inputs_match':common>0,'records':records,
            'scope':'LR changes both gradient-step size and decoupled shrinkage; no EoS proof.'}


def publish(wandb, run, scores, diagnostics, branches):
    """Publish raw numeric tables and plot data, omitting clipping frequency."""
    plots = {}
    refined = any('lanczos' in d['hessian'] for d in diagnostics)
    plots['fluctuations/scores'] = wandb.Table(
        columns=['run','metric','window','final_loss','quadratic_residual_sd','linear_residual_sd','observations'],
        data=[[r['run'],r['metric'],r['window'],r['final_loss'],r['quadratic']['residual_sample_std'],
               r['linear']['residual_sample_std'],r['quadratic']['observations']] for r in scores])
    for metric in ('val/sql_loss','val/english_loss'):
        selected = [r for r in scores if r['metric']==metric and r['window']=='post_warmup']
        plots[f'fluctuations/{metric.replace("/","_")}'] = wandb.plot.line_series(
            xs=[r['tokens'] for r in selected],ys=[r['quadratic']['residuals'] for r in selected],
            keys=[r['run'] for r in selected],title=f'Quadratically detrended {metric}',xname='Training input tokens')
    plots['landscape/slices'] = wandb.plot.line_series(
        xs=[[p['alpha'] for p in d['slice']['points']] for d in diagnostics],
        ys=[[p['loss']-next(p['loss'] for p in d['slice']['points'] if p['alpha']==0) for p in d['slice']['points']] for d in diagnostics],
        keys=[d['label'] for d in diagnostics],title='Directional loss slices (baseline aligned)',xname='Normalized-update perturbation distance')
    if branches:
        for metric in ('probe/fixed_loss','train/loss','val/sql_loss','val/english_loss'):
            rows = [[row for row in r['history'] if metric in row] for r in branches]
            plots[f'branches/{metric.replace("/","_")}'] = wandb.plot.line_series(
                xs=[[row['optimizer_update'] for row in h] for h in rows],ys=[[row[metric] for row in h] for h in rows],
                keys=[r['run'] for r in branches],title=f'Constant-LR branches: {metric}',xname='Updates after shared checkpoint')
    estimate_key = 'hessian/refined_estimates' if refined else 'hessian/estimates'
    plots[estimate_key] = wandb.Table(columns=['checkpoint','actual_lr','signed_dominant_eigenvalue',
                                                    'positive_curvature_estimate','positive_relative_residual','lr_times_estimate'],
        data=[[d['label'],d['actual_lr'],d['hessian']['dominant_eigenvalue_estimate'],
               d['hessian'].get('lambda_max_estimate',d['hessian']['dominant_eigenvalue_estimate']),
               d['hessian'].get('positive_estimate_relative_residual',d['hessian']['relative_residual']),d['lr_times_curvature']] for d in diagnostics])
    plots['landscape/directional_curvature'] = wandb.Table(columns=['checkpoint','epsilon','directional_curvature'],
        data=[[d['label'],c['epsilon'],c['kappa']] for d in diagnostics for c in d['slice']['curvatures']])
    groups = [sorted([d for d in diagnostics if d['peak_lr']==lr and d['lr_times_curvature'] is not None],
                     key=lambda d:d['tokens_seen']) for lr in sorted({d['peak_lr'] for d in diagnostics})]
    groups = [g for g in groups if g]
    if groups:
        coordinates = [d['tokens_seen'] for g in groups for d in g]
        reference_key = 'hessian/refined_lr_times_curvature' if refined else 'hessian/lr_times_curvature'
        plots[reference_key] = wandb.plot.line_series(
            xs=[[d['tokens_seen'] for d in g] for g in groups]+[[min(coordinates),max(coordinates)]],
            ys=[[d['lr_times_curvature'] for d in g] for g in groups]+[[2.,2.]],
            keys=[f'Original peak LR {g[0]["peak_lr"]:g}' for g in groups]+['Unclipped quadratic GD reference'],
            title='Provisional fixed-probe eta × curvature (clipping retained)',xname='Source training input tokens')
    run.log(plots)
