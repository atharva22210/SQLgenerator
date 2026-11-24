"""Republish corrected numerical evidence without repeating GPU work or training."""
import json
from pathlib import Path
import modal
from training.config import image,volume,wandb_secret
from .launch import ROOT,PROJECT

app=modal.App('sqlgen-landscape-finalize')


@app.function(image=image,cpu=2,memory=4096,timeout=300,volumes={'/vol':volume},secrets=[wandb_secret])
def finalize(suite: str, entity: str, republish: bool = False) -> dict:
    """Validate paired inputs, update the SAME comparison run and export artifacts."""
    import time
    import wandb
    from .analysis import branch_summary,publish
    from training.tracking import verify_uploaded_result
    volume.reload();root=ROOT/suite
    def read(name):return json.loads((root/name).read_text())
    plan=read('plan.json');summary=read('summary.json');scores=read('fluctuations.json')
    refined=root/'diagnostics-refined.json'
    if not refined.exists():
        raise ValueError('Wait for the bounded signed-eigenvalue refinement to finish')
    diagnostics=read('diagnostics-refined.json')
    before=root/'diagnostics-before-refinement.json'
    if not before.exists():before.write_text((root/'diagnostics.json').read_text())
    (root/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    branches=[read(case['label']+'/result.json')for case in plan['branches']]
    measured=branch_summary(branches)
    tracking={}
    for result in branches:
        if result['status']=='completed':tracking[result['run']]=verify_uploaded_result(result)
    comparison_id=summary['comparison_url'].rstrip('/').split('/')[-1]
    if republish:
        with wandb.init(entity=entity,project=PROJECT,id=comparison_id,resume='must',group=suite,
                        job_type='comparison',name='SGD loss landscape and constant-LR branches')as run:
            publish(wandb,run,scores,diagnostics,branches)
            run.summary['hessian_refinement']='12-step Lanczos plus residual HVP on negative-dominant power case; original signed estimate retained.'
    # Fresh metric names for corrected Hessian data avoid stale immutable-table
    # metadata after a Forge/W&B run resume. Preserve the original power-only
    # charts as historical evidence; identify the corrected charts explicitly.
    expected={'landscape/slices_table':36,'landscape/directional_curvature':8,'hessian/refined_estimates':4,
              'fluctuations/scores':24,'branches/probe_fixed_loss_table':202,'branches/train_loss_table':400,
              'branches/val_sql_loss_table':8,'branches/val_english_loss_table':8,
              'hessian/refined_lr_times_curvature_table':sum(d['lr_times_curvature']is not None for d in diagnostics)+2}
    delays=(0,2,4,8,16,30)
    for attempt,delay in enumerate(delays):
        if delay:time.sleep(delay)
        remote=wandb.Api(timeout=30).run(f'{entity}/{PROJECT}/{comparison_id}')
        mismatch={k:v for k,v in expected.items() if remote.summary.get(k,{}).get('nrows')!=v}
        if not mismatch:break
        if attempt==len(delays)-1:
            actual={k:remote.summary.get(k,{}).get('nrows')for k in mismatch}
            raise RuntimeError(f'Comparison chart uploads incomplete: expected {mismatch}; actual {actual}')
    summary.update(branches=measured,diagnostics=diagnostics,comparison_tables_verified=expected)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'branch_summary.json').write_text(json.dumps(measured,indent=2))
    volume.commit()
    return {'plan':plan,'summary':summary,'fluctuations':scores,'diagnostics':diagnostics,'branches':branches,
            'tracking_checks':tracking,'comparison_state':remote.state}


@app.local_entrypoint()
def main(suite: str, wandb_entity: str, republish: bool = False):
    result=finalize.remote(suite,wandb_entity,republish)
    Path('/tmp/sqlgen-landscape-final-results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'comparison_url':result['summary']['comparison_url'],
                      'paired_inputs_verified':result['summary']['branches']['actual_future_inputs_match'],
                      'comparison_tables_verified':result['summary']['comparison_tables_verified']},indent=2))
