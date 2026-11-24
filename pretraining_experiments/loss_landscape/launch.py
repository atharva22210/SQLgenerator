"""One bounded investigation; source files are read-only and branch outputs isolated."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import modal
from training.config import TrainConfig, image, volume, wandb_secret
from .plan import (SOURCE_ROOT, BRANCH_EVERY, BRANCH_UPDATES, HESSIAN_STEPS,
                   ALPHAS, EPSILONS, checkpoint_plan, branch_plan)

app = modal.App('sqlgen-loss-landscape')
ROOT = Path('/vol/checkpoints/pretraining_experiments/loss_landscape')
PROJECT = 'sqlgen-loss-landscape'


@app.function(image=image, gpu='A10', cpu=4, memory=8192, timeout=1800,
              max_containers=1, scaledown_window=2, volumes={'/vol':volume})
def diagnose(suite: str) -> list[dict]:
    """FP32 forward slices and 12 HVP iterations at four retained SGD states.

    No optimizer step is taken. Use the exact saved two-document objective,
    trim padding, force differentiable MATH attention, and disable TF32. Persist
    each checkpoint's measurements before proceeding; no automatic retries or
    larger curvature jobs are launched if convergence is poor.
    """
    import torch
    from model.transformer_35m import Transformer, TransformerConfig
    from .objective import ProbeObjective
    from .numerics import directional_slice, power_curvature
    volume.reload()
    torch.backends.cuda.matmul.allow_tf32 = False
    root = ROOT / suite
    root.mkdir(parents=True,exist_ok=True)
    records = []
    for case in checkpoint_plan():
        path = Path(case['checkpoint'])
        state = torch.load(path,map_location='cpu',weights_only=False)
        probe = torch.load(path.parent/'fixed_probe.pt',map_location='cpu',weights_only=False)
        if probe['dataset_identity'] != state['dataset_identity'] or not state.get('update_direction'):
            raise ValueError('Probe identity or retained update direction missing')
        model = Transformer(TransformerConfig(**state['model_config'])).cuda().float().eval()
        model.load_state_dict(state['model'])
        objective = ProbeObjective(model,probe['batch'])
        record = {**case, 'tokens_seen':state['cumulative_tokens'],
                  'actual_lr':state['history'][-1]['train/lr'], 'dataset_identity':state['dataset_identity'],
                  'probe_targets':[b.predicted_tokens for b in probe['batch']],
                  'precision':'FP32 parameters and math SDPA; TF32 disabled'}
        record['slice'] = directional_slice(model.named_parameters(),state['update_direction'],objective,ALPHAS,EPSILONS)
        record['hessian'] = power_curvature(objective,model.parameters(),steps=HESSIAN_STEPS)
        eigenvalue = record['hessian']['lambda_max_estimate']
        record['lr_times_curvature'] = record['actual_lr'] * eigenvalue if eigenvalue is not None else None
        kappas = [p['kappa'] for p in record['slice']['curvatures']]
        record['epsilon_relative_disagreement'] = abs(kappas[0]-kappas[1])/max(abs(kappas[0]),abs(kappas[1]),1e-12)
        (root/f"{case['label']}.json").write_text(json.dumps(record,indent=2))
        records.append(record)
        (root/'diagnostics.json').write_text(json.dumps(records,indent=2))
        volume.commit()
        print(json.dumps({'checkpoint':case['label'],'curvatures':kappas,
                          'eigenvalue':eigenvalue,'residual':record['hessian']['relative_residual']}),flush=True)
        del objective,model,state,probe
        torch.cuda.empty_cache()
    return records


@app.function(image=image,gpu='A10',cpu=4,memory=16384,timeout=1200,
              max_containers=2,scaledown_window=2,volumes={'/vol':volume},secrets=[wandb_secret])
def train_branch(case, suite, dataset, entity):
    """Continue a common checkpoint for 200 constant-LR updates, keeping clipping."""
    from training.trainer import run_training
    from training.errors import NumericalDivergence
    output = ROOT / suite / case['label']
    output.mkdir(parents=True,exist_ok=True)
    config = TrainConfig(**case['train'])
    try:
        result = run_training(config,entity,PROJECT,case['label'],output_dir=output,group=suite,
                              expected_dataset=dataset,branch_from=Path(case['checkpoint']),
                              milestone_updates=(1,50,100,200),probe_every=BRANCH_EVERY,probe_seed=23456,
                              record_sample_hashes=True,x_axis='optimizer_update')
        result['status'] = 'completed'
    except NumericalDivergence as error:
        settings = json.loads((output/'settings.json').read_text())
        metrics = output/'metrics.jsonl'
        hashes = output/'sample_hashes.json'
        result = {'run':case['label'],'status':'diverged','completed':False,
                  'config':case['train'],'dataset_identity':dataset,
                  'history':[json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else [],
                  'sample_hashes':json.loads(hashes.read_text()) if hashes.exists() else [],
                  'branch_origin':settings['branch_origin'],'initial_probe_loss':settings['initial_probe_loss'],
                  'failed_update':error.update,'reason':str(error),'wandb_url':None}
    (output/'result.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


@app.function(image=image,cpu=2,memory=4096,timeout=5400,
              volumes={'/vol':volume},secrets=[wandb_secret])
def run_suite(suite: str, entity: str) -> dict:
    """CPU detrending, one diagnostic GPU, then two isolated continuation arms."""
    import wandb
    from .analysis import fluctuation_scores,branch_summary,publish
    from training.tracking import verify_uploaded_result
    volume.reload()
    source_plan = json.loads((SOURCE_ROOT/'plan.json').read_text())
    results = [json.loads((SOURCE_ROOT/c['label']/'result.json').read_text()) for c in source_plan['cases']]
    scores = fluctuation_scores(results)
    source = next(r for r in results if r['run']=='sgd-batch-16384-lr-0.3')
    cases = branch_plan(source['config'])
    root = ROOT / suite
    root.mkdir(parents=True,exist_ok=True)
    plan = {'source_suite':str(SOURCE_ROOT),'dataset':source['dataset_identity'],
            'checkpoints':checkpoint_plan(),'branches':cases,'branch_probe_every':BRANCH_EVERY,
            'hessian_iterations':HESSIAN_STEPS,'alphas':ALPHAS,'epsilons':EPSILONS,
            'clipping':'Retained at 1.0; no clipping-frequency calculation/report.',
            'prediction':'Lower constant LR may suppress fixed-probe fluctuations; not guaranteed.',
            'scope':'One batch; four saved states; 200 updates per arm. LR also changes decay strength.'}
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    (root/'fluctuations.json').write_text(json.dumps(scores,indent=2))
    volume.commit()
    diagnostics = diagnose.remote(suite)
    branches = list(train_branch.starmap([(case,suite,source['dataset_identity'],entity) for case in cases]))
    measured = branch_summary(branches)
    volume.reload()
    (root/'branch_summary.json').write_text(json.dumps(measured,indent=2))
    volume.commit()
    for result in branches:
        if result['status']=='completed':
            check = verify_uploaded_result(result)
            (root/result['run']/'tracking-check.json').write_text(json.dumps(check,indent=2))
    with wandb.init(entity=entity,project=PROJECT,group=suite,job_type='comparison',
                    name='SGD loss landscape and constant-LR branches',config=plan) as run:
        publish(wandb,run,scores,diagnostics,branches)
        url = run.url
    summary = {'suite':suite,'comparison_url':url,'branches':measured,'diagnostics':diagnostics}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    volume.commit()
    return summary


@app.local_entrypoint()
def main(wandb_entity: str = '', launch: bool = False):
    """Preview is free; --launch explicitly runs the bounded saved-state investigation."""
    print(json.dumps({'checkpoints':checkpoint_plan(),'branch_updates_per_arm':BRANCH_UPDATES,
                      'branch_lrs':[.3,.1],'clipping':1.0,'weight_decay':.1,
                      'hessian_iterations_per_checkpoint':HESSIAN_STEPS,'clip_frequency':'omitted'},indent=2))
    if not launch:
        return
    if not wandb_entity:
        raise ValueError('Supply --wandb-entity before launch')
    suite = f'landscape-{uuid.uuid4().hex[:12]}'
    call = run_suite.spawn(suite,wandb_entity)
    print(json.dumps({'suite':suite,'call_id':call.object_id,'results':str(ROOT/suite)}))


if __name__=='__main__':
    argparse.ArgumentParser(description='Preview; use Modal --launch for execution.').parse_args()
    main()
