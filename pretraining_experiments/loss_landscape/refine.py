"""Bounded correction for signed power iteration; never rerun the training arms."""
import json
from pathlib import Path
import modal
from training.config import image,volume
from .launch import ROOT

app = modal.App('sqlgen-landscape-hessian-refinement')


@app.function(image=image,gpu='A10',cpu=4,memory=8192,timeout=900,
              volumes={'/vol':volume},max_containers=1,scaledown_window=2)
def refine(suite: str) -> list[dict]:
    """Only refine checkpoints where negative dominant power obscured lambda_max."""
    import torch
    from model.transformer_35m import Transformer,TransformerConfig
    from .objective import ProbeObjective
    from .numerics import lanczos_curvature
    volume.reload();root=ROOT/suite
    records=json.loads((root/'diagnostics.json').read_text())
    torch.backends.cuda.matmul.allow_tf32=False
    for record in records:
        if record['hessian']['lambda_max_estimate'] is not None:
            continue
        path=Path(record['checkpoint'])
        state=torch.load(path,map_location='cpu',weights_only=False)
        probe=torch.load(path.parent/'fixed_probe.pt',map_location='cpu',weights_only=False)
        model=Transformer(TransformerConfig(**state['model_config'])).cuda().float().eval()
        model.load_state_dict(state['model'])
        objective=ProbeObjective(model,probe['batch'])
        corrected=lanczos_curvature(objective,model.parameters(),steps=12)
        record['hessian']['lanczos']=corrected
        record['hessian']['lambda_max_estimate']=corrected['lambda_max_estimate']
        record['hessian']['positive_estimate_relative_residual']=corrected['relative_residual']
        record['lr_times_curvature']=record['actual_lr']*corrected['lambda_max_estimate']
        (root/f"{record['label']}.json").write_text(json.dumps(record,indent=2))
        print(json.dumps({'checkpoint':record['label'],'refinement':corrected}),flush=True)
        del model,objective,state,probe
        torch.cuda.empty_cache()
    (root/'diagnostics-refined.json').write_text(json.dumps(records,indent=2))
    volume.commit()
    return records


@app.local_entrypoint()
def main(suite: str):
    result=refine.remote(suite)
    Path('/tmp/sqlgen-landscape-diagnostics-refined.json').write_text(json.dumps(result,indent=2))
