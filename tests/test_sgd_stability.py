"""CPU checks for the SGD experiment's update, replay and observation invariants."""
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import wandb
from data.dataloader import make_batch
from model.transformer_35m import Transformer, TransformerConfig
from training.checkpointing import save_checkpoint, restore_checkpoint
from training.config import TrainConfig
from training.optimizers import build_optimizer
from training.probes import fixed_batch_loss, update_direction
from training.schedules import learning_rate
from training.tracking import curve_payload, define_metrics, verify_history
from pretraining_experiments.sgd_stability.plan import TOKEN_BUDGET, EVAL_TOKENS, experiment_plan
from pretraining_experiments.sgd_stability import launch
from pretraining_experiments.sgd_stability.analysis import analyze, publish


def fixture_results():
    results=[]
    for case in experiment_plan():
        c=case['train']; batch=case['batch_tokens']; rows=[]
        for u in range(1,c['max_steps']+1):
            row={'optimizer_update':u,'tokens_seen':u*batch,'train/loss':2.,'train/lr':c['peak_lr'],
                 'train/clipped':0,'probe/fixed_loss':2.}
            if u%c['eval_every']==0 or u==c['max_steps']:
                row.update({'val/sql_loss':1.,'val/english_loss':3.})
            rows.append(row)
        results.append({'run':case['label'],'config':c,'status':'completed','completed':True,
                        'dataset_identity':{'fixture':True},'history':rows,'wandb_url':'https://example.test'})
    return results


class SGDStabilityTests(unittest.TestCase):
    def test_six_cases_retain_controls_and_match_budget_and_validation(self):
        cases=experiment_plan()
        self.assertEqual(len(cases),6)
        for case in cases:
            c=TrainConfig(**case['train']); batch=case['batch_tokens']
            self.assertEqual((c.optimizer,c.grad_clip,c.weight_decay,c.lr_schedule),('sgd',1.,.1,'cosine'))
            self.assertEqual((c.seq_len,c.micro_batch_size,c.seed,c.validation_seed),(4096,4,1,12345))
            self.assertEqual(c.max_steps*batch,TOKEN_BUDGET)
            self.assertEqual(c.max_steps*c.grad_accum_steps,2000)
            self.assertEqual(c.warmup_steps*batch,TOKEN_BUDGET//10)
            self.assertAlmostEqual(learning_rate(c.warmup_steps,c),c.peak_lr)
            self.assertAlmostEqual(learning_rate(c.max_steps,c),c.min_lr)
            self.assertEqual(c.eval_every*batch,EVAL_TOKENS)
            self.assertIn(1,launch.milestones(c))
            self.assertIn(c.max_steps//2,launch.milestones(c))
        with patch.object(launch,'run_suite') as submit, patch('builtins.print'):
            launch.main()
            submit.spawn.assert_not_called()

    def test_plain_sgd_has_no_momentum_and_decays_outside_clipped_gradient(self):
        model=torch.nn.Module(); model.token_embedding=torch.nn.Embedding(2,2)
        model.norm=torch.nn.Parameter(torch.full((2,),2.))
        with torch.no_grad(): model.token_embedding.weight.fill_(2.)
        config=TrainConfig(optimizer='sgd',peak_lr=.1,weight_decay=.1)
        optimizer=build_optimizer(model,config)
        for p in model.parameters(): p.grad=torch.full_like(p,10.)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gradients={n:p.grad.clone() for n,p in model.named_parameters()}
        optimizer.step()
        for name,p in model.named_parameters():
            decay=.1 if p.ndim>=2 else 0
            torch.testing.assert_close(p,before[name]*(1-.1*decay)-.1*gradients[name])
        self.assertEqual(optimizer.state,{})
        # State replay must preserve group-specific decay and future behavior.
        copied=deepcopy(model); resumed=build_optimizer(copied,config)
        resumed.load_state_dict(deepcopy(optimizer.state_dict()))
        for net,opt in ((model,optimizer),(copied,resumed)):
            for p in net.parameters(): p.grad=torch.ones_like(p)
            opt.step()
        for p,q in zip(model.parameters(),copied.parameters()): torch.testing.assert_close(p,q)

    def test_probe_preserves_parameters_gradients_mode_and_rng_and_checkpoint_direction(self):
        class Corpus:
            def sample(self,kind,rng,length,**kwargs): return [rng.randrange(1,15),2,3][:length]
        config=TrainConfig(optimizer='sgd',mode='masked_pack',seq_len=6,micro_batch_size=1)
        architecture=TransformerConfig(vocab_size=16,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq_len=6)
        with torch.random.fork_rng(devices=[]),tempfile.TemporaryDirectory() as temp:
            model=Transformer(architecture); optimizer=build_optimizer(model,config)
            batch=make_batch(Corpus(),mode='masked_pack',batch_size=1,seq_len=6,sql_fraction=.5,pad_id=0,rng=random.Random(99))
            _,loss=model(**batch.model_inputs('masked_pack',torch.device('cpu')),loss_chunk_size=128)
            loss.backward()
            params={n:p.detach().clone() for n,p in model.named_parameters()}
            grads={n:p.grad.clone() for n,p in model.named_parameters()}
            rng=torch.get_rng_state().clone()
            first=fixed_batch_loss(model,[batch],config,torch.device('cpu'))
            second=fixed_batch_loss(model,[batch],config,torch.device('cpu'))
            self.assertEqual(first,second); self.assertTrue(model.training)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            for name,p in model.named_parameters():
                torch.testing.assert_close(p,params[name],rtol=0,atol=0)
                torch.testing.assert_close(p.grad,grads[name],rtol=0,atol=0)
            optimizer.step(); direction=update_direction(model,params)
            for name,p in model.named_parameters(): torch.testing.assert_close(direction[name],p-params[name])
            path=Path(temp)/'state.pt'; stream=random.Random(1)
            recipe={'every':1,'seed':23456}
            save_checkpoint(path,model,optimizer,config,1,stream,6,'fixture',Path(temp),update_direction=direction,probe_recipe=recipe)
            state=torch.load(path,weights_only=False)
            self.assertEqual(state['probe_recipe'],recipe)
            self.assertEqual(set(state['update_direction']),set(params))
            copied=Transformer(architecture); opt=build_optimizer(copied,config)
            restore_checkpoint(state,copied,opt,config,random.Random(8))
            for p,q in zip(model.parameters(),copied.parameters()): torch.testing.assert_close(p,q)
            with self.assertRaisesRegex(ValueError,'Probe interval'):
                from training.trainer import run_training
                run_training(config,'x','x','x',probe_every=-1)

    def test_measured_analysis_requires_full_budget_and_probe_and_handles_failure(self):
        rows=fixture_results(); result=analyze(rows)
        self.assertEqual(len(result['records']),6)
        rows[0]['history'][-1].pop('probe/fixed_loss')
        with self.assertRaisesRegex(ValueError,'diagnostics'): analyze(rows)
        rows=fixture_results(); rows[0].update(status='diverged',completed=False,history=[],failed_update=1)
        result=analyze(rows)
        self.assertIsNone(result['records'][0]['sql_loss'])

    def test_real_wandb_serialization_and_probe_upload_verification(self):
        rows=fixture_results()
        with tempfile.TemporaryDirectory() as temp:
            with wandb.init(mode='offline',project='sgd-local-only',dir=temp,settings=wandb.Settings(silent=True)) as run:
                define_metrics(run,step_metric='tokens_seen')
                publish(wandb,run,rows)
                self.assertEqual(run.summary['charts/fixed_probe_loss_table']['nrows'],9000)
                self.assertEqual(run.summary['charts/val_sql_loss_table']['nrows'],96)
        remote=Mock(); remote.config={'probe_recipe':{'every':1}}
        remote.scan_history.return_value=[{'optimizer_update':i,'train/lr':.1,'train/loss':2.,'val/sql_loss':1.,'val/english_loss':3.,'probe/fixed_loss':2.} for i in (1,2)]
        remote.summary={f'charts/{k}_table':{'nrows':2,'ncols':3} for k in ('train_lr','train_loss','val_sql_loss','val_english_loss','fixed_probe_loss')}
        api=Mock(); api.run.return_value=remote
        self.assertTrue(verify_history(api,'fixture',[1,2],[1,2])['chart_tables_verified'])
        remote.summary.pop('charts/fixed_probe_loss_table')
        with self.assertRaisesRegex(RuntimeError,'probe chart'): verify_history(api,'fixture',[1,2],[1,2])
