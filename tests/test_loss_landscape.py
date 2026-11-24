"""Numerical correctness and paired-state checks; no remote submission."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch
import wandb
from data.dataloader import make_batch
from model.transformer_35m import Transformer,TransformerConfig
from training.config import TrainConfig
from training.optimizers import build_optimizer
from training.checkpointing import save_checkpoint,restore_branch,restore_checkpoint
from training.probes import fixed_batch_loss,batch_fingerprint
from pretraining_experiments.loss_landscape.numerics import detrended_score,directional_slice,power_curvature,lanczos_curvature
from pretraining_experiments.loss_landscape.objective import ProbeObjective
from pretraining_experiments.loss_landscape.plan import branch_plan,checkpoint_plan
from pretraining_experiments.loss_landscape.analysis import branch_summary,fluctuation_scores,publish
from pretraining_experiments.loss_landscape import launch


class NumericsTests(unittest.TestCase):
    def test_detrending_removes_quadratic_and_exposes_added_fluctuation(self):
        x=np.arange(1,17,dtype=float)
        smooth=3-.2*x+.004*x*x
        self.assertLess(detrended_score(x,smooth)['residual_sample_std'],1e-12)
        self.assertGreater(detrended_score(x,smooth+.1*(-1.)**x)['residual_sample_std'],.09)
        with self.assertRaises(ValueError):detrended_score([1,1,2,3],[1,2,3,4])

    def test_slice_curvature_and_restoration_even_on_failure(self):
        theta=torch.nn.Parameter(torch.tensor([.4,-.2],dtype=torch.float64))
        origin=theta.detach().clone(); h=torch.tensor([2.,6.],dtype=torch.float64)
        def loss():return .5*(h*theta.square()).sum()
        result=directional_slice([('p',theta)],{'p':torch.tensor([3.,4.],dtype=torch.float64)},loss,[-.02,-.01,0,.01,.02],[.01,.02])
        for r in result['curvatures']:self.assertAlmostEqual(r['kappa'],4.56,places=9)
        torch.testing.assert_close(theta,origin,rtol=0,atol=0)
        def fail():raise RuntimeError('intentional')
        with self.assertRaises(RuntimeError):directional_slice([('p',theta)],{'p':torch.ones(2)},fail,[0,.01],[.01])
        torch.testing.assert_close(theta,origin,rtol=0,atol=0)

    def test_hvp_power_iteration_keeps_sign_and_reports_residual(self):
        theta=torch.nn.Parameter(torch.tensor([.4,-.2],dtype=torch.float64))
        original=theta.detach().clone()
        for diagonal,positive in [([2.,8.],True),([-9.,4.],False)]:
            h=torch.tensor(diagonal,dtype=torch.float64)
            result=power_curvature(lambda:.5*(h*theta.square()).sum(),[theta],steps=20)
            self.assertEqual(result['positive_dominant'],positive)
            self.assertLess(result['relative_residual'],.001)
            self.assertAlmostEqual(result['dominant_eigenvalue_estimate'],8. if positive else -9.,places=5)
            if not positive:self.assertIsNone(result['lambda_max_estimate'])
        torch.testing.assert_close(theta,original,rtol=0,atol=0)
        self.assertIsNone(theta.grad)

    def test_trimmed_probe_matches_reference_and_supports_second_derivatives(self):
        class Corpus:
            def sample(self,kind,rng,length,**kwargs):return [1,2,3,4]
        config=TrainConfig(optimizer='sgd',mode='masked_pack',seq_len=8,micro_batch_size=1)
        architecture=TransformerConfig(vocab_size=16,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq_len=8)
        model=Transformer(architecture).eval()
        batch=make_batch(Corpus(),mode='masked_pack',batch_size=1,seq_len=8,sql_fraction=.5,pad_id=0,rng=random.Random(1),validation_kind='english')
        objective=ProbeObjective(model,[batch])
        self.assertAlmostEqual(float(objective().detach()),fixed_batch_loss(model,[batch],config,torch.device('cpu')),places=6)
        estimate=power_curvature(objective,model.parameters(),steps=2)
        self.assertTrue(np.isfinite(estimate['dominant_eigenvalue_estimate']))

    def test_lanczos_recovers_largest_positive_when_negative_dominates(self):
        theta=torch.nn.Parameter(torch.tensor([.4,-.2],dtype=torch.float64))
        h=torch.tensor([-9.,4.],dtype=torch.float64)
        estimate=lanczos_curvature(lambda:.5*(h*theta.square()).sum(),[theta],steps=12)
        self.assertAlmostEqual(estimate['lambda_max_estimate'],4.,places=10)
        self.assertLess(estimate['relative_residual'],1e-10)


class BranchTests(unittest.TestCase):
    def test_restore_same_origin_replays_inputs_and_only_schedule_changes(self):
        class Corpus:
            def sample(self,kind,rng,length,**kwargs):return [rng.randrange(1,15),2,3]
        config=TrainConfig(optimizer='sgd',mode='masked_pack',seq_len=6,micro_batch_size=1,peak_lr=.3,min_lr=.03)
        architecture=TransformerConfig(vocab_size=16,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq_len=6)
        with torch.random.fork_rng(devices=[]),tempfile.TemporaryDirectory() as temp:
            source=Transformer(architecture); opt=build_optimizer(source,config); rng=random.Random(9)
            path=Path(temp)/'source.pt'
            save_checkpoint(path,source,opt,config,7,rng,42,'source',Path(temp),probe_recipe={'every':1,'seed':23456})
            original=path.read_bytes();state=torch.load(path,weights_only=False)
            hashes=[];random_draws=[]
            for lr in (.3,.1):
                branch_config=replace(config,peak_lr=lr,min_lr=lr,max_steps=200,warmup_steps=0)
                model=Transformer(architecture);optimizer=build_optimizer(model,branch_config);stream=random.Random(999)
                restore_branch(state,model,optimizer,branch_config,stream)
                hashes.append([batch_fingerprint([make_batch(Corpus(),mode='masked_pack',batch_size=1,seq_len=6,sql_fraction=.5,pad_id=0,rng=stream)])for _ in range(3)])
                random_draws.append(torch.rand(3))
                for a,b in zip(model.parameters(),source.parameters()):torch.testing.assert_close(a,b)
                with self.assertRaisesRegex(ValueError,'config'):
                    restore_checkpoint(state,model,optimizer,branch_config,stream)
                for change in ({'grad_clip':.5},{'weight_decay':0.},{'grad_accum_steps':4}):
                    with self.assertRaises(ValueError):restore_branch(state,model,optimizer,replace(branch_config,**change),stream)
            self.assertEqual(hashes[0],hashes[1]);torch.testing.assert_close(*random_draws,rtol=0,atol=0)
            self.assertEqual(path.read_bytes(),original)

    @staticmethod
    def fixtures():
        source=asdict(TrainConfig(optimizer='sgd',peak_lr=.3,min_lr=.03,mode='masked_pack',grad_accum_steps=1))
        cases=branch_plan(source);results=[]
        for case in cases:
            config={**case['train'],'max_steps':4}
            h=[{'optimizer_update':i,'tokens_seen':i*16384,'train/loss':2.,'probe/fixed_loss':2.+.01*(-1)**i,'val/sql_loss':1.,'val/english_loss':3.}for i in range(1,5)]
            results.append({'run':case['label'],'status':'completed','config':config,'history':h,
                            'sample_hashes':['a','b','c','d'],'branch_origin':{'checkpoint':'same','update':500},
                            'dataset_identity':{'same':True},'initial_probe_loss':2.,'wandb_url':'https://example.test'})
        return results

    def test_paired_summary_rejects_mismatched_future_data_and_controls(self):
        r=self.fixtures();summary=branch_summary(r)
        self.assertTrue(summary['actual_future_inputs_match'])
        self.assertEqual(summary['matched_successful_updates'],4)
        r[1]['sample_hashes'][2]='wrong'
        with self.assertRaisesRegex(ValueError,'future'):branch_summary(r)
        r=self.fixtures();r[1]['config']['grad_clip']=.5
        with self.assertRaisesRegex(ValueError,'Only'):branch_summary(r)
        with patch.object(launch,'run_suite')as submit,patch('builtins.print'):
            launch.main();submit.spawn.assert_not_called()
        self.assertEqual(len(checkpoint_plan()),4)

    def test_actual_sdk_serializes_all_followup_chart_payloads(self):
        rows=[]
        for lr in (.1,.3):
            config={'micro_batch_size':4,'seq_len':4096,'grad_accum_steps':1,'max_steps':2000,'peak_lr':lr}
            h=[{'tokens_seen':i*2097152,'val/sql_loss':3.-i*.01,'val/english_loss':5.-i*.01}for i in range(1,16)]
            h.append({'tokens_seen':32768000,'val/sql_loss':2.8,'val/english_loss':4.8})
            rows.append({'run':f'lr-{lr}','completed':True,'config':config,'history':h})
        scores=fluctuation_scores(rows)
        diagnostics=[{'label':'fixture','peak_lr':.3,'tokens_seen':8192000,
                      'slice':{'points':[{'alpha':a,'loss':1.+a*a}for a in (-.01,0,.01)],
                               'curvatures':[{'epsilon':.01,'kappa':2.}]},
                      'actual_lr':.3,'hessian':{'dominant_eigenvalue_estimate':2.,'relative_residual':.01},'lr_times_curvature':.6}]
        with tempfile.TemporaryDirectory()as temp:
            with wandb.init(mode='offline',project='landscape-local-check',dir=temp,settings=wandb.Settings(silent=True))as run:
                publish(wandb,run,scores,diagnostics,self.fixtures())
                self.assertEqual(run.summary['landscape/slices_table']['nrows'],3)
                self.assertEqual(run.summary['branches/probe_fixed_loss_table']['nrows'],8)
                diagnostics[0]['hessian']['lanczos']={'steps':12}
                publish(wandb,run,scores,diagnostics,self.fixtures())
                self.assertEqual(run.summary['hessian/refined_lr_times_curvature_table']['nrows'],3)
