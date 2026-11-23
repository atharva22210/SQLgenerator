"""Check the warmup follow-up without submitting any GPU work."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import wandb
from training.config import TrainConfig
from training.schedules import learning_rate, wsd_boundaries
from pretraining_experiments.seed_and_schedules import launch
from pretraining_experiments.seed_and_schedules.warmup_comparison import experiment_plan, load_baselines, publish_comparison, stable_region_gaps


class WarmupComparisonTests(unittest.TestCase):
    def test_two_cases_and_exact_boundaries(self):
        cases = experiment_plan()
        self.assertEqual(len(cases), 2)
        for case in cases:
            config = TrainConfig(**case['train'])
            self.assertEqual(config.warmup_steps, 300)
            self.assertEqual(config.micro_batch_size*config.seq_len*config.grad_accum_steps*config.max_steps, 131072000)
            self.assertAlmostEqual(learning_rate(1,config),config.peak_lr/300)
            self.assertEqual(learning_rate(300,config),config.peak_lr)
            self.assertAlmostEqual(learning_rate(1000,config),config.min_lr)
            if config.lr_schedule == 'wsd':
                self.assertEqual(wsd_boundaries(config),(300,800))
                self.assertEqual(learning_rate(800,config),config.peak_lr)
                self.assertLess(learning_rate(801,config),config.peak_lr)
        self.assertTrue(all(c['train']['warmup_steps']==100 for c in launch.experiment_plan()))

    def test_preview_does_not_submit(self):
        with patch.object(launch,'run_suite') as submit, patch('builtins.print'):
            launch.main(dry_run=True,warmup_comparison=True)
            submit.spawn.assert_not_called()
            with self.assertRaises(ValueError): launch.main(dry_run=True,warmup_comparison=True,smoke=True)

    def test_baseline_identity_and_four_result_analysis(self):
        cases=experiment_plan(); baselines=[]; results=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for case in cases:
                for baseline in (True,False):
                    config={**case['train'],**({'warmup_steps':100,'wsd_stable_frac':.7} if baseline else {})}
                    name=case['baseline_label'] if baseline else case['label']
                    history=[]
                    for i in range(1,1001):
                        row={'optimizer_update':i,'train/lr':.001,'train/loss':2.,'train/grad_norm':.5,
                             'perf/cumulative_non_pad_tokens':i*131072}
                        if i%50==0: row.update({'val/sql_loss':1.,'val/english_loss':3.})
                        history.append(row)
                    result={'run':name,'config':config,'completed':True,'dataset_identity':{'id':'fixture'},
                            'history':history,'wandb_url':'https://example.test'}
                    (baselines if baseline else results).append(result)
                    if baseline:
                        (root/name).mkdir();(root/name/'result.json').write_text(json.dumps(result))
            self.assertEqual(len(load_baselines(root,cases,{'id':'fixture'})),2)
            with self.assertRaises(ValueError): load_baselines(root,cases,{'id':'other'})
            summary=publish_comparison(wandb,Mock(),cases,baselines,results)
            self.assertEqual(len(summary['records']),4)
            gaps=summary['stable_region_gaps']['300']
            self.assertEqual(gaps['train/loss']['observations'],500)
            self.assertEqual(gaps['val/sql_loss']['observations'],10)
            self.assertEqual(gaps['val/sql_loss']['equal'],10)
            self.assertEqual(summary['stable_region_gaps']['100']['val/sql_loss']['observations'],14)
            results[1]['history'][299]['val/sql_loss']=9.  # Warmup endpoint: reference only.
            results[1]['history'][349]['val/sql_loss']=2.  # One point above cosine.
            results[1]['history'][849]['val/sql_loss']=100.  # Decay: excluded.
            measured=stable_region_gaps(results[0],results[1],300)['val/sql_loss']
            self.assertEqual(measured['above'],1)
            self.assertEqual(measured['warmup_end_gap'],8.)
            self.assertAlmostEqual(measured['mean_gap'],.1)
            results[0]['history'].pop(0)
            with self.assertRaises(ValueError): publish_comparison(wandb,Mock(),cases,baselines,results)
