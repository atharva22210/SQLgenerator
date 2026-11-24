"""CPU-only checks for the fixed-token batch/LR design and failure accounting."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import wandb
from training.config import TrainConfig
from training.errors import NumericalDivergence
from training.schedules import learning_rate
from training.tracking import curve_payload, define_metrics
from pretraining_experiments.batch_lr_scaling.plan import BATCHES, EVAL_TOKENS, TOKEN_BUDGET, experiment_plan
from pretraining_experiments.batch_lr_scaling.analysis import analyze, publish, report_markdown
from pretraining_experiments.batch_lr_scaling import launch


def results_for(cases):
    """Synthetic histories for validation tests; these are not experiment findings."""
    results = []
    for case in cases:
        config = case['train']; batch = case['batch_tokens']; history = []
        for update in range(1, config['max_steps'] + 1):
            row = {'optimizer_update': update, 'tokens_seen': update * batch,
                   'perf/cumulative_non_pad_tokens': update * batch, 'train/loss': 2., 'train/lr': .001}
            if update % config['eval_every'] == 0 or update == config['max_steps']:
                row.update({'val/sql_loss': 1., 'val/english_loss': 3.})
            history.append(row)
        results.append({'run': case['label'], 'config': config, 'dataset_identity': {'corpus_id': 'fixture'},
                        'completed': True, 'status': 'completed', 'history': history, 'wandb_url': 'https://example.test'})
    return results


class BatchPlanTests(unittest.TestCase):
    def test_equal_budget_proportional_warmup_and_identical_microbatch_count(self):
        cases = experiment_plan()
        self.assertEqual(len(cases), 6)
        for batch, steps, accum, warmup in zip(BATCHES, (1000, 500), (2, 4), (100, 50)):
            subset = [c for c in cases if c['batch_tokens'] == batch]
            self.assertEqual(len(subset), 3)
            for case in subset:
                config = TrainConfig(**case['train'])
                self.assertEqual((config.seq_len, config.micro_batch_size), (4096, 4))
                self.assertEqual((config.max_steps, config.grad_accum_steps, config.warmup_steps), (steps, accum, warmup))
                self.assertEqual(steps * batch, TOKEN_BUDGET)
                self.assertEqual(steps * accum, 2000)
                self.assertEqual(warmup * batch, TOKEN_BUDGET // 10)
                self.assertEqual(config.seed, 1)
                self.assertEqual(config.validation_seed, 12345)
                self.assertAlmostEqual(config.min_lr, config.peak_lr * .1)
                self.assertAlmostEqual(learning_rate(warmup, config), config.peak_lr)
                self.assertAlmostEqual(learning_rate(steps, config), config.min_lr)
        # At equal token milestones, different batch sizes sample the same LR curve.
        for tokens in (3_276_800, 16_384_000, 32_768_000):
            rates = [learning_rate(tokens // c['batch_tokens'], TrainConfig(**c['train']))
                     for c in cases if c['train']['peak_lr'] == .001]
            self.assertTrue(all(abs(rate - rates[0]) < 1e-12 for rate in rates))

    def test_validation_uses_identical_token_coordinates(self):
        cases = experiment_plan(learning_rates=(.001,))
        coordinate_sets = []
        for c in cases:
            config = c['train']; batch = c['batch_tokens']
            coordinates = [i * batch for i in range(1, config['max_steps'] + 1)
                           if i % config['eval_every'] == 0 or i == config['max_steps']]
            self.assertEqual(coordinates[0], EVAL_TOKENS)
            self.assertEqual(coordinates[-1], TOKEN_BUDGET)
            self.assertEqual(len(coordinates), 16)
            coordinate_sets.append(coordinates)
        self.assertTrue(all(points == coordinate_sets[0] for points in coordinate_sets))

    def test_default_preview_never_submits_and_subsets_are_validated(self):
        with patch.object(launch, 'run_suite') as submit, patch('builtins.print'):
            launch.main()
            submit.spawn.assert_not_called()
            submit.spawn.return_value.object_id = 'fc-test'
            launch.main(launch=True, wandb_entity='test', batch_sizes='65536', learning_rates='0.001')
            self.assertEqual(len(submit.spawn.call_args.args[0]), 1)
        for batches, rates in (('123', ''), ('', 'nan'), ('32768,32768', ''), ('', '.001,.001')):
            with self.assertRaises(ValueError): launch.select_cases(batches, rates)
        with patch.object(launch, 'run_suite') as submit, patch('builtins.print'):
            with self.assertRaises(ValueError): launch.main(launch=True)
            submit.spawn.assert_not_called()


class BatchAnalysisTests(unittest.TestCase):
    def test_completed_grid_and_table_sizes_use_tokens(self):
        cases = experiment_plan(); results = results_for(cases)
        summary = analyze(cases, results)
        self.assertEqual(len(summary['records']), 6)
        self.assertEqual(len(summary['best_tested_sql']), 2)
        run = Mock()
        publish(wandb, run, cases, results)
        plots = run.log.call_args.args[0]
        for key, plot in plots.items():
            if '_vs_tokens' in key:
                self.assertLessEqual(len(plot.table.data), 2406)
                self.assertEqual(max(row[0] for row in plot.table.data), TOKEN_BUDGET)
        self.assertIn('not an exact universal constant', report_markdown(summary))

    def test_mismatched_budget_data_or_evaluation_is_rejected(self):
        cases = experiment_plan(batch_sizes=(32768,), learning_rates=(.001,))
        for corrupt in ('budget', 'history', 'evaluation', 'config'):
            results = results_for(cases)
            if corrupt == 'budget': results[0]['history'][-1]['perf/cumulative_non_pad_tokens'] -= 1
            elif corrupt == 'history': results[0]['history'].pop(0)
            elif corrupt == 'evaluation': results[0]['history'][-1].pop('val/sql_loss')
            else: results[0]['config'] = {**results[0]['config'], 'seed': 2}
            with self.assertRaises(ValueError): analyze(cases, results)
        cases = experiment_plan(batch_sizes=(32768,), learning_rates=(.0003, .001))
        results = results_for(cases); results[1]['dataset_identity'] = {'corpus_id': 'different'}
        with self.assertRaises(ValueError): analyze(cases, results)

    def test_divergence_is_excluded_not_replaced_with_fake_loss(self):
        cases = experiment_plan(batch_sizes=(32768,), learning_rates=(.0003, .001))
        results = results_for(cases)
        results[1].update(status='diverged', completed=False, history=[])
        summary = analyze(cases, results)
        self.assertIsNone(summary['records'][1]['sql_loss'])
        self.assertEqual(summary['best_tested_sql'][0]['peak_lr'], .0003)

    def test_worker_only_catches_numerical_divergence(self):
        case = experiment_plan(batch_sizes=(32768,), learning_rates=(.001,))[0]
        with tempfile.TemporaryDirectory() as temp, patch.object(launch, 'OUTPUT_ROOT', Path(temp)), \
             patch.object(launch.volume, 'commit'), patch('training.trainer.run_training') as train:
            train.side_effect = NumericalDivergence(2, 'nonfinite gradient')
            result = launch.execute_case(case, 'fixture', {}, 'test', 'test')
            self.assertEqual(result['status'], 'diverged')
            self.assertEqual(result['failed_update'], 2)
            self.assertEqual(train.call_args.kwargs['milestone_updates'], (100, 500, 1000))
            train.side_effect = RuntimeError('CUDA out of memory')
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                launch.execute_case(case, 'fixture', {}, 'test', 'test')


class ChartPreflightTests(unittest.TestCase):
    def test_actual_sdk_serializes_complete_chart_tables_offline(self):
        cases = experiment_plan()
        results = results_for(cases)
        with tempfile.TemporaryDirectory() as temp:
            with wandb.init(mode='offline', project='batch-lr-local-check', dir=temp) as run:
                define_metrics(run, step_metric='tokens_seen')
                plots = curve_payload(wandb, [results[0]['history']], [results[0]['run']], x_key='tokens_seen')
                self.assertEqual(len(plots), 4)
                for key, plot in plots.items():
                    expected = 16 if 'val_' in key else 1000
                    self.assertEqual(len(plot.table.data), expected)
                    self.assertEqual(plot.table.data[-1][0], TOKEN_BUDGET)
                run.log(plots)
                publish(wandb, run, cases, results)
            self.assertTrue(list(Path(temp).rglob('*.table.json')))
