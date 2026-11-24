"""Check schedule, resume, validation, and chart invariants on CPU."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import wandb

from model.transformer_35m import Transformer, TransformerConfig
from training.checkpointing import restore_checkpoint, save_checkpoint
from training.dataset import dataset_identity
from training.schedules import learning_rate, wsd_boundaries
from training.tracking import curve_payload, define_metrics, log_update, verify_history
from training.trainer import _evaluate
from training.config import TrainConfig, checkpoint_config_matches
from pretraining_experiments.seed_and_schedules.analysis import analyze, report_markdown
from pretraining_experiments.seed_and_schedules.launch import experiment_plan, main, suite_directory


class ScheduleTests(unittest.TestCase):
    def test_wsd_phases_end_at_exact_updates(self):
        for tail in ('linear', 'cosine'):
            config = TrainConfig(lr_schedule='wsd', max_steps=1000, warmup_steps=100, wsd_decay_type=tail)
            self.assertEqual(wsd_boundaries(config), (100, 800))
            rates = [learning_rate(step, config) for step in range(1, 1001)]
            self.assertAlmostEqual(rates[0], config.peak_lr / 100)
            self.assertEqual(rates[99:800], [config.peak_lr] * 701)
            self.assertLess(rates[800], config.peak_lr)
            self.assertAlmostEqual(rates[-1], config.min_lr)
            self.assertTrue(all(left >= right for left, right in zip(rates[800:], rates[801:])))
            self.assertAlmostEqual(rates[899], (config.peak_lr + config.min_lr) / 2)

    def test_bad_phases_and_updates_fail_before_launch(self):
        for options in ({'wsd_stable_frac': .8}, {'wsd_decay_frac': 0}, {'wsd_stable_frac': float('nan')},
                        {'wsd_decay_type': 'quadratic'}, {'warmup_steps': 0}):
            with self.assertRaises(ValueError):
                TrainConfig(lr_schedule='wsd', max_steps=1000, warmup_steps=options.get('warmup_steps', 100),
                            **{key: value for key, value in options.items() if key != 'warmup_steps'})
        for update in (0, 1001):
            with self.assertRaises(ValueError):
                learning_rate(update, TrainConfig(max_steps=1000))

    def test_plan_varies_only_training_seed_or_schedule(self):
        cases = experiment_plan()
        self.assertEqual(len(cases), 5)
        configs = [case['train'] for case in cases]
        baseline = {key: value for key, value in configs[0].items()
                    if key not in {'seed', 'lr_schedule', 'wsd_decay_type'}}
        for config in configs:
            TrainConfig(**config)
            self.assertEqual({key: value for key, value in config.items() if key in baseline}, baseline)
            self.assertEqual((config['seq_len'], config['micro_batch_size'], config['grad_accum_steps']), (4096, 8, 4))
            self.assertEqual(config['validation_seed'], 12345)
        smoke = experiment_plan(smoke=True)
        self.assertEqual(len(smoke), 1)
        self.assertEqual(smoke[0]['train']['max_steps'], 10)
        self.assertEqual(smoke[0]['train']['seq_len'], 4096)
        with patch('pretraining_experiments.seed_and_schedules.launch.run_suite') as submit, patch('builtins.print'):
            submit.spawn.return_value.object_id = 'fc-test'
            main(dry_run=True)
            submit.spawn.assert_not_called()
            main(wandb_entity='test', smoke=True)
            self.assertEqual(len(submit.spawn.call_args.args[0]), 1)
            self.assertRegex(submit.spawn.call_args.args[1], r'^seed-and-schedules-smoke-[0-9a-f]{12}$')
            self.assertEqual(submit.spawn.call_args.args[3], 'sqlgen-seed-and-schedules')

    def test_suite_directory_resolves_new_and_archived_checkpoints(self):
        root = Path('/vol/checkpoints/pretraining_experiments')
        current = 'seed-and-schedules-smoke-0123456789ab'
        archived = 'day07-smoke-073af7c11d79'
        self.assertEqual(suite_directory(current), root / 'seed_and_schedules' / current)
        self.assertEqual(suite_directory(archived), root / 'day07' / archived)
        with self.assertRaises(ValueError):
            suite_directory('seed-and-schedules-../another-suite')


class ResumeAndValidationTests(unittest.TestCase):
    def test_flat_dataset_identity_checks_manifest_and_binary_lengths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared = {'tokenizer_sha256': 'tokenizer', 'tokenizer_vocab_size': 16,
                      'eos_token_id': 0, 'token_dtype': 'uint32'}
            english = {**shared, 'token_counts': {'english_train': 2, 'english_val': 2}}
            sql = {**shared, 'format_version': 2, 'corpus_id': 'current-build',
                   'audit': {'overlaps': {'db_id': 0, 'repo_name': 0, 'query_hash': 0}},
                   'token_counts': {'sql_train': 2, 'sql_val': 2}}
            for key in (*english['token_counts'], *sql['token_counts']):
                (root / f'{key}.bin').write_bytes(bytes(8))
            (root / 'metadata.json').write_text(json.dumps(english))
            manifest = root / 'sql_metadata.json'
            manifest.write_text(json.dumps(sql))
            first = dataset_identity(root)
            self.assertEqual(first['sql_dir'], str(root))
            sql['corpus_id'] = 'new-build'
            manifest.write_text(json.dumps(sql))
            self.assertNotEqual(first, dataset_identity(root))
            sql['tokenizer_sha256'] = 'different-tokenizer'
            manifest.write_text(json.dumps(sql))
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                dataset_identity(root)
            sql['tokenizer_sha256'] = 'tokenizer'
            manifest.write_text(json.dumps(sql))
            (root / 'sql_train.bin').write_bytes(bytes(4))
            with self.assertRaisesRegex(ValueError, 'completed'):
                dataset_identity(root)

    def test_resume_matches_uninterrupted_updates_and_sample_stream(self):
        config = TrainConfig(lr_schedule='wsd', max_steps=10, warmup_steps=1)
        architecture = TransformerConfig(vocab_size=16, d_model=16, n_layers=1,
                                         n_heads=2, n_kv_heads=1, ffn_dim=32, max_seq_len=6)
        def setup(seed):
            torch.manual_seed(seed)
            model = Transformer(architecture)
            return model, torch.optim.AdamW(model.parameters(), lr=config.peak_lr), random.Random(seed)
        def updates(model, optimizer, rng, first, last):
            history = []
            for update in range(first, last + 1):
                ids = torch.randint(0, 16, (2, 6))
                ids[0, 0] = rng.randrange(16)
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(ids, labels=ids)
                loss.backward()
                optimizer.param_groups[0]['lr'] = learning_rate(update, config)
                optimizer.step()
                history.append(loss.item())
            return history
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as temp:
            model, optimizer, rng = setup(7)
            expected_losses = updates(model, optimizer, rng, 1, 10)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            expected_rng = (rng.random(), torch.rand(1))
            partial, opt, stream = setup(7)
            losses = updates(partial, opt, stream, 1, 4)
            path = Path(temp) / 'latest.pt'
            save_checkpoint(path, partial, opt, config, 4, stream, 123, 'test-id', Path(temp),
                            history=[{'optimizer_update': 4}])
            resumed, resumed_opt, resumed_rng = setup(999)
            state = torch.load(path, weights_only=False)
            restore_checkpoint(state, resumed, resumed_opt, config, resumed_rng)
            losses += updates(resumed, resumed_opt, resumed_rng, 5, 10)
            self.assertEqual(losses, expected_losses)
            for key, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertEqual(resumed_rng.random(), expected_rng[0])
            torch.testing.assert_close(torch.rand(1), expected_rng[1], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'config'):
                restore_checkpoint(state, resumed, resumed_opt, replace(config, peak_lr=.001), resumed_rng)
        # Real older snapshots lack every newly introduced setting.
        legacy = asdict(TrainConfig())
        for key in ('initializer_range', 'validation_seed', 'wsd_stable_frac', 'wsd_decay_frac', 'wsd_decay_type'):
            legacy.pop(key)
        self.assertTrue(checkpoint_config_matches(legacy, TrainConfig()))
        self.assertFalse(checkpoint_config_matches(legacy, TrainConfig(validation_seed=12345)))

    def test_validation_draws_stay_fixed_across_training_seeds(self):
        draws = []
        def batch(*args, rng, **kwargs):
            draws.append(rng.random())
            return SimpleNamespace(predicted_tokens=2, model_inputs=lambda *args: {})
        model = Mock(return_value=(None, torch.tensor(1.)))
        config = TrainConfig(validation_seed=12345, val_batches=3)
        with patch('data.dataloader.make_batch', side_effect=batch), patch('torch.autocast', return_value=nullcontext()):
            for seed in (1, 2, 3):
                _evaluate(model, None, replace(config, seed=seed), 0, torch.device('cpu'))
        self.assertEqual(draws[:6], draws[6:12])
        self.assertEqual(draws[:6], draws[12:])


class TrackingTests(unittest.TestCase):
    def test_scalar_rows_sparse_validation_and_real_chart_tables(self):
        run, output, history = Mock(), io.StringIO(), []
        define_metrics(run)
        run.define_metric.assert_any_call('val/*', step_metric='optimizer_update')
        for update in (1, 2, 3):
            metrics = {'train/lr': .001 / update, 'train/loss': 4. / update}
            if update == 3:
                metrics.update({'val/sql_loss': 2., 'val/english_loss': 3.})
            log_update(run, update, metrics, history, output)
        self.assertEqual(run.log.call_args_list[0].kwargs, {})
        self.assertNotIn('val/sql_loss', history[0])
        self.assertEqual(len(output.getvalue().splitlines()), 3)
        plots = curve_payload(wandb, [history], ['actual'])
        self.assertEqual(plots['charts/val_sql_loss'].table.data, [[3, 'actual', 2.]])
        self.assertEqual(plots['charts/train_lr'].table.data,
                         [[i, 'actual', .001 / i] for i in (1, 2, 3)])
        with self.assertRaises(ValueError):
            log_update(run, 3, {'train/loss': 1.}, history, output)
        with self.assertRaises(ValueError):
            log_update(run, 4, {'train/loss': float('nan')}, history, output)

    def test_offline_sdk_accepts_scalar_axes_and_all_four_charts(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = []
            with wandb.init(project='sqlgen-seed-and-schedules-local-test', mode='offline', dir=temp,
                            settings=wandb.Settings(console='off', silent=True)) as run:
                define_metrics(run)
                for step in (1, 2):
                    log_update(run, step, {'train/lr': .001, 'train/loss': 2.,
                                          'val/sql_loss': 1., 'val/english_loss': 3.}, rows, io.StringIO())
                run.log(curve_payload(wandb, [rows], ['offline']))
                for key in ('train_lr', 'train_loss', 'val_sql_loss', 'val_english_loss'):
                    self.assertIn(f'charts/{key}_table', run.summary.keys())
                    self.assertEqual(run.summary[f'charts/{key}_table']['nrows'], 2)

    def test_public_api_check_rejects_missing_history_or_charts(self):
        api, remote = Mock(), Mock()
        api.run.return_value = remote
        remote.config = {}
        remote.scan_history.return_value = [{'optimizer_update': i, 'train/lr': .001, 'train/loss': 1.,
                                             'val/sql_loss': 2., 'val/english_loss': 3.} for i in (1, 2)]
        remote.scan_history.return_value.append({'optimizer_update': None, 'train/lr': None,
                                                'train/loss': None, 'val/sql_loss': None,
                                                'val/english_loss': None})
        remote.summary = {f'charts/{key}_table': {'nrows': 2, 'ncols': 3} for key in ('train_lr', 'train_loss', 'val_sql_loss', 'val_english_loss')}
        self.assertTrue(verify_history(api, 'team/project/id', [1, 2], [1, 2])['chart_tables_verified'])
        remote.scan_history.assert_called_once_with(use_cache=False)
        with self.assertRaisesRegex(RuntimeError, 'history'):
            verify_history(api, 'team/project/id', [1, 2, 3], [1, 2])
        remote.summary = {}
        with self.assertRaisesRegex(RuntimeError, 'table'):
            verify_history(api, 'team/project/id', [1, 2], [1, 2])


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def fixture_results():
        results = []
        for index, case in enumerate(experiment_plan(max_steps=10)):
            config = case['train']
            rows = [{'optimizer_update': i, 'perf/cumulative_non_pad_tokens': i * 100}
                    for i in range(1, 11)]
            rows[-1].update({'val/sql_loss': 1. + index * .1, 'val/english_loss': 2. + index * .1})
            results.append({'config': config, 'completed': True, 'history': rows,
                            'dataset_identity': {'manifest_sha256': 'test'}, 'run': case['label'], 'wandb_url': 'https://example.test/run'})
        return results

    def test_analysis_reports_sample_deviation_and_paired_difference(self):
        analysis = analyze(self.fixture_results())
        metric = analysis['metrics']['val/sql_loss']
        self.assertAlmostEqual(metric['mean'], 1.1)
        self.assertAlmostEqual(metric['sample_std'], .1)
        self.assertAlmostEqual(metric['paired_seed_1_improvement']['linear'], -.3)
        text = report_markdown(analysis)
        for section in ('Question', 'Prediction', 'Controlled settings', 'Measured results', 'Explanation', 'Limitations', 'Next experiment'):
            self.assertIn('## ' + section, text)
        self.assertIn('not statistical significance', text)

    def test_analysis_rejects_incomplete_or_uncontrolled_comparisons(self):
        for mutation in ('seed', 'dataset', 'incomplete', 'history'):
            results = self.fixture_results()
            if mutation == 'seed': results[1]['config']['validation_seed'] = 999
            elif mutation == 'dataset': results[1]['dataset_identity'] = {}
            elif mutation == 'incomplete': results[1]['completed'] = False
            else: results[1]['history'].pop(0)
            with self.assertRaises(ValueError):
                analyze(results)
