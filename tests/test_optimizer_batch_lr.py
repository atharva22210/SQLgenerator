"""Exercise real optimizer updates and the two fixed-token WSD timelines on CPU."""
from copy import deepcopy
from dataclasses import replace
import unittest

import torch

from pretraining_experiments.optimizer_batch_lr.plan import TOKEN_BUDGET, experiment_plan
from training.config import TrainConfig, checkpoint_config_matches
from training.optimizers import build_optimizer
from training.schedules import learning_rate, wsd_boundaries


class OptimizerSweepTests(unittest.TestCase):
    def test_plan_and_measured_boundaries(self):
        cases = experiment_plan()
        self.assertEqual(len({c['label'] for c in cases}), 18)
        for case in cases:
            config = TrainConfig(**case['train'])
            batch = config.micro_batch_size * config.seq_len * config.grad_accum_steps
            self.assertEqual(batch * config.max_steps, TOKEN_BUDGET)
            warm, stable = wsd_boundaries(config)
            self.assertEqual((warm, stable), (25, 200) if config.max_steps == 250 else (12, 100))
            self.assertEqual(stable % config.eval_every, 0)
            self.assertEqual(learning_rate(warm, config), config.peak_lr)
            self.assertAlmostEqual(learning_rate(config.max_steps, config), config.min_lr)

    def test_next_update_after_restoring_each_optimizer(self):
        for name in ('adamw', 'momentum_sgd', 'muon'):
            with self.subTest(optimizer=name):
                model = torch.nn.Module()
                model.token_embedding = torch.nn.Embedding(8, 4)
                model.hidden = torch.nn.Linear(4, 4)
                config = TrainConfig(optimizer=name, peak_lr=.02)
                optimizer = build_optimizer(model, config)
                for p in model.parameters():
                    p.grad = torch.ones_like(p)
                optimizer.step()
                copied = deepcopy(model)
                resumed = build_optimizer(copied, config)
                resumed.load_state_dict(deepcopy(optimizer.state_dict()))
                if name == 'muon':
                    self.assertIs(optimizer.muon.param_groups[0]['params'][0], model.hidden.weight)
                    self.assertIs(optimizer.adamw.param_groups[0]['params'][0], model.token_embedding.weight)
                    self.assertAlmostEqual(resumed.adamw.param_groups[0]['lr_scale'] * config.peak_lr,
                                           config.companion_peak_lr)
                for net, opt in ((model, optimizer), (copied, resumed)):
                    opt.zero_grad(set_to_none=True)
                    for p in net.parameters():
                        p.grad = torch.full_like(p, .25)
                    opt.step()
                for actual, expected in zip(copied.parameters(), model.parameters()):
                    torch.testing.assert_close(actual, expected)

    def test_old_checkpoints_imply_adamw(self):
        config = TrainConfig()
        saved = vars(config).copy()
        del saved['optimizer'], saved['companion_peak_lr']
        self.assertTrue(checkpoint_config_matches(saved, config))
        self.assertFalse(checkpoint_config_matches(saved, replace(config, optimizer='muon')))
