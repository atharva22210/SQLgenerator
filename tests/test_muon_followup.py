"""Small deterministic checks for causal probe masking and online averaging."""
from copy import deepcopy
from dataclasses import replace
import unittest

import torch

from model.transformer_35m import Transformer, TransformerConfig
from pretraining_experiments.muon_qk_decay_ema.probe import causal_statistics
from training.trainer import update_ema


class FollowupTests(unittest.TestCase):
    def test_ema_half_life_and_resume(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(1)
        shadow = deepcopy(model)
        model.weight.data.fill_(3)
        update_ema(shadow, model, 8, 16)
        restored = deepcopy(model)
        restored.load_state_dict(shadow.state_dict())
        update_ema(restored, model, 8, 16)
        torch.testing.assert_close(restored.weight, torch.full_like(model.weight, 2))
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3))

    def test_masked_future_key_excluded(self):
        q = torch.ones(1, 3, 1)
        k = torch.tensor([[[0.], [0.], [100.]]])
        metrics = causal_statistics(q, k, torch.tensor([1]))
        self.assertEqual(metrics['max_score'].item(), 0.)
        self.assertAlmostEqual(metrics['median_entropy'].item(), 1.)

    def test_qk_pair_has_identical_common_weights(self):
        cfg = TransformerConfig(vocab_size=32, d_model=32, n_heads=4, n_kv_heads=2,
                                n_layers=1, ffn_dim=64, max_seq_len=16)
        torch.manual_seed(1)
        off = Transformer(cfg)
        torch.manual_seed(1)
        on = Transformer(replace(cfg, use_qk_norm=True))
        for name, value in off.state_dict().items():
            torch.testing.assert_close(value, on.state_dict()[name])
