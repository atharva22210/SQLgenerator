"""Llama-style normal initialization, baseline parity and training propagation."""
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from distributed_training.experiment_plan import experiment_plan
from model.transformer_35m import RMSNorm, Transformer, TransformerConfig
from training.config import TrainConfig, checkpoint_config_matches


def make_model(initializer_range=0.02, *, seed=42):
    """CPU fixture with enough weights to check sample standard deviations."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Transformer(TransformerConfig(vocab_size=256, d_model=128, n_layers=2,
            n_heads=4, n_kv_heads=2, ffn_dim=256, max_seq_len=8,
            use_qk_norm=True, initializer_range=initializer_range))


class InitializationTests(unittest.TestCase):
    def test_normal_scale_applies_to_all_projections_and_embeddings(self):
        for std in (.02, .01):
            model = make_model(std)
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    self.assertAlmostEqual(module.weight.std(unbiased=False).item(), std,
                                           delta=std * .05, msg=name)
                elif isinstance(module, RMSNorm):
                    torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
            self.assertEqual(list(model.state_dict()).count("token_embedding.weight"), 1)

    def test_default_is_bit_identical_to_original_initialization(self):
        self.assertEqual(TransformerConfig(vocab_size=16).initializer_range, .02)
        self.assertEqual(TrainConfig().initializer_range, .02)
        actual = make_model()

        def original_initializer(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=.02)

        with patch.object(Transformer, "_init_weights", original_initializer):
            baseline = make_model()
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
        for std in (.02, .01):
            first, second = make_model(std), make_model(std)
            for name, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    def test_matches_hugging_face_llama_initializer_for_same_seed_and_shape(self):
        # This checks the actual library rule, not just its configuration name.
        from transformers import LlamaConfig, LlamaForCausalLM
        reference = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=8, initializer_range=.02))
        model = make_model()
        for factory in (lambda: nn.Linear(128, 64, bias=False), lambda: nn.Embedding(64, 128)):
            actual, expected = factory(), factory()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model._init_weights(actual)
                torch.manual_seed(42)
                reference._init_weights(expected)
            torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)

    def test_forward_backward_and_loading_do_not_reinitialize_weights(self):
        model = make_model()
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        _, loss = model(ids, labels=ids)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        restored = make_model(seed=17)
        restored.load_state_dict(model.state_dict())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)

    def test_legacy_checkpoints_only_match_default_scale(self):
        config = TrainConfig()
        legacy = asdict(config)
        legacy.pop("initializer_range")
        self.assertTrue(checkpoint_config_matches(legacy, config))
        self.assertFalse(checkpoint_config_matches(legacy, TrainConfig(initializer_range=.01)))
        changed = TrainConfig(initializer_range=.01)
        self.assertTrue(checkpoint_config_matches(asdict(changed), changed))
        missing = dict(legacy)
        missing.pop("seq_len")
        self.assertFalse(checkpoint_config_matches(missing, config))
        self.assertFalse(checkpoint_config_matches({**legacy, "weight_init": "he"}, config))
        self.assertNotIn("initializer_range", legacy)

    def test_invalid_scale_fails_in_model_and_training_configs(self):
        for invalid in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaisesRegex(ValueError, "initializer_range"):
                TransformerConfig(vocab_size=16, initializer_range=invalid)
            with self.assertRaisesRegex(ValueError, "initializer_range"):
                TrainConfig(initializer_range=invalid)


class InitializationLaunchTests(unittest.TestCase):
    def test_every_distributed_case_gets_the_same_initialization(self):
        for part in ("a", "b", "ab", "c"):
            cases = experiment_plan(part, max_steps=2, seq_len=8, initializer_range=.01)
            self.assertTrue(all(case["train"]["initializer_range"] == .01 for case in cases))
            self.assertTrue(all(case["train"]["seed"] == 42 for case in cases))
            self.assertTrue(all(case["train"]["initializer_range"] == .02 for case in experiment_plan(part)))

    def test_launchers_pass_scale_to_workers_without_launching_gpus(self):
        from pretraining_experiments.sequence_packing import launch as packing
        from distributed_training import launch

        call = SimpleNamespace(object_id="fc-test", get=Mock())
        with patch.object(packing.train_on_gpu, "spawn", return_value=call) as spawn, patch("builtins.print"):
            packing.main(wandb_entity="test", initializer_range=.01)
            self.assertEqual(spawn.call_args.args[0]["initializer_range"], .01)
            self.assertTrue(spawn.call_args.args[3].endswith("-init-std0.01"))
            call.get.assert_not_called()
        with patch.object(launch.run_suite, "spawn", return_value=call) as spawn, patch("builtins.print"):
            launch.main(part="ab", wandb_entity="test", initializer_range=.01)
            self.assertTrue(all(case["train"]["initializer_range"] == .01 for case in spawn.call_args.args[0]))
        for entrypoint, remote in ((packing.main, packing.train_on_gpu), (launch.main, launch.run_suite)):
            with patch.object(remote, "spawn") as spawn:
                with self.assertRaisesRegex(ValueError, "initializer_range"):
                    entrypoint(wandb_entity="test", initializer_range=-1)
                spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
