"""Validate submission without starting a Modal App or allocating a GPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from packing_experiment.train import TrainConfig, main


class PretrainSubmissionTests(unittest.TestCase):
    def test_default_submission_exits_without_waiting(self) -> None:
        call = SimpleNamespace(object_id="fc-test", get=Mock())
        with patch("packing_experiment.train.train_on_gpu.spawn", return_value=call) as spawn, \
             patch("builtins.print"):
            main(wandb_entity="test-team")
        call.get.assert_not_called()
        config, entity, project, name = spawn.call_args.args
        self.assertEqual(entity, "test-team")
        self.assertEqual((config["micro_batch_size"], config["grad_accum_steps"],
                          config["seq_len"], config["loss_chunk_size"]), (4, 2, 2048, 128))
        self.assertEqual(config, vars(TrainConfig()))
        self.assertIn("prefix", name)

    def test_optional_wait_gets_spawned_result(self) -> None:
        call = SimpleNamespace(object_id="fc-test", get=Mock(return_value={"updates": 2}))
        with patch("packing_experiment.train.train_on_gpu.spawn", return_value=call), \
             patch("builtins.print"):
            main(wandb_entity="test-team", wait=True)
        call.get.assert_called_once_with()

    def test_invalid_run_is_rejected_before_submission(self) -> None:
        with patch("packing_experiment.train.train_on_gpu.spawn") as spawn:
            with self.assertRaisesRegex(ValueError, "run_name"):
                main(wandb_entity="test-team", run_name="../another-run")
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
