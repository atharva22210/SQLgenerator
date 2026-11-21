"""Check Part C settings, checkpoint replay, and trace overlap accounting."""
import copy
import unittest

import torch

from distributed_training.experiment_plan import experiment_plan
from distributed_training.fsdp_checkpointing import kernel_times, prepare_model
from model.transformer_35m import Transformer, TransformerConfig


class PartCTests(unittest.TestCase):
    def test_only_wrapper_and_block_checkpointing_vary(self):
        cases = experiment_plan('c')
        self.assertEqual(len(cases), 5)
        self.assertEqual([case['checkpoint_blocks'] for case in cases], [0, 12, 0, 0, 12])
        self.assertEqual([case['no_sync'] for case in cases], [True, True, False, False, False])
        self.assertEqual(cases[0]['train']['max_steps'], 20)
        for case in cases:
            self.assertEqual(case['train'], cases[0]['train'])
            self.assertEqual(case['model'], cases[0]['model'])
            self.assertEqual(case['world_size'], 2)

    def test_checkpoint_replay_preserves_tied_weight_gradient(self):
        config = TransformerConfig(vocab_size=32, n_layers=2, d_model=32,
                                   n_heads=4, n_kv_heads=2, ffn_dim=64)
        plain = Transformer(config)
        replay = prepare_model(copy.deepcopy(plain), 'ddp', 1, torch.device('cpu'))
        ids = torch.tensor([[1, 3, 4, 5]])
        for model in (plain, replay):
            _, loss = model(ids, labels=ids, loss_chunk_size=2, valid_target_count=3)
            loss.backward()
        for expected, actual in zip(plain.parameters(), replay.parameters(), strict=True):
            torch.testing.assert_close(expected.grad, actual.grad)

    def test_trace_counts_concurrent_compute_once(self):
        def kernel(name, start, duration):
            return dict(cat='kernel', name=name, ts=start, dur=duration)
        events = [kernel('gemm', 0, 5000), kernel('gemm', 3000, 5000),
                  kernel('ncclKernel', 6000, 4000), dict(cat='gpu_memcpy', ts=0, dur=9999)]
        self.assertEqual(kernel_times(events),
                         {'nccl_kernel_ms': 4, 'other_kernel_ms': 8, 'overlap_ms': 2})
