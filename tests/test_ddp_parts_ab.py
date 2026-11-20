"""Executable learning checks for distributed-training invariants.

These use tiny synthetic data, not the Modal corpus. The real two-process test
checks the numerical objective as well as communication frequency, because a
program can run without hanging while silently optimizing the wrong average.
The final tests guard spawn-only submission and filtered SQL-only evaluation.
"""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.dataloader import make_batch
from distributed_training.ddp_core import accumulated_backward, batch_rng, microbatch_ids
from distributed_training.experiment_plan import experiment_plan
from model.transformer_35m import Transformer, TransformerConfig


class TinyCorpus:
    """Unequal documents expose incorrectly averaged local mean losses."""
    def __init__(self, index):
        self.index = index

    def sample(self, kind, rng, length, **kwargs):
        return ([1 + self.index] * (self.index + 1) + [9])[:length]


def _counting_average(state, bucket):
    # A test-only hook counts gradient bucket collectives (not target counts).
    # Production uses DDP's asynchronous default reducer for overlap.
    state['calls'] += 1
    tensor = bucket.buffer()
    dist.all_reduce(tensor)
    future = torch.futures.Future()
    future.set_result(tensor / 2)
    return future


def _two_rank_check(rank, rendezvous):
    """CPU worker launched by mp.spawn; rank is 0/1, rendezvous is a file URI.

    The shared rendezvous connects Gloo processes on this machine. One CPU
    thread per process keeps this tiny regression test inexpensive. Real CUDA
    correctness is checked separately by cuda_preflight on the allocated GPUs.
    """
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        initial = Transformer(TransformerConfig(vocab_size=16, d_model=16, n_layers=1,
                                                n_heads=2, n_kv_heads=1, ffn_dim=32, max_seq_len=8))
        # Distinct document lengths give distinct target counts; otherwise an
        # incorrect average of rank means could accidentally pass the test.
        all_batches = [make_batch(TinyCorpus(i), mode='masked_pack', batch_size=1, seq_len=8,
                                  sql_fraction=1, pad_id=0, rng=batch_rng(42, 1, i),
                                  validation_kind='sql') for i in range(4)]
        reference = copy.deepcopy(initial)
        accumulated_backward(reference, all_batches, torch.device('cpu'), world_size=1,
                             no_sync=False, loss_chunk_size=2, bf16=False)
        # Tiny model fits one gradient bucket: two backwards produce two
        # reductions normally, versus one on the final backward with no_sync.
        # This exact bucket count is a test fixture, not a production promise.
        for no_sync, expected_calls in ((False, 2), (True, 1)):
            raw = copy.deepcopy(initial)
            ddp = torch.nn.parallel.DistributedDataParallel(raw, broadcast_buffers=False)
            state = {'calls': 0}
            ddp.register_comm_hook(state, _counting_average)
            batches = [all_batches[i] for i in microbatch_ids(rank, 2, 2)]
            local_sum, counts = accumulated_backward(ddp, batches, torch.device('cpu'),
                                                     world_size=2, no_sync=no_sync,
                                                     loss_chunk_size=2, bf16=False)
            assert counts[0].item() == sum(b.predicted_tokens for b in all_batches)
            assert state['calls'] == expected_calls, (no_sync, state)
            for actual, expected in zip(raw.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
            # Equal gradients alone are insufficient if optimizer integration
            # differs. Check an actual AdamW step on the same initial state.
            actual_opt = torch.optim.AdamW(raw.parameters(), lr=1e-3)
            ref_copy = copy.deepcopy(initial)
            for parameter, expected in zip(ref_copy.parameters(), reference.parameters()):
                parameter.grad = expected.grad.clone()
            expected_opt = torch.optim.AdamW(ref_copy.parameters(), lr=1e-3)
            actual_opt.step()
            expected_opt.step()
            for actual, expected in zip(raw.parameters(), ref_copy.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    finally:
        dist.destroy_process_group()


class DDPPartsTests(unittest.TestCase):
    def test_a_global_draws_match_and_a3_repeats(self):
        single = microbatch_ids(0, 1, 8)
        two = [i for rank in range(2) for i in microbatch_ids(rank, 2, 4)]
        self.assertEqual(single, two)
        self.assertEqual(microbatch_ids(0, 2, 4, True), microbatch_ids(1, 2, 4, True))
        for i in single:
            self.assertEqual(batch_rng(42, 3, i).random(), batch_rng(42, 3, i).random())

    def test_plan_changes_only_intended_variables(self):
        a1, a2, a3 = experiment_plan('a')
        self.assertEqual(a1['world_size'] * a1['train']['grad_accum_steps'],
                         a2['world_size'] * a2['train']['grad_accum_steps'])
        self.assertTrue(a3['duplicate'])
        b1, b2 = experiment_plan('b')
        self.assertEqual({k: v for k, v in b1.items() if k not in ('label', 'no_sync')},
                         {k: v for k, v in b2.items() if k not in ('label', 'no_sync')})
        self.assertFalse(b1['no_sync'])
        self.assertTrue(b2['no_sync'])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_real_ddp_matches_reference_and_skips_communication(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory) / 'gloo-init'
            mp.spawn(_two_rank_check, args=(rendezvous.as_uri(),), nprocs=2, join=True)

    def test_combined_plan_is_a1_a2_b1_b2(self):
        cases = experiment_plan('ab')
        self.assertEqual([case['label'][:2] for case in cases], ['A1', 'A2', 'B1', 'B2'])
        self.assertEqual([case['train']['max_steps'] for case in cases], [300, 300, 100, 100])
        self.assertFalse(any(case['duplicate'] for case in cases))

    def test_launcher_submits_with_spawn(self):
        from unittest.mock import patch, MagicMock
        import modal_ddp_parts_ab as launcher
        call = MagicMock(object_id='fc-test')
        # Mocking the cloud boundary proves the CLI submits without allocating
        # hardware or waiting for remote results during the unit test.
        with patch.object(launcher, 'run_suite') as suite, patch('builtins.print'):
            suite.spawn.return_value = call
            launcher.main(part='ab', wandb_entity='test')
            suite.spawn.assert_called_once()
            suite.remote.assert_not_called()
            self.assertEqual(len(suite.spawn.call_args.args[0]), 4)

    def test_sql_evaluation_only_visits_retained_rows(self):
        from types import SimpleNamespace
        import numpy as np
        from distributed_training.sql_validation import evaluate_sql
        from train_pretrain import TrainConfig
        source = SimpleNamespace(tokens=np.array([7, 9, 2, 3, 9]),
                                 ends=np.array([1, 4]), eligible_indices=np.array([1]))
        corpus = SimpleNamespace(sql=[source])  # Deliberately has no English reader.
        model = Transformer(TransformerConfig(vocab_size=16, d_model=16, n_layers=1,
                                              n_heads=2, n_kv_heads=1, ffn_dim=32, max_seq_len=8))
        metrics = evaluate_sql(model, corpus, TrainConfig(seq_len=8), 0, torch.device('cpu'), full=True)
        self.assertEqual(metrics['val/sql_documents_filtered_all'], 1)
        self.assertEqual(metrics['val/sql_targets_filtered_all'], 2)
        self.assertTrue(model.training)
        self.assertTrue(all('english' not in key for key in metrics))
