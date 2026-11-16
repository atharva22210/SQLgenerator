"""Verify assignment and real two-process gradient math before GPU experiments."""
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
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        initial = Transformer(TransformerConfig(vocab_size=16, d_model=16, n_layers=1,
                                                n_heads=2, n_kv_heads=1, ffn_dim=32, max_seq_len=8))
        all_batches = [make_batch(TinyCorpus(i), mode='masked_pack', batch_size=1, seq_len=8,
                                  sql_fraction=1, pad_id=0, rng=batch_rng(42, 1, i),
                                  validation_kind='sql') for i in range(4)]
        reference = copy.deepcopy(initial)
        accumulated_backward(reference, all_batches, torch.device('cpu'), world_size=1,
                             no_sync=False, loss_chunk_size=2, bf16=False)
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
