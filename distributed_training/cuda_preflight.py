"""Small real-hardware gate before committing to the larger A/B experiments.

The CPU/Gloo tests cannot exercise CUDA FlashAttention, NCCL, BF16 or fused
AdamW. This module tests their combined path on synthetic token IDs, so it can
run before scanning the real dataset or creating a training W&B run.

Success means tiny-model gradients agree with a same-device BF16 reference
and an optimizer step remains finite. It does NOT prove full-model batch memory
fit, long-run convergence, dataset quality, or distributed throughput.
"""
import copy
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataloader import make_batch
from distributed_training.ddp_core import accumulated_backward, batch_rng
from model.transformer_35m import Transformer, TransformerConfig


def main():
    """Run under torchrun with two local ranks; report checks from rank 0 only.

    Arguments come from torchrun's environment, not Python parameters. Both
    ranks calculate the full reference independently, then each trains on half
    the same synthetic microbatches through DDP. Failure raises and stops the
    suite; finally always tears down the NCCL process group.
    """
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    try:
        # Use a tiny vocabulary/context to test the operations cheaply. The
        # real corpus uses the much larger saved tokenizer; IDs here are just
        # synthetic integers, with 0 reserved for PAD and 63 used as EOS.
        torch.manual_seed(7)
        initial = Transformer(TransformerConfig(vocab_size=64, d_model=64, n_layers=1,
                                                n_heads=4, n_kv_heads=2, ffn_dim=128,
                                                max_seq_len=32)).to(device)
        class Corpus:
            """Minimal read interface expected by the real batch builder."""
            def sample(self, kind, rng, length, **kwargs):
                """Use the supplied draw RNG, and respect the remaining row length.

                kind/kwargs match the normal reader API; synthetic data needs
                no English/SQL distinction. Unequal lengths exercise masking.
                """
                return ([rng.randrange(1, 60) for _ in range(rng.randrange(4, 16))] + [63])[:length]
        batches = [make_batch(Corpus(), mode='masked_pack', batch_size=1, seq_len=32,
                              sql_fraction=1, pad_id=0, rng=batch_rng(42, 1, i)) for i in range(4)]
        # No parameter update on the reference: its .grad is our expected
        # global token-mean gradient for these four microbatches. deepcopy keeps
        # identical weights without sharing .grad tensors between models.
        reference = copy.deepcopy(initial)
        accumulated_backward(reference, batches, device, world_size=1, no_sync=False,
                             loss_chunk_size=8)
        # Test both communication schedules from the SAME initial weights.
        # Each rank handles two draws. Normalizing over their global targets
        # should agree with the four-draw reference to BF16 numerical tolerance.
        for no_sync in (False, True):
            raw = copy.deepcopy(initial)
            wrapped = DDP(raw, device_ids=[rank], broadcast_buffers=False)
            accumulated_backward(wrapped, batches[rank * 2:(rank + 1) * 2], device,
                                 world_size=2, no_sync=no_sync, loss_chunk_size=8)
            error = sum((a.grad.float() - b.grad.float()).square().sum() for a, b in zip(raw.parameters(), reference.parameters()))
            norm = sum(b.grad.float().square().sum() for b in reference.parameters())
            # Relative L2 error = ||g_DDP - g_reference|| / ||g_reference||.
            # This is a tolerance check, not an expectation of bit identity.
            # Scalar reads are acceptable in this small diagnostic, but are
            # deliberately avoided inside production backward microsteps.
            relative = (error / norm).sqrt().item()
            if not relative < 0.03:
                raise AssertionError(f'NCCL/BF16 relative gradient error: {relative}')
            # Exercise the fused CUDA optimizer too. This checks finiteness,
            # not exact optimizer equivalence (the CPU integration test checks
            # matching AdamW updates under controlled FP32 arithmetic).
            torch.optim.AdamW(raw.parameters(), lr=1e-3, fused=True).step()
            if any(not torch.isfinite(p).all().item() for p in raw.parameters()):
                raise AssertionError('Nonfinite parameter after fused AdamW')
            if rank == 0:
                print(json.dumps({'preflight': 'passed', 'no_sync': no_sync,
                                  'relative_gradient_error': relative}), flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
