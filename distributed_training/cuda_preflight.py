"""Small two-A10 NCCL/BF16 gradient check before long A/B suites."""
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
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    try:
        torch.manual_seed(7)
        initial = Transformer(TransformerConfig(vocab_size=64, d_model=64, n_layers=1,
                                                n_heads=4, n_kv_heads=2, ffn_dim=128,
                                                max_seq_len=32)).to(device)
        class Corpus:
            def sample(self, kind, rng, length, **kwargs):
                return ([rng.randrange(1, 60) for _ in range(rng.randrange(4, 16))] + [63])[:length]
        batches = [make_batch(Corpus(), mode='masked_pack', batch_size=1, seq_len=32,
                              sql_fraction=1, pad_id=0, rng=batch_rng(42, 1, i)) for i in range(4)]
        reference = copy.deepcopy(initial)
        accumulated_backward(reference, batches, device, world_size=1, no_sync=False,
                             loss_chunk_size=8)
        for no_sync in (False, True):
            raw = copy.deepcopy(initial)
            wrapped = DDP(raw, device_ids=[rank], broadcast_buffers=False)
            accumulated_backward(wrapped, batches[rank * 2:(rank + 1) * 2], device,
                                 world_size=2, no_sync=no_sync, loss_chunk_size=8)
            error = sum((a.grad.float() - b.grad.float()).square().sum() for a, b in zip(raw.parameters(), reference.parameters()))
            norm = sum(b.grad.float().square().sum() for b in reference.parameters())
            relative = (error / norm).sqrt().item()
            if not relative < 0.03:
                raise AssertionError(f'NCCL/BF16 relative gradient error: {relative}')
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
