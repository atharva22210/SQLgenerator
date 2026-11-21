"""C3 memory experiment: fixed model, data, and optimizer across wrappers."""
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from training.config import TrainConfig


def part_c_plan(steps: int, batch: int, seq_len: int, sql_fraction: float,
                eval_every: int, val_batches: int, checkpoint_blocks: int,
                accumulation: int, *, initializer_range: float = 0.02) -> list[dict]:
    """Return JSON-ready C3 baseline and fixes; batch and accumulation are per GPU.

    checkpoint_blocks selects the first N of 24 blocks in both checkpointed
    runs. Vocabulary-loss chunking remains enabled identically in every run.
    initializer_range is fixed across wrappers so memory comparisons do not also
    change the initial model distribution.
    """
    if not 1 <= checkpoint_blocks <= 24:
        raise ValueError('Part C requires between 1 and 24 checkpointed blocks')
    steps = steps or 20
    train = TrainConfig(mode='masked_pack', micro_batch_size=batch or 4,
                        grad_accum_steps=accumulation, max_steps=steps,
                        warmup_steps=min(steps, 5), seq_len=seq_len,
                        sql_fraction=sql_fraction, eval_every=eval_every,
                        val_batches=val_batches, initializer_range=initializer_range)
    cases = [
        ('C3-ddp-baseline', 'ddp', 0),
        ('C3-fix1-ddp-checkpoint', 'ddp', checkpoint_blocks),
        ('C3-fix2-shard-grad-op', 'fsdp_zero2', 0),
        ('C3-fix3-full-shard', 'fsdp_zero3', 0),
        ('C3-fix4-full-shard-checkpoint', 'fsdp_zero3', checkpoint_blocks),
    ]
    return [
        dict(label=label, part_c=True, world_size=2, duplicate=False, no_sync=method == 'ddp',
             method=method, checkpoint_blocks=blocks, profile_update=steps,
             benchmark_warmup=min(5, max(0, steps - 2)), train=asdict(train),
             model=dict(n_layers=24, d_model=1536, n_heads=24, n_kv_heads=12, ffn_dim=4096))
        for label, method, blocks in cases
    ]


def prepare_model(model, method: str, checkpoint_blocks: int, device):
    """Wrap selected blocks and return an FSDP model or a device-local DDP input.

    Each FSDP block is a separate gather unit. The root owns the tied embedding
    so input lookup and chunked vocabulary loss share the same gathered weight.
    Checkpoint inside the FSDP unit so replay does not invoke its forward gather.
    """
    for index in range(checkpoint_blocks):
        model.blocks[index] = checkpoint_wrapper(model.blocks[index])
    if method == 'ddp':
        return model.to(device)
    strategy = {
        'fsdp_zero2': ShardingStrategy.SHARD_GRAD_OP,
        'fsdp_zero3': ShardingStrategy.FULL_SHARD,
    }[method]
    # Original parameters preserve the optimizer's pre-wrap decay groups,
    # even when a rank's view of a local shard becomes one-dimensional.
    options = dict(sharding_strategy=strategy, device_id=device, use_orig_params=True)
    for index, block in enumerate(model.blocks):
        model.blocks[index] = FSDP(block, **options)
    return FSDP(model, **options)


def profile_update(enabled: bool):
    """Capture CPU dispatch and CUDA kernels for one optimizer update."""
    if not enabled:
        return nullcontext()
    return torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ])


def kernel_times(events: list[dict]) -> dict[str, float]:
    """Return CUDA kernel wall times in ms, counting overlap within each class once.

    NCCL names identify communication. Other kernels include optimizer and
    compute work. Copies are omitted. Offsets use Kineto's microsecond timestamps.
    """
    edges = defaultdict(lambda: [0, 0])
    for event in events:
        if event.get('cat') != 'kernel' or event.get('dur', 0) <= 0:
            continue
        kind = 0 if 'nccl' in event['name'].lower() else 1
        edges[event['ts']][kind] += 1
        edges[event['ts'] + event['dur']][kind] -= 1
    communication = compute = overlap = 0.0
    active_comm = active_compute = 0
    previous = min(edges, default=0)
    for timestamp, (comm_change, compute_change) in sorted(edges.items()):
        duration = timestamp - previous
        if active_comm:
            communication += duration
        if active_compute:
            compute += duration
        if active_comm and active_compute:
            overlap += duration
        active_comm += comm_change
        active_compute += compute_change
        previous = timestamp
    return {'nccl_kernel_ms': communication / 1000,
            'other_kernel_ms': compute / 1000, 'overlap_ms': overlap / 1000}


def export_profile(profiler, output: Path, rank: int) -> dict[str, float]:
    """Save one rank's Chrome/Perfetto trace and summarize its CUDA activity."""
    trace = output / f'trace-rank-{rank}.json'
    profiler.export_chrome_trace(str(trace))
    return kernel_times(json.loads(trace.read_text())['traceEvents'])
