"""Small, testable pieces of Parts A (data assignment) and B (synchronization)."""
from contextlib import nullcontext
import random

import torch
import torch.distributed as dist


def microbatch_ids(rank: int, world_size: int, accumulation: int, duplicate: bool = False) -> list[int]:
    """Return this rank's microbatch slots within ONE global optimizer update.

    rank: Process/GPU index, starting at zero.
    world_size: Number of cooperating processes.
    accumulation: Forward/backward passes per rank before the optimizer step.
    duplicate: A3's deliberate bug; every rank takes rank 0's slots.

    A1 (one rank, accumulation=8) sees IDs 0..7. A2 (two ranks,
    accumulation=4) assigns 0..3 to rank 0 and 4..7 to rank 1. Thus both
    configurations use the same global microbatches. A3 repeats IDs 0..3.
    IDs identify draws, not necessarily unique documents in random sampling.
    """
    if world_size < 1 or not 0 <= rank < world_size or accumulation < 1:
        raise ValueError('Invalid rank, world size, or accumulation')
    first = (0 if duplicate else rank) * accumulation
    return list(range(first, first + accumulation))


def batch_rng(seed: int, update: int, microbatch_id: int) -> random.Random:
    """Give a draw its own RNG so rank layout cannot change its sampled data.

    update is one-based. No process-local RNG history or Python hash() enters
    this seed, so A1/A2 and B1/B2 replay the same draws independently.
    """
    return random.Random(f'{seed}:{update}:{microbatch_id}')


def accumulated_backward(model, batches, device, *, world_size: int, no_sync: bool,
                         loss_chunk_size: int, bf16: bool = True):
    """Accumulate a GLOBAL token-mean gradient; leave clipping/step to caller.

    model: Plain model for world_size=1, otherwise a DDP wrapper.
    batches: CPU Batch objects for this update, with valid target counts.
    device: This process's torch.device.
    no_sync: Skip DDP gradient communication except on the last microbatch.
    loss_chunk_size: Positions projected into vocabulary logits at once.
    bf16: CUDA autocast for production; False allows exact CPU correctness tests.

    Each forward returns a LOCAL mean loss L_i over n_i valid targets.
    We backpropagate L_i * n_i * world_size / global_target_count. DDP
    averages gradients by world_size, canceling that factor. The result is
    sum(all token gradients) / global_target_count, even for unequal lengths.
    There is NO additional division by accumulation steps or GPU count.
    """
    if not batches or any(batch.predicted_tokens <= 0 for batch in batches):
        raise ValueError('Every microbatch must have valid next-token targets')
    local_counts = torch.tensor([
        sum(b.predicted_tokens for b in batches),
        sum(b.non_pad_tokens for b in batches),
        sum(b.sql_tokens for b in batches),
    ], dtype=torch.long, device=device)
    counts = local_counts.clone()
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    loss_sum = torch.zeros((), device=device)
    for index, batch in enumerate(batches):
        suppress_sync = world_size > 1 and no_sync and index < len(batches) - 1
        context = model.no_sync() if suppress_sync else nullcontext()
        # BOTH forward and backward belong inside no_sync. Merely wrapping
        # backward leaves DDP's forward synchronization preparation enabled.
        with context:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                _, loss = model(**batch.model_inputs(batch.mode, device),
                                loss_chunk_size=loss_chunk_size)
            (loss * (batch.predicted_tokens * world_size / counts[0])).backward()
        loss_sum += loss.detach() * batch.predicted_tokens
    return loss_sum, counts
