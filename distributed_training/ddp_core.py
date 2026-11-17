"""Data assignment and loss-to-gradient mathematics, shared by A and B.

One optimizer update consists of several local forward/backward passes on
EACH rank. A rank is a process with one GPU and a full copy of the model.
This module accumulates gradients but never clears them or updates weights;
the worker owns zero_grad -> accumulated_backward -> clip -> optimizer.step.

The CPU tests exercise these same functions with Gloo instead of CUDA/NCCL.
Keeping the mathematics here makes it possible to test unequal token counts
without needing the full dataset, W&B credentials, or a cloud GPU allocation.
"""
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
    # Distinct slot ranges partition the global draw list. Choosing zero for
    # every rank is A3's deliberate bug: two GPUs repeat the same RNG inputs.
    # This partitions DRAWS, not physical files; both ranks can map the same
    # .bin files safely because sampling is read-only.
    first = (0 if duplicate else rank) * accumulation
    return list(range(first, first + accumulation))


def batch_rng(seed: int, update: int, microbatch_id: int) -> random.Random:
    """Give a draw its own RNG so rank layout cannot change its sampled data.

    Args:
        seed: Experiment-wide data seed, identical in the compared runs.
        update: One-based optimizer update, not the local microstep number.
        microbatch_id: Global draw slot returned by microbatch_ids().
    Returns:
        Independent random.Random object used for document choice/packing.

    A new RNG per draw avoids dependence on which GPU requested earlier draws.
    Model initialization uses torch's RNG separately; changing a rank's data
    assignment should not change its initial weights.

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
    bf16: CUDA autocast for production; False uses FP32 in CPU tests.

    Returns:
        loss_sum: Detached scalar sum of LOCAL target losses, still on device.
            The worker sums it across ranks before reporting global mean loss.
        counts: Device int64 tensor [global targets, non-PAD tokens, SQL tokens].
            Non-PAD input tokens and predicted targets are different: the first
            token of a document has no preceding position to predict it, and
            masked packing also ignores cross-document prediction targets.

    Caller must clear gradients before this call. This function adds to .grad;
    it intentionally does not return gradient tensors or call optimizer.step().

    Each forward returns a LOCAL mean loss L_i over n_i valid targets.
    We backpropagate L_i * n_i * world_size / global_target_count. DDP
    averages gradients by world_size, canceling that factor. The result is
    sum(all token gradients) / global_target_count, even for unequal lengths.
    There is NO additional division by accumulation steps or GPU count.
    """
    if not batches or any(batch.predicted_tokens <= 0 for batch in batches):
        raise ValueError('Every microbatch must have valid next-token targets')
    # AllReduce(SUM) combines small integer counts before any backward pass.
    # Every rank MUST call collectives in the same order; a missing participant
    # would leave the others waiting. Counts stay on the GPU for the scaling
    # below, avoiding a CPU scalar read inside each microbatch.
    local_counts = torch.tensor([
        sum(b.predicted_tokens for b in batches),
        sum(b.non_pad_tokens for b in batches),
        sum(b.sql_tokens for b in batches),
    ], dtype=torch.long, device=device)
    counts = local_counts.clone()
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    # Example: rank 0 has 100 targets and rank 1 has 900. Giving each rank's
    # mean equal weight would treat 100 tokens like 900. Multiplying the local
    # mean by its target count first reconstructs its loss SUM.
    loss_sum = torch.zeros((), device=device)
    for index, batch in enumerate(batches):
        suppress_sync = world_size > 1 and no_sync and index < len(batches) - 1
        # The final backward must be synchronized: it reduces the accumulated
        # .grad buffers from ALL local microbatches, not only the final one's
        # contribution. nullcontext is a uniform "do nothing" context for A1
        # and for B1's deliberately synchronized microbatches.
        context = model.no_sync() if suppress_sync else nullcontext()
        # BOTH forward and backward belong inside no_sync. Merely wrapping
        # backward leaves DDP's forward synchronization preparation enabled.
        with context:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                _, loss = model(**batch.model_inputs(batch.mode, device),
                                loss_chunk_size=loss_chunk_size)
            # If L_i = S_i/n_i, backward sees S_i*G/N. DDP then divides the
            # summed rank gradients by G, leaving grad(sum(S_i))/N. This also
            # handles unequal numbers of PAD/masked targets in different ranks.
            (loss * (batch.predicted_tokens * world_size / counts[0])).backward()
        # Detach the reporting value so logging cannot retain computation
        # graphs across the accumulation cycle. .grad contains the gradients;
        # loss_sum is only for displaying the resulting objective.
        loss_sum += loss.detach() * batch.predicted_tokens
    return loss_sum, counts
