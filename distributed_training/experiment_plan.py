"""Describe experiments before allocating hardware or launching processes.

A: change the number of GPUs while preserving the global batch, then show a
   deliberately duplicated-data case. B: change only gradient sync timing.
Returning plain dictionaries makes the plan readable in launcher logs and
serializable as JSON for torchrun workers and as W&B run configuration.
"""
from dataclasses import asdict
from training_common import TrainConfig


def experiment_plan(part: str, max_steps: int = 0, micro_batch_size: int = 0,
                    seq_len: int = 2048, sql_fraction: float = 0.5, *, eval_every: int = 25,
                    val_batches: int = 128, checkpoint_blocks: int = 12,
                    accumulation: int = 2) -> list[dict]:
    """Build a controlled suite; zero steps/batch selects the curriculum default.

    part='a': 300 updates, width 512, A1=8x8x1 and A2/A3=8x4x2.
    part='ab': A1, A2, B1, B2 in that order, omitting the deliberate A3 bug.
    part='b': 100 updates, width 768, B1/B2=4x8x2; only no_sync differs.
    seq_len: Slots per row; use 64 for a cheap initial GPU check.
    sql_fraction: Probability of drawing a SQL document when filling a row.
    eval_every: Optimizer updates between samples of val_batches documents.
    The larger preset is named by dimensions, not a misleading parameter count:
    the saved Qwen tokenizer makes the embedding alone larger than 100M.
    """
    # The combined suite is explicitly ordered, not parallel: one case must
    # exit successfully before the next begins. Slice away A3 because the user
    # requested A1 -> A2 -> B1 -> B2 for this suite.
    if part == 'ab':
        a = experiment_plan('a', max_steps, micro_batch_size, seq_len, sql_fraction,
                            eval_every=eval_every, val_batches=val_batches)
        b = experiment_plan('b', max_steps, micro_batch_size, seq_len, sql_fraction,
                            eval_every=eval_every, val_batches=val_batches)
        return a[:2] + b
    if part == 'c' and min(max_steps, micro_batch_size) >= 0:
        from distributed_training.fsdp_checkpointing import part_c_plan
        return part_c_plan(max_steps, micro_batch_size, seq_len, sql_fraction,
                           eval_every, val_batches, checkpoint_blocks, accumulation)
    if part not in {'a', 'b'} or max_steps < 0 or micro_batch_size < 0:
        raise ValueError('part must be a/b/ab/c; steps and microbatch must be nonnegative')
    # Zero is a CLI sentinel for "use the experiment default". An explicit
    # --max-steps affects both parts in a combined run; leaving it zero keeps
    # A at 300 updates and B at 100. This is an update count, not an epoch count.
    steps = max_steps or (300 if part == 'a' else 100)
    batch = micro_batch_size or (8 if part == 'a' else 4)
    # Tuple fields: (human label, GPU/process count, accumulation steps,
    #                deliberately duplicate data?, defer gradient sync?).
    # A1: 8*8*1 = 64 sequences/update; A2: 8*4*2 = 64 sequences/update.
    # B1/B2 both use 4*8*2 = 64; only the final tuple flag changes.
    cases = [('A1-single-reference', 1, 8, False, True),
             ('A2-ddp-distinct-draws', 2, 4, False, True),
             ('A3-ddp-duplicate-draws', 2, 4, True, True)] if part == 'a' else [
             ('B1-sync-every-microbatch', 2, 8, False, False),
             ('B2-sync-last-microbatch', 2, 8, False, True)]
    # An empty model dict inherits TransformerConfig's smaller defaults.
    # Naming the larger model by dimensions avoids assuming "135M": the
    # tokenizer vocabulary can dominate its actual trainable parameter count.
    model = {} if part == 'a' else dict(n_layers=12, d_model=768, n_heads=12,
                                       n_kv_heads=4, ffn_dim=2048)
    # benchmark_warmup removes cold initial updates from throughput summaries.
    # It is DISTINCT from learning-rate warmup_steps. Tiny checks retain at
    # least one timed update but are not reliable steady-state benchmarks.
    # asdict converts the frozen TrainConfig into a JSON-compatible snapshot.
    return [dict(label=label, world_size=world, duplicate=duplicate, no_sync=no_sync,
                 model=model, benchmark_warmup=min(10, max(0, steps - 1)),
                 train=asdict(TrainConfig(mode='masked_pack', micro_batch_size=batch,
                                         grad_accum_steps=accumulation, max_steps=steps,
                                         warmup_steps=min(50, steps), seq_len=seq_len,
                                         sql_fraction=sql_fraction, eval_every=eval_every,
                                         val_batches=val_batches)))
            for label, world, accumulation, duplicate, no_sync in cases]
