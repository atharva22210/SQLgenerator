"""Explicit A1/A2/A3 and B1/B2 settings, independent of GPU allocation."""
from dataclasses import asdict
from train_pretrain import TrainConfig


def experiment_plan(part: str, max_steps: int = 0, micro_batch_size: int = 0,
                    seq_len: int = 2048, sql_fraction: float = 0.5) -> list[dict]:
    """Build a controlled suite; zero steps/batch selects the curriculum default.

    part='a': 300 updates, width 512, A1=8x8x1 and A2/A3=8x4x2.
    part='ab': A1, A2, B1, B2 in that order, omitting the deliberate A3 bug.
    part='b': 100 updates, width 768, B1/B2=4x8x2; only no_sync differs.
    seq_len: Slots per row; use 64 for a cheap initial GPU check.
    sql_fraction: Probability of drawing a SQL document when filling a row.
    The larger preset is named by dimensions, not a misleading parameter count:
    the saved Qwen tokenizer makes the embedding alone larger than 100M.
    """
    if part == 'ab':
        a = experiment_plan('a', max_steps, micro_batch_size, seq_len, sql_fraction)
        b = experiment_plan('b', max_steps, micro_batch_size, seq_len, sql_fraction)
        return a[:2] + b
    if part not in {'a', 'b'} or max_steps < 0 or micro_batch_size < 0:
        raise ValueError('part must be a/b/ab; steps and microbatch must be nonnegative')
    steps = max_steps or (300 if part == 'a' else 100)
    batch = micro_batch_size or (8 if part == 'a' else 4)
    cases = [('A1-single-reference', 1, 8, False, True),
             ('A2-ddp-distinct-draws', 2, 4, False, True),
             ('A3-ddp-duplicate-draws', 2, 4, True, True)] if part == 'a' else [
             ('B1-sync-every-microbatch', 2, 8, False, False),
             ('B2-sync-last-microbatch', 2, 8, False, True)]
    model = {} if part == 'a' else dict(n_layers=12, d_model=768, n_heads=12,
                                       n_kv_heads=4, ffn_dim=2048)
    return [dict(label=label, world_size=world, duplicate=duplicate, no_sync=no_sync,
                 model=model, benchmark_warmup=min(10, max(0, steps - 1)),
                 train=asdict(TrainConfig(mode='masked_pack', micro_batch_size=batch,
                                         grad_accum_steps=accumulation, max_steps=steps,
                                         warmup_steps=min(50, steps), seq_len=seq_len,
                                         sql_fraction=sql_fraction)))
            for label, world, accumulation, duplicate, no_sync in cases]
