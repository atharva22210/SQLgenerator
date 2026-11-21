"""Learning rates indexed by optimizer updates, never by microbatches."""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import TrainConfig


def wsd_boundaries(config: TrainConfig) -> tuple[int, int]:
    """Return inclusive warmup/stable ends; reject ambiguous rounded phases.

    Fractions refer to the entire run. For 1000 updates, warmup=100,
    stable_frac=.7 and decay_frac=.2 produce ends (100,800). Requiring the
    rounded phase lengths to sum exactly avoids silently moving a boundary.
    """
    fractions = (config.wsd_stable_frac, config.wsd_decay_frac)
    if any(not math.isfinite(value) or not 0 < value < 1 for value in fractions):
        raise ValueError("WSD fractions must be finite and between zero and one")
    stable = round(config.max_steps * config.wsd_stable_frac)
    decay = round(config.max_steps * config.wsd_decay_frac)
    if min(stable, decay) < 1 or config.warmup_steps + stable + decay != config.max_steps:
        raise ValueError("WSD warmup + rounded stable/decay steps must equal max_steps")
    return config.warmup_steps, config.warmup_steps + stable


def learning_rate(update: int, config: TrainConfig) -> float:
    """Return the LR used for a one-based optimizer update.

    Warmup reaches peak_lr on its final update. WSD holds that peak through
    the stable end, then reaches min_lr on the run's final update. Cosine's
    existing boundary convention is preserved, including zero-warmup runs.
    The returned rate is applied before the optimizer update that it labels.
    """
    if not 1 <= update <= config.max_steps:
        raise ValueError("update must be within the training run")
    if config.warmup_steps and update <= config.warmup_steps:
        return config.peak_lr * update / config.warmup_steps
    if config.lr_schedule == "wsd":
        _, stable_end = wsd_boundaries(config)
        if update <= stable_end:
            return config.peak_lr
        progress = (update - stable_end) / (config.max_steps - stable_end)
        remaining = 1 - progress if config.wsd_decay_type == "linear" else (1 + math.cos(math.pi * progress)) / 2
        return config.min_lr + (config.peak_lr - config.min_lr) * remaining
    if config.lr_schedule == "inverse_sqrt":
        anchor = max(1, config.warmup_steps)
        return max(config.min_lr, config.peak_lr * math.sqrt(anchor / update))
    start = max(1, config.warmup_steps)
    if start == config.max_steps:
        return config.peak_lr
    progress = (update - start) / (config.max_steps - start)
    return config.min_lr + (config.peak_lr - config.min_lr) * (1 + math.cos(math.pi * progress)) / 2
