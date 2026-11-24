"""Atomic snapshots preserve the model, optimizer, sample stream, and history."""
from dataclasses import asdict
import os
from pathlib import Path
import random

from .config import TrainConfig, checkpoint_config_matches


def save_checkpoint(path: Path, model, optimizer, config: TrainConfig, update: int,
                    rng: random.Random, cumulative_tokens: int, wandb_id: str,
                    grouped_sql_dir: Path, *, history=(), dataset=None, update_direction=None,
                    probe_recipe=None, branch_origin=None, sample_hashes=()) -> None:
    """Save a completed update; the caller commits its Volume after all writes.

    History ends at this same update. On resume, discard any later buffered
    metric rows and restore ALL RNG states after constructing/loading the model.
    Otherwise initialization would consume random numbers and shift future data.
    CPU snapshots use cuda_rng=None so this invariant can be tested locally.
    """
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({"config": asdict(config), "model_config": asdict(model.config),
                "update": update, "grouped_sql_dir": str(grouped_sql_dir),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "cumulative_tokens": cumulative_tokens, "wandb_id": wandb_id,
                "history": list(history), "dataset_identity": dataset,
                "update_direction": update_direction, "probe_recipe": probe_recipe or {},
                "branch_origin": branch_origin, "sample_hashes": list(sample_hashes)}, temporary)
    os.replace(temporary, path)


def restore_checkpoint(state: dict, model, optimizer, config: TrainConfig, rng: random.Random) -> None:
    """Restore in place, requiring unchanged optimizer/schedule/batch settings."""
    import torch
    if not checkpoint_config_matches(state['config'], config):
        raise ValueError("Checkpoint config differs from this run")
    if state.get('model_config', asdict(model.config)) != asdict(model.config):
        raise ValueError("Checkpoint model architecture differs from this run")
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    rng.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if state.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())


def restore_branch(state: dict, model, optimizer, config: TrainConfig, rng: random.Random) -> None:
    """Explicitly fork one saved SGD state; only branch schedule/length may change.

    Restore using the source configuration first, so architecture, weights,
    optimizer history and RNG replay obey the ordinary checkpoint invariant.
    The caller resets only the branch's counters/history and tracking ID.
    Clipping, decay, batching, seed and all data/loss controls remain identical.
    """
    source = TrainConfig(**state['config'])
    allowed = {'peak_lr', 'min_lr', 'max_steps', 'warmup_steps', 'eval_every'}
    changed = {key for key, value in asdict(config).items() if value != asdict(source)[key]}
    if source.optimizer != 'sgd' or not changed <= allowed:
        raise ValueError('Branch may change only plain-SGD schedule, length and evaluation interval')
    if config.peak_lr != config.min_lr or config.warmup_steps != 0 or config.lr_schedule != 'cosine':
        raise ValueError('Branch must use a constant LR with no warmup')
    restore_checkpoint(state, model, optimizer, source, rng)
