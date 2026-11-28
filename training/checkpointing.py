"""Atomic snapshots preserve the model, optimizer, sample stream, and history."""
from dataclasses import asdict
import os
from pathlib import Path
import random

from .config import TrainConfig, checkpoint_config_matches


def save_checkpoint(path: Path, model, optimizer, config: TrainConfig, update: int,
                    rng: random.Random, cumulative_tokens: int, wandb_id: str,
                    grouped_sql_dir: Path, *, history=(), dataset=None, ema_model=None) -> None:
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
                **({"ema_model": ema_model.state_dict()} if ema_model is not None else {}),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "cumulative_tokens": cumulative_tokens, "wandb_id": wandb_id,
                "history": list(history), "dataset_identity": dataset}, temporary)
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
