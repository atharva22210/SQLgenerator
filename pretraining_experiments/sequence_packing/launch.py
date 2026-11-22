"""Run one readable, single-A10 pretraining experiment on saved token files.

Run the same command three times with ``--mode pad``, ``naive_pack``, and
``masked_pack``. The Modal Volume supplies data and receives checkpoints;
W&B records the metrics. Training starts only when you run the command.

Example (small correctness run before a longer comparison)::

    uv run --locked modal run --detach --env main --profile sanika-05 \
        -m pretraining_experiments.sequence_packing.launch --mode pad --max-steps 2 --warmup-steps 1 \
        --wandb-entity YOUR_WANDB_ENTITY

Defaults target one A10: 4 x 4096 tokens per microbatch, two accumulation
steps, and 128-position loss chunks. With 32,768 vocabulary entries the tied
embedding accounts for 16.78 million of the model's 34.48 million parameters.
Measure peak memory and throughput in a short correctness run first.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import modal

from training.config import TrainConfig, image, volume, wandb_secret
from training.trainer import _checkpoint_path, run_training

WANDB_PROJECT = "sqlgen-packing"
app = modal.App("sqlgen-packing")


@app.function(
    image=image, volumes={"/vol": volume}, secrets=[wandb_secret],
    gpu="A10", cpu=4, memory=16_384, timeout=86_400,
    startup_timeout=600, scaledown_window=2,
)
def train_on_gpu(config: dict, wandb_entity: str, wandb_project: str, run_name: str) -> dict:
    """Remote boundary: runs on one A10, while the local entrypoint only submits."""
    return run_training(TrainConfig(**config), wandb_entity, wandb_project, run_name)


@app.local_entrypoint()
def main(
    mode: str = "pad", sql_fraction: float = 0.5,
    micro_batch_size: int = 4, grad_accum_steps: int = 2, seq_len: int = 4096,
    max_steps: int = 500, warmup_steps: int = 50, lr_schedule: str = "cosine",
    eval_every: int = 50,
    val_batches: int = 8, seed: int = 42, loss_chunk_size: int = 128,
    wandb_entity: str = "", wandb_project: str = WANDB_PROJECT,
    run_name: str = "", wait: bool = False, initializer_range: float = 0.02,
    wsd_stable_frac: float = 0.7, wsd_decay_frac: float = 0.2,
    wsd_decay_type: str = "linear", validation_seed: int = -1,
) -> None:
    """Submit and exit by default; use modal run --detach to keep the App alive.

    ``--wait`` optionally retrieves the result for short debugging runs. It
    does not change the independent lifetime of the spawned invocation.
    ``--initializer-range`` controls the normal standard deviation for a
    fresh run (default 0.02). A different-scale checkpoint cannot resume.
    WSD options configure its phase fractions/tail; --validation-seed fixes
    evaluation draws independently of training seed (-1 retains legacy behavior).
    """
    config = TrainConfig(
        mode=mode, sql_fraction=sql_fraction, micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps, seq_len=seq_len, max_steps=max_steps,
        warmup_steps=warmup_steps, lr_schedule=lr_schedule,
        eval_every=eval_every, val_batches=val_batches,
        seed=seed, loss_chunk_size=loss_chunk_size, initializer_range=initializer_range,
        wsd_stable_frac=wsd_stable_frac, wsd_decay_frac=wsd_decay_frac, wsd_decay_type=wsd_decay_type,
        validation_seed=None if validation_seed == -1 else validation_seed,
    )
    if not wandb_entity:
        raise ValueError("Supply --wandb-entity for W&B logging")
    name = run_name or f"{mode}-{lr_schedule}-prefix-sql{sql_fraction:g}-seed{seed}"
    if not run_name and initializer_range != 0.02:
        name += f"-init-std{initializer_range:g}"
    if not run_name and lr_schedule == "wsd":
        name += f"-tail-{wsd_decay_type}"
    _checkpoint_path(name)  # Validate before requesting a GPU.
    call = train_on_gpu.spawn(asdict(config), wandb_entity, wandb_project, name)
    print(json.dumps({"run": name, "call_id": call.object_id,
                      "checkpoint": str(_checkpoint_path(name))}), flush=True)
    if wait:
        print(json.dumps(call.get(), indent=2))
    else:
        print("Submitted. Use modal run --detach so the App survives launcher exit.", flush=True)
