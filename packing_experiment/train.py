"""Run one readable, single-A10 pretraining experiment on saved token files.

Run the same command three times with ``--mode pad``, ``naive_pack``, and
``masked_pack``. The Modal Volume supplies data and receives checkpoints;
W&B records the metrics. Training starts only when you run the command.

Example (small correctness run before a longer comparison)::

    uv run --locked modal run --detach --env main --profile sanika-05 \
        -m packing_experiment.train --mode pad --max-steps 2 --warmup-steps 1 \
        --wandb-entity YOUR_WANDB_ENTITY

Defaults target one A10: 4 x 4096 tokens per microbatch, two accumulation
steps, and 128-position loss chunks. With 32,768 vocabulary entries the tied
embedding accounts for 16.78 million of the model's 34.48 million parameters.
Measure peak memory and throughput in a short correctness run first.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
import uuid

import modal


from training_common import DATA_DIR, TrainConfig, _load_data_and_tokenizer, image, learning_rate, volume, wandb_secret


CHECKPOINT_DIR = Path("/vol/checkpoints/packing")
WANDB_PROJECT = "sqlgen-packing"
MODE_TITLES = {"pad": "Padding", "naive_pack": "Unmasked packing", "masked_pack": "Masked packing"}
app = modal.App("sqlgen-packing")


def _evaluate(model, corpus, config: TrainConfig, pad_id: int, device) -> dict[str, float]:
    """Use the same standalone validation documents at every evaluation.

    Every mode is evaluated as one document per row. This holds the benchmark
    fixed while training layout changes; packed validation would confound the
    comparison with another document-boundary choice.
    """
    import torch
    from data.dataloader import make_batch

    model.eval()
    results = {}
    for kind, offset in (("english", 10_000), ("sql", 20_000)):
        rng = random.Random(config.seed + offset)
        total_loss, total_targets = torch.zeros((), device=device), 0
        with torch.no_grad():
            for _ in range(config.val_batches):
                batch = make_batch(
                    corpus, mode="pad", batch_size=1, seq_len=config.seq_len,
                    sql_fraction=config.sql_fraction, pad_id=pad_id, rng=rng,
                    validation_kind=kind,
                )
                # Use the same bf16 compute path for both LR schedules. Model
                # weights and optimizer state remain float32 for stable updates.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(**batch.model_inputs("pad", device), loss_chunk_size=config.loss_chunk_size)
                total_loss += loss.detach() * batch.predicted_tokens
                total_targets += batch.predicted_tokens
        # One scalar read per corpus, not per validation microbatch.
        results[f"val/{kind}_loss"] = float(total_loss / total_targets)
    model.train()
    return results


def _checkpoint_path(run_name: str) -> Path:
    """Keep each mode/ratio's optimizer and model state in its own directory."""
    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in run_name):
        raise ValueError("run_name may contain only letters, digits, hyphens, dots, underscores")
    return CHECKPOINT_DIR / run_name / "latest.pt"


def _save_checkpoint(path: Path, model, optimizer, config: TrainConfig, update: int,
                     rng: random.Random, cumulative_tokens: int, wandb_id: str,
                     grouped_sql_dir: Path) -> None:
    """Write then rename and commit, so resume never reads a partial snapshot."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "config": asdict(config), "update": update, "grouped_sql_dir": str(grouped_sql_dir),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "cumulative_tokens": cumulative_tokens, "wandb_id": wandb_id,
    }, temporary)
    os.replace(temporary, path)
    volume.commit()


def _run(config: TrainConfig, wandb_entity: str, wandb_project: str, run_name: str) -> dict:
    """Execute updates; Modal provides the GPU, Volume, and W&B credential."""
    import torch
    import wandb
    from data.dataloader import make_batch
    from model.transformer_35m import Transformer, TransformerConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run needs a CUDA GPU with bf16 support")
    # bf16 has a wide exponent range, so this mixed-precision path needs no
    # GradScaler. Autocast changes eligible compute ops, not parameter dtype.
    if not wandb_entity:
        raise ValueError("Pass --wandb-entity to record a reproducible run")
    from data.grouped_corpus import resolve_grouped_sql

    volume.reload()
    path = _checkpoint_path(run_name)
    state = torch.load(path, map_location="cpu", weights_only=False) if path.is_file() else None
    if state is not None:
        if state["config"] != asdict(config):
            raise ValueError(f"Checkpoint config differs from this run: {path}")
        if not state.get("grouped_sql_dir"):
            raise ValueError(f"Checkpoint predates Grouped SQL. Use a new run name: {path}")
    saved_dir = Path(state["grouped_sql_dir"]) if state is not None else None
    grouped_sql_dir, _ = resolve_grouped_sql(DATA_DIR, saved_dir)
    tokenizer, train, validation = _load_data_and_tokenizer(grouped_sql_dir, reload_volume=False)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    rng = random.Random(config.seed)
    device = torch.device("cuda")
    model = Transformer(TransformerConfig(vocab_size=len(tokenizer), max_seq_len=config.seq_len)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # AdamW decouples weight decay from its adaptive gradient update. Biases
    # and one-dimensional normalization scales are normally left undecayed.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.peak_lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    # Own the stable ID used by W&B and our checkpoint without depending on
    # W&B's internal ID helper, which is not exposed by the pinned release.
    start_update, cumulative_tokens, wandb_id = 0, 0, uuid.uuid4().hex
    if state is not None:
        # Restore the same corpus and stream position the checkpoint recorded.
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Loading restores param-group flags too. Older snapshots used the
        # automatic implementation; keep this run's fused CUDA execution.
        for group in optimizer.param_groups:
            group["fused"] = True
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"])
        start_update = state["update"]
        cumulative_tokens = state["cumulative_tokens"]
        wandb_id = state["wandb_id"]

    print(json.dumps({"run": run_name, "parameter_count": parameter_count,
                      "gpu": torch.cuda.get_device_name(), "resume_after_update": start_update,
                      "compute_dtype": "bfloat16", "parameter_dtype": "float32",
                      "data_dir": str(DATA_DIR), "grouped_sql_dir": str(grouped_sql_dir), "checkpoint": str(path)}), flush=True)
    with wandb.init(
        entity=wandb_entity, project=wandb_project,
        name=f"Packing · {MODE_TITLES[config.mode]} · {config.lr_schedule} · seed {config.seed}",
        group=f"Packing · Grouped SQL {grouped_sql_dir.name[:8]} · {config.seq_len:,} tokens",
        job_type="training",
        id=wandb_id, resume="allow", config={**asdict(config), "parameter_count": parameter_count,
                                              "grouped_sql_dir": str(grouped_sql_dir),
                                              "corpus_id": grouped_sql_dir.name, "checkpoint_name": run_name,
                                              "gpu": torch.cuda.get_device_name(),
                                              "compute_dtype": "bfloat16", "parameter_dtype": "float32"},
        settings=wandb.Settings(init_timeout=120),
    ) as run:
        run.define_metric("optimizer_update")
        for namespace in ("train", "val", "data", "perf"):
            run.define_metric(f"{namespace}/*", step_metric="optimizer_update")
        torch.cuda.reset_peak_memory_stats()
        for update in range(start_update + 1, config.max_steps + 1):
            started = time.perf_counter()
            lr = learning_rate(update, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            batches = [make_batch(
                train, mode=config.mode, batch_size=config.micro_batch_size,
                seq_len=config.seq_len, sql_fraction=config.sql_fraction,
                pad_id=tokenizer.pad_token_id, rng=rng,
            ) for _ in range(config.grad_accum_steps)]
            target_count = sum(batch.predicted_tokens for batch in batches)
            weighted_loss = torch.zeros((), device=device)
            for batch in batches:
                # Training and validation both use bf16 autocast, whether the
                # LR came from cosine or inverse-square-root decay.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(**batch.model_inputs(config.mode, device),
                                    loss_chunk_size=config.loss_chunk_size)
                # A padded microbatch has fewer real targets. Weight its mean
                # loss by its target count, so the update is a mean over all
                # predicted tokens across all accumulated microbatches.
                weight = batch.predicted_tokens / target_count
                (loss * weight).backward()
                weighted_loss += loss.detach() * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            # Metrics are the only scalar read in an update. Copy both after
            # optimizer.step so timing includes completed GPU work, without a
            # separate synchronize() or any reads between microbatches.
            logged_loss, logged_grad_norm = torch.stack((weighted_loss, grad_norm)).tolist()
            elapsed = time.perf_counter() - started
            real_tokens = sum(batch.non_pad_tokens for batch in batches)
            sql_tokens = sum(batch.sql_tokens for batch in batches)
            cumulative_tokens += real_tokens
            slots = config.micro_batch_size * config.seq_len * config.grad_accum_steps
            metrics = {
                "train/loss": logged_loss,
                "train/grad_norm": logged_grad_norm,
                "train/lr": lr,
                "data/observed_sql_token_fraction": sql_tokens / real_tokens,
                "perf/raw_tokens_per_sec": slots / elapsed,
                "perf/effective_non_pad_tokens_per_sec": real_tokens / elapsed,
                "perf/padding_waste_pct": 100 * (slots - real_tokens) / slots,
                "perf/cumulative_non_pad_tokens": cumulative_tokens,
                "perf/peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "perf/peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            if update % config.eval_every == 0 or update == config.max_steps:
                metrics.update(_evaluate(model, validation, config, tokenizer.pad_token_id, device))
                _save_checkpoint(path, model, optimizer, config, update,
                                 rng, cumulative_tokens, wandb_id, grouped_sql_dir)
            run.log({"optimizer_update": update, **metrics}, step=update)
            print(json.dumps({"update": update, **metrics}), flush=True)
        return {"run": run_name, "updates": config.max_steps,
                "parameter_count": parameter_count, "checkpoint": str(path), "wandb_url": run.url,
                "grouped_sql_dir": str(grouped_sql_dir)}


@app.function(
    image=image, volumes={"/vol": volume}, secrets=[wandb_secret],
    gpu="A10", cpu=4, memory=16_384, timeout=86_400,
    startup_timeout=600, scaledown_window=2,
)
def train_on_gpu(config: dict, wandb_entity: str, wandb_project: str, run_name: str) -> dict:
    """Remote boundary: runs on one A10, while the local entrypoint only submits."""
    return _run(TrainConfig(**config), wandb_entity, wandb_project, run_name)


@app.local_entrypoint()
def main(
    mode: str = "pad", sql_fraction: float = 0.5,
    micro_batch_size: int = 4, grad_accum_steps: int = 2, seq_len: int = 4096,
    max_steps: int = 500, warmup_steps: int = 50, lr_schedule: str = "cosine",
    eval_every: int = 50,
    val_batches: int = 8, seed: int = 42, loss_chunk_size: int = 128,
    wandb_entity: str = "", wandb_project: str = WANDB_PROJECT,
    run_name: str = "", wait: bool = False,
) -> None:
    """Submit and exit by default; use modal run --detach to keep the App alive.

    ``--wait`` optionally retrieves the result for short debugging runs. It
    does not change the independent lifetime of the spawned invocation.
    """
    config = TrainConfig(
        mode=mode, sql_fraction=sql_fraction, micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps, seq_len=seq_len, max_steps=max_steps,
        warmup_steps=warmup_steps, lr_schedule=lr_schedule,
        eval_every=eval_every, val_batches=val_batches,
        seed=seed, loss_chunk_size=loss_chunk_size,
    )
    if not wandb_entity:
        raise ValueError("Supply --wandb-entity for W&B logging")
    name = run_name or f"{mode}-{lr_schedule}-prefix-sql{sql_fraction:g}-seed{seed}"
    _checkpoint_path(name)  # Validate before requesting a GPU.
    call = train_on_gpu.spawn(asdict(config), wandb_entity, wandb_project, name)
    print(json.dumps({"run": name, "call_id": call.object_id,
                      "checkpoint": str(_checkpoint_path(name))}), flush=True)
    if wait:
        print(json.dumps(call.get(), indent=2))
    else:
        print("Submitted. Use modal run --detach so the App survives launcher exit.", flush=True)
