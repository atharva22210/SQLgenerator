"""Run one readable, single-L4 pretraining experiment on saved token files.

Run the same command three times with ``--mode pad``, ``naive_pack``, and
``masked_pack``. The Modal Volume supplies data and receives checkpoints;
W&B records the metrics. Training starts only when you run the command.

Example (small correctness run before a longer comparison)::

    uv run --locked modal run --env main --profile sanika-05 \
        train_pretrain.py --mode pad --max-steps 2 --warmup-steps 1 \
        --wandb-entity YOUR_WANDB_ENTITY

The default batch is deliberately smaller than the photographed 8 x 2048
recipe. With 151,670 vocabulary entries the tied embedding alone is about
78 million parameters, so this is not a 35M-parameter model. Measure L4
memory and throughput before increasing microbatch size or sequence length.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import modal


ROOT = Path(__file__).resolve().parent
VOLUME_NAME = "sqlgen-data"
DATA_DIR = Path("/vol/pretrain_data")
CHECKPOINT_DIR = Path("/vol/checkpoints/day5")
TOKENIZER_DIR = Path("/vol/tokenizer")

app = modal.App("sqlgen-day5-pretrain")
volume = modal.Volume.from_name(VOLUME_NAME)
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])
image = (
    modal.Image.debian_slim(python_version="3.13.15")
    .uv_sync(
        str(ROOT), extra_options="--locked", uv_version="0.13.7", frozen=False,
        env={"UV_DEFAULT_INDEX": "https://pypi.org/simple", "UV_INDEX_URL": "https://pypi.org/simple"},
    )
    # uv_sync installs dependencies; the project itself is not installed as a
    # wheel. Copy the teaching modules separately for remote imports.
    .add_local_dir(ROOT / "model", "/root/model")
    .add_local_dir(ROOT / "data", "/root/data")
)


@dataclass(frozen=True)
class TrainConfig:
    """Values that should stay fixed while comparing the three batch modes.

    Args:
        mode: One of pad, naive_pack, masked_pack.
        sql_fraction: Probability a sampled training document comes from SQL.
            The observed SQL *token* fraction is logged separately.
        micro_batch_size: Rows sent to the GPU per forward/backward pass.
        grad_accum_steps: Number of microbatches contributing to one update.
        seq_len: Token slots per row; it must fit model.max_seq_len.
        max_steps: Number of optimizer updates, not microbatches.
        warmup_steps: Updates with linearly rising learning rate.
        eval_every: Updates between standalone-document validation passes.
        val_batches: Fixed number of batches per validation corpus.
        seed: Shared initial model and data sampling seed across modes.
        loss_chunk_size: Output positions projected at once to limit VRAM.
    """

    mode: str = "pad"
    sql_fraction: float = 0.5
    micro_batch_size: int = 1
    grad_accum_steps: int = 4
    seq_len: int = 512
    max_steps: int = 500
    warmup_steps: int = 50
    eval_every: int = 50
    val_batches: int = 8
    seed: int = 42
    loss_chunk_size: int = 64
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"pad", "naive_pack", "masked_pack"}:
            raise ValueError("mode must be pad, naive_pack, or masked_pack")
        if not 0 <= self.sql_fraction <= 1:
            raise ValueError("sql_fraction must be between 0 and 1")
        if min(self.micro_batch_size, self.grad_accum_steps, self.seq_len,
               self.max_steps, self.eval_every, self.val_batches,
               self.loss_chunk_size) <= 0 or self.seq_len < 2:
            raise ValueError("Batch sizes, lengths, and step counts must be positive")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError("warmup_steps must be between 0 and max_steps")
        if not 0 < self.min_lr <= self.peak_lr or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Learning rates, weight decay, or gradient clip are invalid")


def learning_rate(update: int, config: TrainConfig) -> float:
    """Warm up linearly, then follow half a cosine to ``min_lr``.

    ``update`` is one-based: for a 50-step warmup, update 50 uses peak_lr;
    update 500 uses min_lr. The cosine slope is zero at both ends, so the
    decay begins and ends gently.
    """
    if not 1 <= update <= config.max_steps:
        raise ValueError("update must be within the training run")
    if update <= config.warmup_steps and config.warmup_steps:
        return config.peak_lr * update / config.warmup_steps
    decay_steps = config.max_steps - config.warmup_steps
    if decay_steps == 0:
        return config.peak_lr
    progress = (update - config.warmup_steps) / decay_steps
    return config.min_lr + 0.5 * (config.peak_lr - config.min_lr) * (1 + math.cos(math.pi * progress))


def _tokenizer_path() -> Path:
    """Accept either Volume layout caused by earlier tokenizer uploads."""
    for path in (TOKENIZER_DIR, TOKENIZER_DIR / "tokenizer"):
        if (path / "tokenizer.json").is_file() and (path / "tokenizer_config.json").is_file():
            return path
    raise FileNotFoundError(f"Tokenizer files missing under {TOKENIZER_DIR}")


def _load_data_and_tokenizer():
    """Require the completion metadata before trusting raw binary files."""
    from transformers import AutoTokenizer
    from data.dataloader import Corpus, SQL_SOURCES

    volume.reload()
    metadata_path = DATA_DIR / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No completed corpus metadata at {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("token_dtype") != "uint32":
        raise ValueError("Expected uint32 token files from the preparation job")
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(), local_files_only=True)
    if len(tokenizer) != metadata["tokenizer_vocab_size"]:
        raise ValueError("Saved tokenizer vocabulary disagrees with corpus metadata")
    if tokenizer.eos_token_id != metadata["eos_token_id"] or tokenizer.pad_token_id is None:
        raise ValueError("Saved tokenizer needs matching EOS and a PAD token")
    # EOS at the end of a file alone cannot prove a long interrupted run
    # finished. Match every file's length to the completion metadata too.
    for name in ("english", *SQL_SOURCES):
        for split in ("train", "val"):
            key = f"{name}_{split}"
            path = DATA_DIR / f"{key}.bin"
            expected_bytes = metadata["token_counts"][key] * 4
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"{path} does not match completed metadata ({expected_bytes} bytes)")
            checkpoint_path = DATA_DIR / f"{key}.bin.checkpoint.json"
            if checkpoint_path.is_file() and not json.loads(checkpoint_path.read_text())["complete"]:
                raise ValueError(f"{path} still has an unfinished preparation checkpoint")
    train = Corpus(DATA_DIR, "train", tokenizer.eos_token_id)
    validation = Corpus(DATA_DIR, "val", tokenizer.eos_token_id)
    return tokenizer, train, validation


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
        total_loss, total_targets = 0.0, 0
        with torch.no_grad():
            for _ in range(config.val_batches):
                batch = make_batch(
                    corpus, mode="pad", batch_size=1, seq_len=config.seq_len,
                    sql_fraction=config.sql_fraction, pad_id=pad_id, rng=rng,
                    validation_kind=kind,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(**batch.model_inputs("pad", device), loss_chunk_size=config.loss_chunk_size)
                total_loss += float(loss) * batch.predicted_tokens
                total_targets += batch.predicted_tokens
        results[f"val/{kind}_loss"] = total_loss / total_targets
    model.train()
    return results


def _checkpoint_path(run_name: str) -> Path:
    """Keep each mode/ratio's optimizer and model state in its own directory."""
    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in run_name):
        raise ValueError("run_name may contain only letters, digits, hyphens, dots, underscores")
    return CHECKPOINT_DIR / run_name / "latest.pt"


def _save_checkpoint(path: Path, model, optimizer, config: TrainConfig, update: int,
                     rng: random.Random, cumulative_tokens: int, wandb_id: str) -> None:
    """Write then rename and commit, so resume never reads a partial snapshot."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "config": asdict(config), "update": update,
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
    if not wandb_entity:
        raise ValueError("Pass --wandb-entity to record a reproducible run")
    tokenizer, train, validation = _load_data_and_tokenizer()
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
        lr=config.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
    path = _checkpoint_path(run_name)
    start_update, cumulative_tokens, wandb_id = 0, 0, wandb.util.generate_id()
    if path.is_file():
        # Resume only the same experiment. A different mix or layout needs a
        # new run name, otherwise its optimizer history would contaminate it.
        state = torch.load(path, map_location=device, weights_only=False)
        if state["config"] != asdict(config):
            raise ValueError(f"Checkpoint config differs from this run: {path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"])
        start_update = state["update"]
        cumulative_tokens = state["cumulative_tokens"]
        wandb_id = state["wandb_id"]

    print(json.dumps({"run": run_name, "parameter_count": parameter_count,
                      "gpu": torch.cuda.get_device_name(), "resume_after_update": start_update,
                      "data_dir": str(DATA_DIR), "checkpoint": str(path)}), flush=True)
    with wandb.init(
        entity=wandb_entity, project=wandb_project, name=run_name,
        id=wandb_id, resume="allow", config={**asdict(config), "parameter_count": parameter_count,
                                              "gpu": torch.cuda.get_device_name()},
        settings=wandb.Settings(init_timeout=120),
    ) as run:
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
            weighted_loss = 0.0
            for batch in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(**batch.model_inputs(config.mode, device),
                                    loss_chunk_size=config.loss_chunk_size)
                # A padded microbatch has fewer real targets. Weight its mean
                # loss by its target count, so the update is a mean over all
                # predicted tokens across all accumulated microbatches.
                weight = batch.predicted_tokens / target_count
                (loss * weight).backward()
                weighted_loss += float(loss.detach()) * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            real_tokens = sum(batch.non_pad_tokens for batch in batches)
            sql_tokens = sum(batch.sql_tokens for batch in batches)
            cumulative_tokens += real_tokens
            slots = config.micro_batch_size * config.seq_len * config.grad_accum_steps
            metrics = {
                "train/loss": weighted_loss,
                "train/grad_norm": float(grad_norm),
                "train/lr": lr,
                "data/observed_sql_token_fraction": sql_tokens / real_tokens,
                "perf/raw_tokens_per_sec": slots / elapsed,
                "perf/effective_non_pad_tokens_per_sec": real_tokens / elapsed,
                "perf/padding_waste_pct": 100 * (slots - real_tokens) / slots,
                "perf/cumulative_non_pad_tokens": cumulative_tokens,
            }
            if update % config.eval_every == 0 or update == config.max_steps:
                metrics.update(_evaluate(model, validation, config, tokenizer.pad_token_id, device))
                _save_checkpoint(path, model, optimizer, config, update,
                                 rng, cumulative_tokens, wandb_id)
            run.log(metrics, step=update)
            print(json.dumps({"update": update, **metrics}), flush=True)
        return {"run": run_name, "updates": config.max_steps,
                "parameter_count": parameter_count, "checkpoint": str(path), "wandb_url": run.url}


@app.function(
    image=image, volumes={"/vol": volume}, secrets=[wandb_secret],
    gpu="L4", cpu=4, memory=16_384, timeout=86_400,
    startup_timeout=600, scaledown_window=2,
)
def train_on_gpu(config: dict, wandb_entity: str, wandb_project: str, run_name: str) -> dict:
    """Remote boundary: runs on one L4, while the local entrypoint only submits."""
    return _run(TrainConfig(**config), wandb_entity, wandb_project, run_name)


@app.local_entrypoint()
def main(
    mode: str = "pad", sql_fraction: float = 0.5,
    micro_batch_size: int = 1, grad_accum_steps: int = 4, seq_len: int = 512,
    max_steps: int = 500, warmup_steps: int = 50, eval_every: int = 50,
    val_batches: int = 8, seed: int = 42, loss_chunk_size: int = 64,
    wandb_entity: str = "", wandb_project: str = "sqlgen-day5",
    run_name: str = "",
) -> None:
    """CLI arguments map directly to TrainConfig; change one for an experiment."""
    config = TrainConfig(
        mode=mode, sql_fraction=sql_fraction, micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps, seq_len=seq_len, max_steps=max_steps,
        warmup_steps=warmup_steps, eval_every=eval_every, val_batches=val_batches,
        seed=seed, loss_chunk_size=loss_chunk_size,
    )
    if not wandb_entity:
        raise ValueError("Supply --wandb-entity for W&B logging")
    name = run_name or f"{mode}-sql{sql_fraction:g}-seed{seed}"
    _checkpoint_path(name)  # Validate before requesting a GPU.
    print(json.dumps(train_on_gpu.remote(asdict(config), wandb_entity, wandb_project, name), indent=2))
