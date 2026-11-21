"""Shared training configuration, corpus loading, and Modal image."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
VOLUME_NAME = "sqlgen-data"
# Pin both layouts together: raw Qwen IDs cannot index a 32K embedding.
TRAINING_BUILD = Path("/vol/bpe_tokenization")
DATA_DIR = TRAINING_BUILD / "pretrain_data"
TOKENIZER_DIR = TRAINING_BUILD / "tokenizer"

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
    .add_local_file(ROOT / "training_common.py", "/root/training_common.py")
)


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer and batching settings shared by the training experiments.

    Args:
        mode: One of pad, naive_pack, masked_pack.
        sql_fraction: Probability a sampled training document comes from SQL.
            The observed SQL *token* fraction is logged separately.
        micro_batch_size: Rows sent to the GPU per forward/backward pass.
        grad_accum_steps: Number of microbatches contributing to one update.
        seq_len: Token slots per row; it must fit model.max_seq_len.
        max_steps: Number of optimizer updates, not microbatches.
        warmup_steps: Updates with linearly rising learning rate.
        lr_schedule: Decay after warmup: cosine or inverse_sqrt. Both use
            bf16 mixed precision and the same model, data, and optimizer.
        eval_every: Updates between standalone-document validation passes.
        val_batches: Fixed number of batches per validation corpus.
        seed: Shared initial model and data sampling seed across modes.
        loss_chunk_size: Output positions projected at once to limit VRAM.
    """

    mode: str = "pad"
    sql_fraction: float = 0.5
    micro_batch_size: int = 4
    grad_accum_steps: int = 2
    seq_len: int = 4096
    max_steps: int = 500
    warmup_steps: int = 50
    lr_schedule: str = "cosine"
    eval_every: int = 50
    val_batches: int = 8
    seed: int = 42
    loss_chunk_size: int = 128
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"pad", "naive_pack", "masked_pack"}:
            raise ValueError("mode must be pad, naive_pack, or masked_pack")
        if self.lr_schedule not in {"cosine", "inverse_sqrt"}:
            raise ValueError("lr_schedule must be cosine or inverse_sqrt")
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
    """Return the LR for one-based optimizer update ``update``.

    Warmup reaches peak_lr on update ``warmup_steps``. With no warmup, update
    1 starts at peak_lr. Cosine reaches min_lr at max_steps and flattens at
    both ends; inverse_sqrt decays as 1/sqrt(update) and may not reach min_lr
    by max_steps. The floor prevents its LR from falling below min_lr later.
    """
    if not 1 <= update <= config.max_steps:
        raise ValueError("update must be within the training run")
    if update <= config.warmup_steps and config.warmup_steps:
        return config.peak_lr * update / config.warmup_steps
    if config.lr_schedule == "inverse_sqrt":
        # Anchor at the end of warmup: step 50 has peak_lr, while step 200
        # has peak_lr * sqrt(50 / 200) = half the peak. With no warmup,
        # anchor at step 1 so the first update uses peak_lr.
        anchor = max(1, config.warmup_steps)
        return max(config.min_lr, config.peak_lr * math.sqrt(anchor / update))
    # With zero warmup, step 1 itself is the cosine peak. Otherwise the
    # warmup's last step is the peak and the cosine begins on the next step.
    decay_start = max(1, config.warmup_steps)
    decay_steps = config.max_steps - decay_start
    if decay_steps == 0:
        return config.peak_lr
    progress = (update - decay_start) / decay_steps
    return config.min_lr + 0.5 * (config.peak_lr - config.min_lr) * (1 + math.cos(math.pi * progress))


def _tokenizer_path() -> Path:
    """Accept either Volume layout caused by earlier tokenizer uploads."""
    for path in (TOKENIZER_DIR, TOKENIZER_DIR / "tokenizer"):
        if (path / "tokenizer.json").is_file() and (path / "tokenizer_config.json").is_file():
            return path
    raise FileNotFoundError(f"Tokenizer files missing under {TOKENIZER_DIR}")


def _load_data_and_tokenizer(grouped_sql_dir: Path | None = None, *,
                             reload_volume: bool = True, sql_only_validation: bool = False):
    """Load the saved tokenizer and bound English/Grouped SQL corpora.

    Packing evaluates both sources. Distributed training opens SQL validation
    to measure held-out SQL loss. Both use the same audited SQL binaries.
    """
    from transformers import AutoTokenizer
    from data.dataloader import Corpus
    from data.grouped_corpus import resolve_grouped_sql
    from data.prepare_pretrain_data import tokenizer_fingerprint

    if reload_volume:
        volume.reload()
    directory = DATA_DIR
    grouped_sql_dir, grouped = resolve_grouped_sql(directory, grouped_sql_dir)
    english = json.loads((directory / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(), local_files_only=True)
    for metadata in (english, grouped):
        if (metadata["token_dtype"] != "uint32" or len(tokenizer) != metadata["tokenizer_vocab_size"]
                or tokenizer.eos_token_id != metadata["eos_token_id"] or tokenizer.pad_token_id is None):
            raise ValueError("Tokenizer and corpus identity mismatch")
    if grouped["tokenizer_sha256"] != tokenizer_fingerprint(tokenizer):
        raise ValueError("Grouped SQL tokenizer fingerprint differs from the saved tokenizer")
    _validate_corpus_files(directory, english, english_only=True)
    train = Corpus(directory, "train", tokenizer.eos_token_id, grouped_sql_dir=grouped_sql_dir)
    validation = Corpus(directory, "val", tokenizer.eos_token_id, grouped_sql_dir=grouped_sql_dir,
                        include_english=not sql_only_validation)
    validation.sql[0].eligible_indices = range(len(validation.sql[0].ends))
    return tokenizer, train, validation


def _validate_corpus_files(directory: Path, metadata: dict, *, english_only: bool = False) -> None:
    """Check durable file lengths and current preparation completion markers.

    Pipeline v2 checkpoints both splits in one source marker. Its migration
    deliberately leaves legacy split markers unchanged, so a completed v2
    source can coexist with an old incomplete split snapshot. Prefer the
    current source marker, checking its identity and both token counts; use
    legacy markers only when the source marker does not exist.
    """
    from data.dataloader import SQL_SOURCES

    for name in (("english",) if english_only else ("english", *SQL_SOURCES)):
        source_path = directory / f"{name}.checkpoint.json"
        source = json.loads(source_path.read_text()) if source_path.is_file() else None
        if source is not None:
            expected = {"pipeline_version": metadata.get("pipeline_version"),
                        "dtype": metadata["token_dtype"],
                        "vocab_size": metadata["tokenizer_vocab_size"],
                        "tokenizer_sha256": metadata.get("tokenizer_sha256"),
                        "revision": metadata.get("dataset_revisions", {}).get(name)}
            if any(source.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{source_path} disagrees with corpus metadata")
        for split in ("train", "val"):
            key = f"{name}_{split}"
            path = directory / f"{key}.bin"
            expected_tokens = metadata["token_counts"][key]
            expected_bytes = expected_tokens * 4
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"{path} does not match completed metadata ({expected_bytes} bytes)")
            legacy_path = directory / f"{key}.bin.checkpoint.json"
            progress = None
            if source is not None:
                progress = source["splits"][split]
            elif legacy_path.is_file():
                progress = json.loads(legacy_path.read_text())
            if progress is not None:
                if not progress["complete"]:
                    raise ValueError(f"{path} still has an unfinished preparation checkpoint")
                if progress["tokens_written"] != expected_tokens:
                    raise ValueError(f"{path} checkpoint token count disagrees with corpus metadata")
