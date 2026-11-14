"""Stream pre-training corpora, tokenize them, and save token IDs to a Volume.

This is deliberately a CPU Modal job. Tokenization is preprocessing, not the
GPU training step, so doing it once on CPU keeps future training jobs focused on
the model and avoids repeating the same work every epoch.

The script processes FineWeb-Edu and three text-to-SQL streams:
``NumbersStation/NSText2SQL``, ``b-mc2/sql-create-context``, and
``Clinton/Text-to-sql-v1``. The Gretel dataset is intentionally not included
in this first pre-training corpus.

Run the preprocessing job::

    uv run --locked modal run data/prepare_pretrain_data.py \\
        --profile YOUR_PROFILE --env main
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Iterator

import modal


ROOT = Path(__file__).resolve().parents[1]
# The local repository is copied into the Modal image, but the Volume is
# persistent cloud storage. Keeping these as separate concepts helps prevent
# accidentally writing large generated datasets into the container image.
VOLUME_NAME = "sqlgen-data"
VOLUME_MOUNT = Path("/vol")
TOKENIZER_PATH = VOLUME_MOUNT / "tokenizer"
OUTPUT_PATH = VOLUME_MOUNT / "pretrain_data"

ENGLISH_DATASET = "HuggingFaceFW/fineweb-edu"
ENGLISH_CONFIG = "sample-10BT"
NSTEXT2SQL_DATASET = "NumbersStation/NSText2SQL"
SQL_CONTEXT_DATASET = "b-mc2/sql-create-context"
TEXT_TO_SQL_V1_DATASET = "Clinton/Text-to-sql-v1"

# Qwen's vocabulary is larger than uint16 can represent. uint32 is still
# compact compared with Python lists or strings and is lossless for token IDs.
TOKEN_DTYPE = "uint32"
# Save progress by row, so a preempted container can resume without repeating
# hours of tokenization. A checkpoint is written only after its .bin bytes.
CHECKPOINT_EVERY_ROWS = 5_000

app = modal.App("sqlgen-pretrain-data")
# ``from_name`` refers to an existing shared Volume; it does not download the
# Volume to the laptop. The decorator below mounts it inside the remote job.
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    # Tokenization is CPU work, so this image intentionally omits the model
    # dependency group and therefore avoids installing PyTorch/CUDA libraries.
    modal.Image.debian_slim(python_version="3.13.15")
    .uv_sync(
        str(ROOT),
        extra_options="--locked --no-group model",
        uv_version="0.13.7",
        frozen=False,
        env={
            "UV_DEFAULT_INDEX": "https://pypi.org/simple",
            "UV_INDEX_URL": "https://pypi.org/simple",
        },
    )
    .add_local_file(
        ROOT / "data" / "prepare_pretrain_data.py",
        "/root/prepare_pretrain_data.py",
    )
)


def format_english(row: dict) -> str:
    """Return the text field used for ordinary language pre-training."""
    # FineWeb is not a text-to-SQL dataset. It supplies general language
    # exposure, which helps a small model learn grammar and common words before
    # the SQL-specific examples teach schema and query behavior.
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("FineWeb row has no non-empty text field")
    return text


def format_nstext2sql(row: dict) -> str:
    """Put schema/question context and SQL answer in one causal sequence."""
    # ``instruction`` already contains the schema and natural-language request;
    # ``output`` is the target SQL. The markers make the target boundary
    # explicit instead of asking the model to infer where SQL begins.
    instruction = row.get("instruction")
    output = row.get("output")
    if not isinstance(instruction, str) or not isinstance(output, str):
        raise ValueError("NSText2SQL row must contain instruction and output strings")
    return f"{instruction}\n<sql>\n{output}\n</sql>"


def format_sql_context(row: dict) -> str:
    """Normalize sql-create-context's schema, question, and answer fields."""
    # This dataset stores schema, question, and SQL in separate columns, so we
    # impose one stable text format before tokenization. Consistent formatting
    # is a modeling choice: it becomes part of what the model must learn.
    context = row.get("context")
    question = row.get("question")
    answer = row.get("answer")
    if not all(isinstance(value, str) for value in (context, question, answer)):
        raise ValueError("sql-create-context row has an unexpected schema")
    return f"Schema:\n{context}\n\nQuestion:\n{question}\n<sql>\n{answer}\n</sql>"


def format_text_to_sql_v1(row: dict) -> str | None:
    """Build an example from instruction, schema inside text, and response."""
    # The dataset's ``text`` contains a *complete* example, including its own
    # instruction and SQL answer. Appending ``response`` to all of ``text``
    # would show the answer before the model reaches our <sql> marker. Extract
    # only its schema; use the separate fields for the question and SQL target.
    instruction = row.get("instruction")
    text = row.get("text")
    answer = row.get("response")
    if not all(isinstance(value, str) for value in (instruction, text, answer)):
        raise ValueError("Text-to-sql-v1 needs instruction, text, and response strings")
    # Some source rows have an empty instruction (and an empty instruction in
    # ``text``). There is no question to learn from, so omit those rows while
    # keeping the original row-index based train/validation assignment.
    if not all(value.strip() for value in (instruction, text, answer)):
        return None

    _, instruction_marker, remainder = text.partition("### Instruction:")
    text_instruction, input_marker, remainder = remainder.partition("### Input:")
    schema, response_marker, text_answer = remainder.rpartition("### Response:")
    if not all((instruction_marker, input_marker, response_marker, schema.strip())):
        raise ValueError("Text-to-sql-v1 text has no complete instruction/input/response")
    # Check alignment: a schema copied from a different row would teach the
    # model an incorrect question-to-query mapping.
    if text_instruction.strip() != instruction.strip() or text_answer.strip() != answer.strip():
        raise ValueError("Text-to-sql-v1 text disagrees with instruction or response")

    return (
        f"Schema:\n{schema.strip()}\n\nQuestion:\n{instruction.strip()}\n"
        f"<sql>\n{answer.strip()}\n</sql>"
    )


def token_stream(
    rows: Iterable[dict],
    formatter,
    tokenizer,
) -> Iterator[int]:
    """Yield complete tokenized examples from a selected row stream."""
    for row_index, row in enumerate(rows):
        # ``encode`` applies the saved tokenizer's vocabulary and algorithm.
        # We do not manually read merges/vocabulary files here; the tokenizer
        # artifact loaded by the caller owns those rules.
        formatted = formatter(row)
        if formatted is None:
            print(f"Skipping empty {formatter.__name__} row {row_index}", flush=True)
            continue
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        # EOS separates examples. Without it, the model could see the final SQL
        # token of one example immediately followed by the next prompt and
        # learn a false continuation relationship.
        ids.append(tokenizer.eos_token_id)

        yield from ids


def write_token_stream(path: Path, tokens: Iterable[int], chunk_size: int = 65_536) -> int:
    """Write IDs in bounded chunks and return the number of written tokens."""
    import numpy as np

    # We write raw fixed-width integers rather than JSON/text. This is compact
    # and lets the future training DataLoader memory-map the files. Chunking
    # keeps the entire corpus out of RAM while the dataset is streamed.
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    chunk: list[int] = []
    with path.open("wb") as output:
        for token_id in tokens:
            chunk.append(token_id)
            if len(chunk) == chunk_size:
                np.asarray(chunk, dtype=TOKEN_DTYPE).tofile(output)
                count += len(chunk)
                chunk.clear()
        if chunk:
            np.asarray(chunk, dtype=TOKEN_DTYPE).tofile(output)
            count += len(chunk)
    return count


def load_stream(dataset: str, *, config: str | None = None):
    """Load a public Hugging Face dataset without downloading it all first."""
    from datasets import load_dataset

    # ``streaming=True`` returns an iterable dataset. We fetch examples as the
    # tokenizer asks for them instead of materializing a multi-gigabyte dataset
    # in the Modal container's ephemeral disk.
    kwargs = {"path": dataset, "split": "train", "streaming": True}
    if config is not None:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def find_tokenizer_path() -> Path:
    """Find the saved tokenizer while rejecting unrelated config files.

    The intended Volume layout is ``/vol/tokenizer/<tokenizer files>``. An
    earlier upload can accidentally create ``/vol/tokenizer/tokenizer`` when
    the remote destination ends with ``/``. Checking for tokenizer-specific
    files lets us handle either layout and avoids passing a stale outer
    ``config.json`` to ``AutoTokenizer``.
    """
    candidates = (TOKENIZER_PATH, TOKENIZER_PATH / "tokenizer")
    required_files = ("tokenizer_config.json", "tokenizer.json")
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in required_files):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Tokenizer files not found. Expected tokenizer_config.json and "
        f"tokenizer.json in one of: {searched}"
    )


def select_split(rows: Iterable[dict], split: str) -> Iterator[dict]:
    """Yield a deterministic, non-overlapping train or validation partition.

    Hugging Face streaming datasets do not give us a cheap random-access split.
    We therefore assign every tenth row to validation and the other nine rows
    to training. Both jobs see the same deterministic row numbering, so a row
    cannot appear in both files.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'val'")
    for row_index, row in enumerate(rows):
        is_validation = row_index % 10 == 0
        if (split == "val") == is_validation:
            yield row


def write_split(
    output_name: str,
    dataset: str,
    formatter,
    tokenizer,
    split: str,
    config: str | None = None,
) -> int:
    """Tokenize one partition, resuming at the last committed row checkpoint."""
    import numpy as np

    path = OUTPUT_PATH / output_name
    checkpoint_path = OUTPUT_PATH / f"{output_name}.checkpoint.json"
    identity = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "dtype": TOKEN_DTYPE,
        "vocab_size": len(tokenizer),
    }
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    if checkpoint is not None and any(checkpoint.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Checkpoint for {output_name} belongs to different input data")
    rows_done = checkpoint["rows_done"] if checkpoint else 0
    tokens_written = checkpoint["tokens_written"] if checkpoint else 0
    saved_bytes = tokens_written * np.dtype(TOKEN_DTYPE).itemsize
    if checkpoint and (not path.exists() or path.stat().st_size < saved_bytes):
        raise ValueError(f"Token file for {output_name} is shorter than its checkpoint")
    # An interruption may leave uncheckpointed bytes after the saved boundary.
    # Truncate those bytes before appending so resumed rows never appear twice.
    with path.open("r+b" if checkpoint else "wb") as output:
        output.truncate(saved_bytes)
    if checkpoint and checkpoint["complete"]:
        print(f"Already complete: {output_name} ({tokens_written} tokens)", flush=True)
        return tokens_written

    def save_checkpoint(complete: bool) -> None:
        state = {**identity, "rows_done": rows_done, "tokens_written": tokens_written,
                 "complete": complete}
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state) + "\n")
        temporary.replace(checkpoint_path)
        volume.commit()
        print(f"Checkpoint {output_name}: {rows_done} rows, {tokens_written} tokens", flush=True)

    # ``select_split`` numbers the original source before filtering. Here we
    # count selected rows, including skipped empty examples, for exact resume.
    rows = select_split(load_stream(dataset, config=config), split)
    pending: list[int] = []
    output = path.open("ab")
    try:
        for selected_index, row in enumerate(rows):
            if selected_index < rows_done:
                continue
            formatted = formatter(row)
            if formatted is None:
                print(f"Skipping empty {formatter.__name__} row {selected_index}", flush=True)
            else:
                ids = tokenizer.encode(formatted, add_special_tokens=False)
                ids.append(tokenizer.eos_token_id)
                pending.extend(ids)
                tokens_written += len(ids)
            rows_done = selected_index + 1
            if len(pending) >= 65_536:
                np.asarray(pending, dtype=TOKEN_DTYPE).tofile(output)
                pending.clear()
            if rows_done % CHECKPOINT_EVERY_ROWS == 0:
                if pending:
                    np.asarray(pending, dtype=TOKEN_DTYPE).tofile(output)
                    pending.clear()
                output.close()
                save_checkpoint(complete=False)
                output = path.open("ab")
        if pending:
            np.asarray(pending, dtype=TOKEN_DTYPE).tofile(output)
        output.close()
        save_checkpoint(complete=True)
    finally:
        if not output.closed:
            output.close()
    return tokens_written


def prepare() -> dict:
    """Run all streams inside one CPU container and commit their outputs."""
    from transformers import AutoTokenizer

    # A Volume mount can have a stale view if another process uploaded files
    # after this container started. Reload before resolving the tokenizer.
    volume.reload()
    tokenizer_path = find_tokenizer_path()

    # Loading the directory restores the exact vocabulary, merge rules,
    # tokenizer implementation, and special-token IDs used during setup.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    if len(tokenizer) > 65_536:
        print(
            f"Tokenizer vocabulary has {len(tokenizer)} entries; using {TOKEN_DTYPE} "
            "to avoid uint16 ID overflow.",
            flush=True,
        )

    # ``mkdir`` creates only the directory path inside the Volume; it does not
    # create a second Volume or affect files elsewhere in the workspace.
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}

    # Each source is loaded once per output because streaming datasets are
    # iterators. Checkpoints allow a restart to skip finished rows/files.
    results["english_val"] = write_split(
        "english_val.bin",
        ENGLISH_DATASET,
        format_english,
        tokenizer,
        "val",
        ENGLISH_CONFIG,
    )
    results["english_train"] = write_split(
        "english_train.bin",
        ENGLISH_DATASET,
        format_english,
        tokenizer,
        "train",
        ENGLISH_CONFIG,
    )
    results["nstext2sql_val"] = write_split(
        "nstext2sql_val.bin", NSTEXT2SQL_DATASET, format_nstext2sql, tokenizer, "val"
    )
    results["nstext2sql_train"] = write_split(
        "nstext2sql_train.bin", NSTEXT2SQL_DATASET, format_nstext2sql, tokenizer, "train"
    )
    results["sql_context_val"] = write_split(
        "sql_context_val.bin", SQL_CONTEXT_DATASET, format_sql_context, tokenizer, "val"
    )
    results["sql_context_train"] = write_split(
        "sql_context_train.bin", SQL_CONTEXT_DATASET, format_sql_context, tokenizer, "train"
    )
    results["text_to_sql_v1_val"] = write_split(
        "text_to_sql_v1_val.bin",
        TEXT_TO_SQL_V1_DATASET,
        format_text_to_sql_v1,
        tokenizer,
        "val",
    )
    results["text_to_sql_v1_train"] = write_split(
        "text_to_sql_v1_train.bin",
        TEXT_TO_SQL_V1_DATASET,
        format_text_to_sql_v1,
        tokenizer,
        "train",
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": len(tokenizer),
        "token_dtype": TOKEN_DTYPE,
        "split_strategy": "row_index % 10 == 0 -> validation; otherwise train",
        "eos_token_id": tokenizer.eos_token_id,
        "special_tokens": tokenizer.additional_special_tokens,
        "datasets": {
            "english": f"{ENGLISH_DATASET}/{ENGLISH_CONFIG}",
            "nstext2sql": NSTEXT2SQL_DATASET,
            "sql_context": SQL_CONTEXT_DATASET,
            "text_to_sql_v1": TEXT_TO_SQL_V1_DATASET,
        },
        "token_counts": results,
    }
    # Metadata makes the artifact auditable: later we can recover which datasets,
    # tokenizer, dtype, and row split produced these binary files.
    (OUTPUT_PATH / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Split checkpoints are committed along the way; commit metadata after all
    # eight files finish so it acts as the completion signal for the corpus.
    volume.commit()
    return metadata


@app.function(
    image=image,
    volumes={"/vol": volume},
    cpu=8,
    memory=16_384,
    # The job processes complete streaming datasets, so allow the Modal
    # Function to run for the platform maximum of 24 hours.
    timeout=86_400,
    retries=3,
    startup_timeout=300,
    scaledown_window=2,
)
def prepare_pretrain_data() -> dict:
    return prepare()


@app.local_entrypoint()
def main() -> None:
    result = prepare_pretrain_data.remote()
    print(json.dumps(result, indent=2), flush=True)
