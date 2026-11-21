"""Stream pre-training corpora, tokenize them, and save token IDs to a Volume.

This is deliberately a CPU Modal job. Tokenization is preprocessing, not the
GPU training step, so doing it once on CPU keeps future training jobs focused on
the model and avoids repeating the same work every epoch.

This script prepares FineWeb-Edu. Grouped SQL uses the separate
``data.prepare_grouped_sql`` job so its database and repository holdouts are
audited before any SQL binaries are published.

Run the preprocessing job::

    uv run --locked modal run --detach data/prepare_pretrain_data.py \\
        --profile YOUR_PROFILE --env main --submit-only
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
# The local repository is copied into the Modal image, but the Volume is
# persistent cloud storage. Keeping these as separate concepts helps prevent
# accidentally writing large generated datasets into the container image.
VOLUME_NAME = "sqlgen-data"
VOLUME_MOUNT = Path("/vol")
QWEN_BUILD = VOLUME_MOUNT / "qwen_tokenization"
TOKENIZER_PATH = QWEN_BUILD / "tokenizer"
OUTPUT_PATH = QWEN_BUILD / "pretrain_data"

ENGLISH_DATASET = "HuggingFaceFW/fineweb-edu"
ENGLISH_CONFIG = "sample-10BT"

# Qwen's vocabulary is larger than uint16 can represent. uint32 is still
# compact compared with Python lists or strings and is lossless for token IDs.
TOKEN_DTYPE = "uint32"
# These are source rows (before the 90/10 split). A single checkpoint covers
# both output files and the source cursor, so they always resume together.
CHECKPOINT_EVERY_ROWS = 50_000
CHECKPOINT_SECONDS = 60
BATCH_ROWS = 256
BATCH_CHARACTERS = 1_000_000
PROGRESS_SECONDS = 10
CPU_CORES = 8
PIPELINE_VERSION = 2

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
    # Set these before the Rust tokenizer initializes its global Rayon pool.
    # Batches, unlike individual encode calls, expose document parallelism.
    .env({"TOKENIZERS_PARALLELISM": "true", "RAYON_NUM_THREADS": str(CPU_CORES)})
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


def load_stream(dataset: str, *, config: str | None = None, revision: str):
    """Load a public Hugging Face dataset without downloading it all first."""
    from datasets import load_dataset

    # ``streaming=True`` returns an iterable dataset. We fetch examples as the
    # tokenizer asks for them instead of materializing a multi-gigabyte dataset
    # in the Modal container's ephemeral disk.
    kwargs = {"path": dataset, "split": "train", "streaming": True, "revision": revision}
    if dataset == ENGLISH_DATASET:
        # FineWeb's other nine columns are unused. Parquet column projection
        # avoids fetching/decoding URLs, scores, dates, and other metadata.
        kwargs["columns"] = ["text"]
    if config is not None:
        kwargs["name"] = config
    stream = load_dataset(**kwargs)
    if dataset == ENGLISH_DATASET:
        # The locked datasets version can treat each Parquet row group as a
        # shard, retaining file/row-group order. Cursor restoration then skips
        # completed row groups rather than rereading a whole Parquet file.
        stream = stream.reshard()
    return stream


def find_tokenizer_path() -> Path:
    """Find the saved tokenizer while rejecting unrelated config files.

    The intended Volume layout is ``/vol/qwen_tokenization/tokenizer/<tokenizer files>``. An
    earlier upload can accidentally create ``/vol/qwen_tokenization/tokenizer/tokenizer`` when
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


def tokenizer_fingerprint(tokenizer) -> str:
    """Detect vocabulary/merge changes even when vocabulary size is unchanged."""
    # Include wrapper behavior and special IDs as well as Rust's vocabulary.
    identity = {"backend": json.loads(tokenizer.backend_tokenizer.to_str()),
                "special_tokens": tokenizer.special_tokens_map,
                "eos_token_id": tokenizer.eos_token_id,
                "class": type(tokenizer).__name__}
    # Padding/truncation are transient backend settings; our batch calls always
    # disable both. Exclude them so merely encoding a batch cannot change ID.
    identity["backend"]["padding"] = None
    identity["backend"]["truncation"] = None
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def read_batch(iterator, stream, source_rows: int, formatter, floors: dict) -> dict:
    """Read at most one bounded batch and snapshot its *own* source position.

    Only the producer thread touches the iterator. The consumer may tokenize
    this batch while the producer reads the next one. Never checkpoint the
    live stream state from the consumer: prefetch may have advanced it already.
    """
    texts, splits = [], []
    characters = 0
    complete = False
    for _ in range(BATCH_ROWS):
        try:
            row = next(iterator)
        except StopIteration:
            complete = True
            break
        is_val = source_rows % 10 == 0
        split = "val" if is_val else "train"
        # Global row 10 is validation row 1; global row 11 is training row 9.
        selected_index = source_rows // 10 if is_val else source_rows - (source_rows + 9) // 10
        source_rows += 1
        if floors[split]["complete"] or selected_index < floors[split]["rows_done"]:
            continue
        formatted = formatter(row)
        if formatted is not None:
            texts.append(formatted)
            splits.append(split)
            characters += len(formatted)
        # This is a soft bound: one unusually large document stays intact.
        # Truncating/splitting text here would change the token stream.
        if characters >= BATCH_CHARACTERS:
            break
    return {"texts": texts, "splits": splits, "source_rows": source_rows,
            "stream_state": deepcopy(stream.state_dict()), "complete": complete}


def legacy_progress(name: str, dataset: str, config, tokenizer) -> dict:
    """Import old row-only checkpoints without discarding their token prefix."""
    result = {}
    for split in ("train", "val"):
        path = OUTPUT_PATH / f"{name}_{split}.bin.checkpoint.json"
        state = json.loads(path.read_text()) if path.exists() else None
        identity = {"dataset": dataset, "config": config, "split": split,
                    "dtype": TOKEN_DTYPE, "vocab_size": len(tokenizer)}
        if state and any(state.get(k) != v for k, v in identity.items()):
            raise ValueError(f"Legacy checkpoint {path.name} belongs to different input data")
        result[split] = ({k: state[k] for k in ("rows_done", "tokens_written", "complete")}
                         if state else {"rows_done": 0, "tokens_written": 0, "complete": False})
    return result


def write_corpus(name: str, dataset: str, formatter, tokenizer, config=None) -> dict:
    """Read once, tokenize batches in parallel, and atomically checkpoint both splits.

    A checkpoint represents one consumed source prefix and two durable output
    prefixes. Prefetched rows are excluded until their tokens have been written.
    A retry truncates speculative bytes in *both* files before restoring that
    source position. Original row numbering is retained across retries.
    """
    import datasets
    from huggingface_hub import HfApi
    import numpy as np

    checkpoint_path = OUTPUT_PATH / f"{name}.checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    identity = {"pipeline_version": PIPELINE_VERSION, "dataset": dataset, "config": config,
                "dtype": TOKEN_DTYPE, "vocab_size": len(tokenizer),
                "stream_layout": "row_groups" if dataset == ENGLISH_DATASET else "files",
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                "formatter": formatter.__name__, "datasets_version": datasets.__version__}
    if checkpoint and any(checkpoint.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Checkpoint for {name} belongs to different input data or code")
    progress = deepcopy(checkpoint["splits"]) if checkpoint else legacy_progress(
        name, dataset, config, tokenizer)
    paths = {split: OUTPUT_PATH / f"{name}_{split}.bin" for split in progress}
    # Validate every prefix *before* mutating either output. Refuse orphaned
    # output rather than silently replacing data whose checkpoint is missing.
    for split, path in paths.items():
        saved_bytes = progress[split]["tokens_written"] * np.dtype(TOKEN_DTYPE).itemsize
        if saved_bytes and (not path.exists() or path.stat().st_size < saved_bytes):
            raise ValueError(f"Token file for {path.name} is shorter than its checkpoint")
        if checkpoint is None and not (OUTPUT_PATH / f"{path.name}.checkpoint.json").exists():
            if path.exists() and path.stat().st_size:
                raise ValueError(f"Token file {path.name} has no checkpoint; refusing to overwrite")
    for split, path in paths.items():
        with path.open("r+b" if path.exists() else "wb") as output:
            output.truncate(progress[split]["tokens_written"] * np.dtype(TOKEN_DTYPE).itemsize)
    if all(state["complete"] for state in progress.values()):
        print(f"Already complete: {name}", flush=True)
        return {split: state["tokens_written"] for split, state in progress.items()}

    # Pin the source revision for cursor stability. Legacy files did not record
    # one; migration assumes their original source ordering has not changed.
    revision = checkpoint["revision"] if checkpoint else HfApi().dataset_info(dataset).sha
    stream = load_stream(dataset, config=config, revision=revision)
    source_rows = checkpoint["source_rows"] if checkpoint else 0
    if checkpoint:
        if checkpoint["stream_state"] is not None:
            stream.load_state_dict(checkpoint["stream_state"])
        print(f"Resuming {name}: source row {source_rows}, revision {revision}; "
              "restoring shard cursor (current shard may be reread)", flush=True)
    else:
        print(f"Starting {name}: revision {revision}, legacy prefixes={progress}", flush=True)
        # Establish an empty durable prefix before writing any new bytes. Even
        # a preemption before the first periodic checkpoint must be resumable.
        state = {**identity, "revision": revision, "source_rows": 0,
                 "stream_state": None, "splits": progress}
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state) + "\n")
        temporary.replace(checkpoint_path)
        volume.commit()
    iterator = iter(stream)
    # Floors remain fixed for this invocation. They matter only during legacy
    # migration, when validation may be ahead of the newly created train file.
    floors = deepcopy(progress)
    started = last_checkpoint = last_report = time.monotonic()
    cpu_started = time.process_time()
    committed_rows = source_rows
    initial_tokens = sum(state["tokens_written"] for state in progress.values())
    tokenize_seconds = 0.0
    outputs = {split: path.open("ab") for split, path in paths.items()}
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="source-prefetch") as producer:
            future = producer.submit(read_batch, iterator, stream, source_rows, formatter, floors)
            while True:
                try:
                    batch = future.result(timeout=PROGRESS_SECONDS)
                except FutureTimeout:
                    if future.done():
                        # A source TimeoutError is a real failure, whereas an
                        # unfinished Future only means it is time to heartbeat.
                        batch = future.result()
                    else:
                        print(f"Waiting for {name} source; last processed row={source_rows}", flush=True)
                        continue
                # One future bounds read-ahead to one additional batch. Its
                # cursor can advance while this batch's frozen state is saved.
                if not batch["complete"]:
                    future = producer.submit(read_batch, iterator, stream, batch["source_rows"],
                                             formatter, floors)
                token_start = time.monotonic()
                if batch["texts"]:
                    encoded = tokenizer(batch["texts"], add_special_tokens=False, padding=False,
                                        truncation=False, return_attention_mask=False,
                                        return_token_type_ids=False)["input_ids"]
                    # [batch, variable_length] -> two flat uint32 token streams.
                    # Rust preserves batch order; append exactly one EOS per row.
                    flat = {"train": [], "val": []}
                    for split, ids in zip(batch["splits"], encoded, strict=True):
                        flat[split].extend(ids)
                        flat[split].append(tokenizer.eos_token_id)
                    for split, ids in flat.items():
                        if ids:
                            np.asarray(ids, dtype=TOKEN_DTYPE).tofile(outputs[split])
                            progress[split]["tokens_written"] += len(ids)
                tokenize_seconds += time.monotonic() - token_start
                source_rows = batch["source_rows"]
                val_rows = (source_rows + 9) // 10
                for split, count in (("val", val_rows), ("train", source_rows - val_rows)):
                    progress[split]["rows_done"] = max(progress[split]["rows_done"], count)
                now = time.monotonic()
                tokens_added = sum(state["tokens_written"] for state in progress.values()) - initial_tokens
                if now - last_report >= PROGRESS_SECONDS or batch["complete"]:
                    print(f"Progress {name}: source_rows={source_rows}, new_tokens={tokens_added}, "
                          f"tokens/s={tokens_added / max(now - started, 0.001):.0f}, "
                          f"avg_cpu_cores={(time.process_time() - cpu_started) / max(now - started, 0.001):.2f}, "
                          f"tokenize+write_s={tokenize_seconds:.1f}, elapsed_s={now - started:.1f}",
                          flush=True)
                    last_report = now
                if (batch["complete"] or source_rows - committed_rows >= CHECKPOINT_EVERY_ROWS
                        or now - last_checkpoint >= CHECKPOINT_SECONDS):
                    for output in outputs.values():
                        output.flush()
                        os.fsync(output.fileno())
                        output.close()
                    if batch["complete"]:
                        for split, count in (("val", val_rows), ("train", source_rows - val_rows)):
                            if progress[split]["rows_done"] != count:
                                raise ValueError("Legacy checkpoint exceeds source length")
                        for state in progress.values():
                            state["complete"] = True
                    state = {**identity, "revision": revision, "source_rows": source_rows,
                             "stream_state": batch["stream_state"], "splits": progress}
                    temporary = checkpoint_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state) + "\n")
                    temporary.replace(checkpoint_path)
                    commit_start = time.monotonic()
                    volume.commit()
                    print(f"Checkpoint {name}: source_rows={source_rows}, splits={progress}, "
                          f"commit_s={time.monotonic() - commit_start:.1f}", flush=True)
                    committed_rows = source_rows
                    last_checkpoint = time.monotonic()
                    if not batch["complete"]:
                        outputs = {split: path.open("ab") for split, path in paths.items()}
                if batch["complete"]:
                    break
    finally:
        for output in outputs.values():
            if not output.closed:
                output.close()
    return {split: state["tokens_written"] for split, state in progress.items()}


def prepare() -> dict:
    """Build the English train and validation files used alongside Grouped SQL."""
    from transformers import AutoTokenizer

    # A Volume mount can have a stale view if another process uploaded files
    # after this container started. Reload before resolving the tokenizer.
    volume.reload()
    tokenizer_path = find_tokenizer_path()

    # Loading the directory restores the exact vocabulary, merge rules,
    # tokenizer implementation, and special-token IDs used during setup.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Parallel batched preprocessing requires a fast Rust tokenizer")
    print(f"Tokenizer threads={os.environ.get('RAYON_NUM_THREADS', 'library default')}, "
          f"parallelism={os.environ.get('TOKENIZERS_PARALLELISM', 'library default')}, "
          f"batch_rows={BATCH_ROWS}, batch_characters={BATCH_CHARACTERS}", flush=True)
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
    counts = write_corpus("english", ENGLISH_DATASET, format_english, tokenizer, ENGLISH_CONFIG)
    checkpoint = OUTPUT_PATH / "english.checkpoint.json"
    # Older completed English files can predate revision tracking.
    revision = json.loads(checkpoint.read_text())["revision"] if checkpoint.exists() else None

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": len(tokenizer),
        "token_dtype": TOKEN_DTYPE,
        "split_strategy": "row_index % 10 == 0 -> validation; otherwise train",
        "eos_token_id": tokenizer.eos_token_id,
        "special_tokens": tokenizer.additional_special_tokens,
        "datasets": {"english": f"{ENGLISH_DATASET}/{ENGLISH_CONFIG}"},
        "dataset_revisions": {"english": revision},
        "token_counts": {f"english_{split}": count for split, count in counts.items()},
    }
    # Metadata makes the artifact auditable: later we can recover which datasets,
    # tokenizer, dtype, and row split produced these binary files.
    (OUTPUT_PATH / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Publish completion metadata after both English files finish.
    volume.commit()
    return metadata


@app.function(
    image=image,
    volumes={"/vol": volume},
    cpu=CPU_CORES,
    memory=16_384,
    # The job processes complete streaming datasets, so allow the Modal
    # Function to run for the platform maximum of 24 hours.
    timeout=86_400,
    retries=3,
    startup_timeout=300,
    scaledown_window=2,
    # Shared output filenames permit one writer. Detached invocations must
    # also be serialized by the team: this limit is per App, not per Volume.
    max_containers=1,
)
def prepare_pretrain_data() -> dict:
    return prepare()


@app.local_entrypoint()
def main(submit_only: bool = False) -> None:
    # App detachment and invocation lifetime are separate. A synchronous
    # .remote() call can be cancelled with its caller; .spawn() submits an
    # independent job. With --detach --submit-only the launcher can exit
    # normally, without keeping a local wait/connection alive for hours.
    call = prepare_pretrain_data.spawn()
    print(f"Submitted preprocessing call: {call.object_id}", flush=True)
    if submit_only:
        print("Use --detach with --submit-only so the App survives launcher exit.", flush=True)
        return
    result = call.get()
    print(json.dumps(result, indent=2), flush=True)
