"""Decode a few saved pretraining documents from the Modal Volume.

Run::

    uv run --locked modal run --env main --profile sanika-05 \
        data/inspect_pretrain_data.py

This inspects validation data by default, which is the data behind the W&B
validation losses. It reads existing files only; it does not tokenize again,
train a model, or modify the Volume.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from data.grouped_corpus import resolve_grouped_sql


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/vol/qwen_tokenization/pretrain_data")
TOKENIZER_DIR = Path("/vol/qwen_tokenization/tokenizer")
SOURCES = ("english", "sql")

app = modal.App("sqlgen-inspect-pretrain-data")
volume = modal.Volume.from_name("sqlgen-data")
# The saved tokens need only NumPy and the exact tokenizer library version.
# No model weights or GPU are needed to turn token IDs back into readable text.
image = (
    modal.Image.debian_slim(python_version="3.13.15")
    .pip_install("numpy==2.5.4", "transformers==4.57.6")
    .add_local_file(ROOT / "data" / "inspect_pretrain_data.py", "/root/inspect_pretrain_data.py")
    .add_local_file(ROOT / "data" / "grouped_corpus.py", "/root/data/grouped_corpus.py")
)


def _tokenizer_path() -> Path:
    """Find the tokenizer in either layout used by earlier Volume uploads."""
    for directory in (TOKENIZER_DIR, TOKENIZER_DIR / "tokenizer"):
        if (directory / "tokenizer.json").is_file() and (directory / "tokenizer_config.json").is_file():
            return directory
    raise FileNotFoundError(f"No saved tokenizer found under {TOKENIZER_DIR}")


def _preview(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep long examples readable while showing both prompt and SQL answer."""
    if len(text) <= max_chars:
        return text, False
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - max_chars
    return text[:head] + f"\n... [{omitted} characters omitted] ...\n" + text[-tail:], True


def _question_and_sql(source: str, decoded: str) -> tuple[str | None, str | None]:
    """Extract the source question and reference SQL from a decoded row.

    This is an inspection aid, not a model prediction. SQL sources were saved
    as prompt + ``<sql>`` + answer + ``</sql>`` for SynSQL. Repository files
    contain SQL text without the question and answer wrapper.
    If an exceptionally long row was only partly decoded, a missing field is
    shown as null rather than guessing its contents.
    """
    if source == "english":
        return None, None
    prompt, marker, response = decoded.rpartition("<sql>")
    if not marker:
        return None, None
    answer, closing_marker, _ = response.partition("</sql>")
    if not closing_marker:
        return None, None

    _, question_marker, question_text = prompt.rpartition("Question:\n")
    question = question_text.strip() if question_marker else None
    return question, answer.strip()


@app.function(image=image, volumes={"/vol": volume}, cpu=1, memory=2_048, timeout=180)
def inspect(split: str, samples_per_source: int, max_chars: int) -> dict:
    """Return decoded examples from each saved ``<source>_<split>.bin`` file.

    Args:
        split: ``val`` for held-out examples, or ``train`` for training examples.
        samples_per_source: Number of complete EOS-delimited rows to inspect.
        max_chars: Most text characters displayed per row. Long rows show
            their beginning and end, with the omitted length marked.

    The binary format is little-endian uint32 IDs with one EOS after each
    source row. We memory-map each file and find the first few EOS boundaries;
    this avoids loading the full corpus into RAM just to inspect a few rows.
    """
    import numpy as np
    from transformers import AutoTokenizer

    if split not in {"train", "val"} or not 1 <= samples_per_source <= 10 or max_chars < 300:
        raise ValueError("Use train/val, 1-10 samples per source, and at least 300 characters")

    volume.reload()
    metadata_path = DATA_DIR / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Completed corpus metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("token_dtype") != "uint32":
        raise ValueError("This reader expects uint32 token IDs")
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(), local_files_only=True)
    eos_id = tokenizer.eos_token_id
    if eos_id != metadata["eos_token_id"] or len(tokenizer) != metadata["tokenizer_vocab_size"]:
        raise ValueError("The saved tokenizer does not match the token files")

    grouped_sql_dir, grouped = resolve_grouped_sql(DATA_DIR)
    if eos_id != grouped["eos_token_id"] or len(tokenizer) != grouped["tokenizer_vocab_size"]:
        raise ValueError("The tokenizer does not match the Grouped SQL files")

    examples = {}
    for source in SOURCES:
        key = f"{source}_{split}"
        directory, source_metadata = (DATA_DIR, metadata) if source == "english" else (grouped_sql_dir, grouped)
        path = directory / f"{key}.bin"
        expected_bytes = source_metadata["token_counts"][key] * 4
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"{path} is missing or differs from completion metadata")
        tokens = np.memmap(path, mode="r", dtype="<u4")
        if len(tokens) == 0 or int(tokens[-1]) != eos_id:
            raise ValueError(f"{path} does not end with EOS")

        # np.flatnonzero finds EOS positions in a small window at a time. An
        # EOS closes one document: [start, end] is exactly one prepared row.
        ends: list[int] = []
        for offset in range(0, len(tokens), 65_536):
            chunk = tokens[offset : offset + 65_536]
            ends.extend((np.flatnonzero(chunk == eos_id) + offset).tolist())
            if len(ends) >= samples_per_source:
                break
        if len(ends) < samples_per_source:
            raise ValueError(f"{path} has only {len(ends)} complete rows")

        rows = []
        start = 0
        for row_number, end in enumerate(ends[:samples_per_source], start=1):
            count = end + 1 - start
            # A very large source row can exceed the model's context. Decode
            # its first and last 5,000 IDs so inspection stays bounded; say
            # explicitly when the middle was not decoded.
            if count <= 10_000:
                ids = tokens[start : end + 1].astype(np.int64).tolist()
                decoded = tokenizer.decode(ids, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
                middle_not_decoded = False
            else:
                head_ids = tokens[start : start + 5_000].astype(np.int64).tolist()
                tail_ids = tokens[end - 4_999 : end + 1].astype(np.int64).tolist()
                decoded = (
                    tokenizer.decode(head_ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
                    + f"\n... [{count - 10_000} token IDs not decoded] ...\n"
                    + tokenizer.decode(tail_ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
                )
                middle_not_decoded = True
            preview, text_clipped = _preview(decoded, max_chars)
            question, sql_query = _question_and_sql(source, decoded)
            rows.append({
                "row_number_in_file": row_number,
                "token_count_including_eos": count,
                "eos_id": eos_id,
                "middle_not_decoded": middle_not_decoded,
                "display_clipped": text_clipped,
                # These fields stay complete even if decoded_text is shortened
                # for display. English rows have no question/SQL pair.
                "question": question,
                "sql_query": sql_query,
                "decoded_text": preview,
            })
            start = end + 1
        examples[source] = rows

    result = {"split": split, "tokenizer": str(_tokenizer_path()), "grouped_sql_dir": str(grouped_sql_dir), "examples": examples}
    # Print inside the Modal function so the decoded examples appear in its
    # Logs tab, including when the local terminal is no longer available.
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


@app.local_entrypoint()
def main(split: str = "val", samples_per_source: int = 2, max_chars: int = 2_000) -> None:
    """Run a read-only CPU inspection and print the decoded examples."""
    inspect.remote(split, samples_per_source, max_chars)
