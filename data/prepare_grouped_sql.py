"""Prepare a versioned Grouped SQL corpus on the existing pretraining volume."""

import json
from pathlib import Path
import sqlite3
import time
from tempfile import TemporaryDirectory
from uuid import uuid4

import modal

from data.grouped_corpus import GROUPED_SQL_FORMAT_VERSION
from data.grouped_sql import split_and_audit, stage_rows
from data.prepare_pretrain_data import (
    BATCH_CHARACTERS, BATCH_ROWS, OUTPUT_PATH, ROOT, find_tokenizer_path, image,
    tokenizer_fingerprint, volume,
)


SYN_SQL = "iNeil77/SynSQL-2.5M"
REPOSITORY_SQL = "bigcode/starcoderdata"
app = modal.App("sqlgen-grouped-sql")
image = (image.add_local_file(ROOT / "data" / "__init__.py", "/root/data/__init__.py")
         .add_local_file(ROOT / "data" / "grouped_corpus.py", "/root/data/grouped_corpus.py")
         .add_local_file(ROOT / "data" / "grouped_sql.py", "/root/data/grouped_sql.py")
         .add_local_file(ROOT / "data" / "prepare_pretrain_data.py", "/root/data/prepare_pretrain_data.py"))


def split_examples(db, split):
    """Return up to five audited documents per source from distinct groups."""
    examples = []
    for kind in ("db_id", "repo_name"):
        rows = db.execute("""
            SELECT d.group_id, d.text FROM documents d JOIN (
                SELECT MIN(id) AS id FROM documents
                WHERE split = ? AND kind = ? GROUP BY group_id ORDER BY id LIMIT 5
            ) selected ON selected.id = d.id ORDER BY d.id
        """, (split, kind))
        for group, text in rows:
            # Preserve the question/answer at the end of long SynSQL prompts.
            limit = 6000
            preview = text if len(text) <= limit else text[:3000] + "\n[… excerpt …]\n" + text[-3000:]
            examples.append(dict(kind=kind, group=group, text=preview, truncated=len(text) > limit))
    return examples


def write_split(db, split, path, tokenizer):
    """Write retained SQLite documents as little-endian uint32 IDs plus EOS."""
    import numpy as np

    def write_batch(output, texts):
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False,
                            return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        flat = []
        for ids in encoded:
            if not ids or tokenizer.eos_token_id in ids:
                raise ValueError("Document must contain text and no embedded EOS token")
            flat.extend(ids)
            flat.append(tokenizer.eos_token_id)
        np.asarray(flat, dtype="<u4").tofile(output)
        return len(flat)

    total = characters = 0
    started = last_report = time.monotonic()
    texts = []
    with path.open("wb") as output:
        for text, in db.execute("SELECT text FROM documents WHERE split = ? ORDER BY id", (split,)):
            if texts and (len(texts) >= BATCH_ROWS or characters + len(text) > BATCH_CHARACTERS):
                total += write_batch(output, texts)
                now = time.monotonic()
                if now - last_report >= 60:
                    print(f"Tokenized {split}: {total:,} tokens, {total / (now - started):,.0f} tokens/s",
                          flush=True)
                    last_report = now
                texts = []
                characters = 0
            texts.append(text)
            characters += len(text)
        if texts:
            total += write_batch(output, texts)
    print(f"Finished tokenizing {split}: {total:,} tokens", flush=True)
    return total


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=16_384,
              timeout=86_400, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def prepare(seed: int, corpus_id: str, fingerprint_workers: int = 4) -> dict:
    """Stage pinned Parquet streams, audit before writing, and commit the overlay."""
    import resource

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    volume.reload()
    tokenizer = AutoTokenizer.from_pretrained(find_tokenizer_path(), local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Expected the saved fast tokenizer with EOS and PAD")
    api = HfApi(token=True)
    revisions = {source: api.dataset_info(source).sha for source in (SYN_SQL, REPOSITORY_SQL)}
    synsql = load_dataset(SYN_SQL, split="train", streaming=True, revision=revisions[SYN_SQL],
                          columns=["db_id", "schema", "question", "sql", "external_knowledge"])
    repository_sql = load_dataset(REPOSITORY_SQL, data_dir="sql", split="train", streaming=True,
                                  revision=revisions[REPOSITORY_SQL], token=True,
                                  columns=["max_stars_repo_name", "content"])
    output = OUTPUT_PATH.parent / "builds" / corpus_id / "pretrain_data"
    with TemporaryDirectory() as scratch:
        db = sqlite3.connect(Path(scratch) / "split.sqlite")
        try:
            started = time.monotonic()
            cpu_start = time.process_time()
            children_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            staging = stage_rows(db, synsql, repository_sql, eos_marker=tokenizer.eos_token,
                                 workers=fingerprint_workers)
            children_end = resource.getrusage(resource.RUSAGE_CHILDREN)
            child_seconds = ((children_end.ru_utime - children_start.ru_utime)
                             + (children_end.ru_stime - children_start.ru_stime))
            wall_seconds = time.monotonic() - started
            staging_perf = {
                "fingerprint_workers": fingerprint_workers,
                "seconds": wall_seconds,
                "avg_cpu_cores": (time.process_time() - cpu_start + child_seconds) / wall_seconds,
            }
            print(json.dumps({"staging_perf": staging_perf}), flush=True)
            audit = split_and_audit(db, seed)
            print(json.dumps({"staging": staging, "audit": audit}), flush=True)
            output.mkdir(parents=True, exist_ok=False)
            examples = {split: split_examples(db, split) for split in ("train", "val")}
            (output / "sql_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
            # Publish the audited preview early. Readers still require
            # metadata.json, which appears only after both bins are complete.
            volume.commit()
            print(f"Audited examples ready at {output / 'sql_examples.json'}", flush=True)
            counts = {f"sql_{split}": write_split(db, split, output / f"sql_{split}.bin", tokenizer)
                      for split in ("train", "val")}
        finally:
            db.close()
    metadata = dict(format_version=GROUPED_SQL_FORMAT_VERSION, seed=seed, corpus_id=corpus_id,
                    dataset_revisions=revisions, token_dtype="uint32",
                    tokenizer_vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                    tokenizer_sha256=tokenizer_fingerprint(tokenizer), token_counts=counts,
                    staging=staging, staging_perf=staging_perf, audit=audit)
    temporary = output / "metadata.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(output / "sql_metadata.json")
    volume.commit()
    return {"directory": str(output), **metadata}


@app.local_entrypoint()
def main(seed: int = 42, submit_only: bool = False, fingerprint_workers: int = 4):
    """Submit the CPU-only build and print its durable Modal call ID."""
    if not 1 <= fingerprint_workers <= 7:
        raise ValueError("Use 1–7 fingerprint workers, leaving one of the eight cores for the reader/writer")
    corpus_id = uuid4().hex
    call = prepare.spawn(seed, corpus_id, fingerprint_workers)
    print(f"Submitted grouped SQL preparation: {call.object_id}", flush=True)
    print(f"Corpus directory: {OUTPUT_PATH.parent / 'builds' / corpus_id / 'pretrain_data'}", flush=True)
    if not submit_only:
        print(json.dumps(call.get(), indent=2), flush=True)
