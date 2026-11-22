"""Fit a 32K byte-level tokenizer and re-encode the existing audited corpus.

Run with .venv/bin/modal run --detach -m data.byte_bpe.train_byte_bpe. The returned build
directory contains the tokenizer and a pretrain_data tree with the original
document order, split assignments, and SQL overlap audit.
"""
import json
import time
from pathlib import Path
from uuid import uuid4

import modal

from data.grouped_corpus import resolve_grouped_sql
from data.prepare_pretrain_data import OUTPUT_PATH as DATA_DIR, find_tokenizer_path
from training.config import _validate_corpus_files, image, volume

app = modal.App("sqlgen-byte-bpe")
settings = dict(image=image, volumes={"/vol": volume}, timeout=86400)
BLOCK = 1_000_000


def documents(path, eos, start, stop):
    """Yield batches of complete Qwen documents whose first ID lies in [start, stop).

    A worker beginning mid-document skips its tail. The preceding worker reads
    beyond its nominal stop to finish that document, keeping every EOS once.
    """
    import numpy as np
    tokens = np.memmap(path, dtype="<u4", mode="r")
    cursor = start
    first = start if start == 0 or tokens[start - 1] == eos else None
    batch = []
    size = 0
    while cursor < len(tokens):
        if cursor >= stop and (first is None or first >= stop):
            break
        chunk = tokens[cursor:cursor + BLOCK]
        for offset in np.flatnonzero(chunk == eos):
            end = cursor + int(offset)
            if first is not None:
                if first >= stop:
                    break
                batch.append(tokens[first:end].tolist())
                size += end - first + 1
                if size >= 65536:
                    yield batch
                    batch, size = [], 0
            first = end + 1
            if first >= stop:
                break
        cursor += len(chunk)
        if first is not None and first >= stop:
            break
    if batch:
        yield batch


def decode(base, ids):
    """Preserve whitespace and project markers while stripping only structural EOS."""
    return base.batch_decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


@app.function(**settings, cpu=4, memory=8192)
def convert(source: str, output: str, tokenizer_dir: str, start: int, stop: int):
    """Write one ordered uint32 shard and return its length and context-fit metrics."""
    import numpy as np
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(find_tokenizer_path(), local_files_only=True)
    bpe = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    count = rows = fit = old_fit = 0
    # The final closing answer marker measures answer retention even if a
    # document has trailing text. Repository SQL lacks this marked Q&A format.
    answer_counts = {}
    track_answers = Path(source).name.startswith("sql_")
    closing = [base.convert_tokens_to_ids("</sql>"), bpe.convert_tokens_to_ids("</sql>")]
    began = time.monotonic()
    with Path(output).open("wb") as stream:
        for batch in documents(source, base.eos_token_id, start, stop):
            encoded = bpe(decode(base, batch), add_special_tokens=False, verbose=False)["input_ids"]
            if track_answers:
                for original, new in zip(batch, encoded):
                    for label, row, marker in zip(("qwen", "bpe"), (original, new), closing):
                        if marker in row:
                            end = len(row) - row[::-1].index(marker)
                            for window in (2048, 4096):
                                key = f"{label}_answer_fits_{window}"
                                answer_counts[key] = answer_counts.get(key, 0) + int(end <= window)
                            key = f"{label}_marked_documents"
                            answer_counts[key] = answer_counts.get(key, 0) + 1
            for row in encoded:
                row.append(bpe.eos_token_id)
            flat = np.fromiter((token for row in encoded for token in row), dtype="<u4")
            flat.tofile(stream)
            count += len(flat)
            rows += len(batch)
            fit += sum(len(row) <= 2048 for row in encoded)
            old_fit += sum(len(row) + 1 <= 2048 for row in batch)
    volume.commit()
    return dict(tokens=count, documents=rows, fit=fit, qwen_fit=old_fit, seconds=time.monotonic()-began, **answer_counts)


def fit_bpe(base, sql, vocab_size, windows=64):
    """Fit merges from seeded train-only windows in both corpus files."""
    import random
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from data.tokenizer_setup import ADDITIONAL_SPECIAL_TOKENS, PAD_TOKEN
    rng = random.Random(42)
    def sample():
        # Spread short windows across the two files rather than learning only
        # from their prefixes. Equal old-token budgets approximate a 50/50 mix.
        for path in (DATA_DIR / "english_train.bin", sql / "sql_train.bin"):
            length = path.stat().st_size // 4
            for i in range(windows):
                start = rng.randrange(i * length // windows, (i + 1) * length // windows - 65536)
                for batch in documents(path, base.eos_token_id, start, start + 65536):
                    yield decode(base, batch)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    specials = [base.eos_token, PAD_TOKEN, *ADDITIONAL_SPECIAL_TOKENS]
    backend.train_from_iterator(sample(), trainer=trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    bpe = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token=base.eos_token,
                                  pad_token=PAD_TOKEN, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS,
                                  clean_up_tokenization_spaces=False)
    for text in [*specials, " SELECT café_名\n\tFROM x; 🎉"]:
        ids = bpe.encode(text, add_special_tokens=False)
        if bpe.decode(ids) != text or (text in specials and len(ids) != 1):
            raise ValueError(f"Tokenizer round-trip failed for {text!r}")
    if len(bpe) != vocab_size:
        raise ValueError("Training sample did not reach requested vocabulary size")
    return bpe


@app.function(**settings, cpu=8, memory=16384)
def build(tokenizer_build: str = "", build_id: str = ""):
    """Fit using training documents, then dispatch independent document-aligned shards."""
    import shutil
    from transformers import AutoTokenizer
    from data.prepare_pretrain_data import tokenizer_fingerprint

    sql, sql_meta = resolve_grouped_sql(DATA_DIR)
    english = json.loads((DATA_DIR / "metadata.json").read_text())
    _validate_corpus_files(DATA_DIR, english, english_only=True)
    base = AutoTokenizer.from_pretrained(find_tokenizer_path(), local_files_only=True)
    if any(meta["tokenizer_sha256"] != tokenizer_fingerprint(base) for meta in (english, sql_meta)):
        raise ValueError("Source tokenizer fingerprint mismatch")
    root = Path("/vol/bpe_tokenization/builds") / (build_id or uuid4().hex)
    tokenizer_dir = root / "tokenizer"
    data = root / "pretrain_data"
    grouped = data
    parts = root / "parts"
    for directory in (tokenizer_dir, data, parts):
        directory.mkdir(parents=True)
    print(f"Build {root}", flush=True)
    if tokenizer_build:
        # Reuse the fitted tokenizer; isolated output avoids trusting partial shards.
        saved = Path("/vol/bpe_tokenization/builds") / tokenizer_build / "tokenizer"
        shutil.copytree(saved, tokenizer_dir, dirs_exist_ok=True)
        bpe = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        volume.commit()
    else:
        began = time.monotonic()
        bpe = fit_bpe(base, sql, 32768)
        bpe.save_pretrained(tokenizer_dir)
        volume.commit()
        print(f"Fitted {len(bpe)} IDs in {time.monotonic()-began:.1f}s", flush=True)
    jobs = []
    files = []
    for source_root, dest_root, kind in ((DATA_DIR, data, "english"), (sql, grouped, "sql")):
        for split in ("train", "val"):
            name = f"{kind}_{split}"
            source = source_root / f"{name}.bin"
            length = source.stat().st_size // 4
            shards = []
            for start in range(0, length, 100_000_000):
                shard = parts / f"{name}-{start}.bin"
                jobs.append((str(source), str(shard), str(tokenizer_dir), start, min(start+100_000_000, length)))
                shards.append(shard)
            files.append((name, dest_root / source.name, shards))
    stats = list(convert.starmap(jobs))
    volume.reload()
    summary = {}
    for name, dest, shards in files:
        group_stats, stats = stats[:len(shards)], stats[len(shards):]
        summary[name] = {key: sum(s.get(key, 0) for s in group_stats) for key in set().union(*group_stats)}
        with dest.open("wb") as out:
            for shard in shards:
                with shard.open("rb") as stream:
                    shutil.copyfileobj(stream, out, 8 << 20)
        if dest.stat().st_size != summary[name]["tokens"] * 4:
            raise ValueError(f"Incomplete output {dest}")
        for shard in shards:
            shard.unlink()
    # Only describe corpora actually converted, preserving SQL's group audit.
    common = dict(tokenizer_vocab_size=len(bpe), tokenizer_sha256=tokenizer_fingerprint(bpe),
                  tokenizer_path=str(tokenizer_dir), eos_token_id=bpe.eos_token_id)
    for dest, original, kind in ((data, english, "english"), (grouped, sql_meta, "sql")):
        metadata = dict(original, **common)
        if kind == "english":
            for field in ("datasets", "dataset_revisions"):
                metadata[field] = {"english": original[field]["english"]}
        metadata["token_counts"] = {k: v["tokens"] for k,v in summary.items() if k.startswith(kind)}
        (dest / ("sql_metadata.json" if kind == "sql" else "metadata.json")).write_text(json.dumps(metadata, indent=2))
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return str(root), summary


@app.local_entrypoint()
def main(tokenizer_build: str = ""):
    build_id = uuid4().hex
    call = build.spawn(tokenizer_build, build_id)
    print(f"Corpus output /vol/bpe_tokenization/builds/{build_id}", flush=True)
    print(f"Submitted {call.object_id}", flush=True)
