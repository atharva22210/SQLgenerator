"""Compare vocabularies on identical documents without converting the full corpus."""
import json
import random
from pathlib import Path
from uuid import uuid4

import modal

from data.byte_bpe.train_byte_bpe import documents, decode, fit_bpe, settings
from data.grouped_corpus import resolve_grouped_sql
from data.prepare_pretrain_data import OUTPUT_PATH as DATA_DIR, find_tokenizer_path
from training.config import volume

app = modal.App("sqlgen-compare-bpe")


@app.function(**settings, cpu=8, memory=16384)
def compare(saved: str):
    """Score per-corpus compression and prefix coverage on distributed validation windows."""
    from transformers import AutoTokenizer
    sql, _ = resolve_grouped_sql(DATA_DIR)
    base = AutoTokenizer.from_pretrained(find_tokenizer_path(), local_files_only=True)
    root = Path("/vol/bpe_comparison") / uuid4().hex
    root.mkdir(parents=True)
    candidates = {"Qwen": base, "32K": AutoTokenizer.from_pretrained(saved, local_files_only=True)}
    for name, size, windows in (("50K", 50000, 64), ("32K_double_sample", 32768, 128)):
        candidates[name] = fit_bpe(base, sql, size, windows)
        candidates[name].save_pretrained(root / name)
    results = {}
    for kind, path in (("English", DATA_DIR / "english_val.bin"), ("SQL", sql / "sql_val.bin")):
        rng = random.Random(2026)
        length = path.stat().st_size // 4
        scores = {name: dict(tokens=0, bytes=0, docs=0, mismatches=0, fit=0,
                             marked=0, answer2048=0, answer4096=0) for name in candidates}
        for i in range(64):
            start = rng.randrange(i*length//64, (i+1)*length//64 - 65536)
            for batch in documents(path, base.eos_token_id, start, start+65536):
                texts = decode(base, batch)
                for name, tokenizer in candidates.items():
                    rows = batch if name == "Qwen" else tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
                    score = scores[name]
                    closing = tokenizer.convert_tokens_to_ids("</sql>")
                    for text, row, restored in zip(texts, rows, decode(tokenizer, rows)):
                        score["tokens"] += len(row)
                        score["bytes"] += len(text.encode())
                        score["docs"] += 1
                        score["mismatches"] += restored != text
                        score["fit"] += len(row)+1 <= 2048
                        if kind == "SQL" and text.endswith("</sql>") and "\nQuestion:\n" in text:
                            if row[-1] != closing:
                                raise ValueError("Lost closing SQL marker")
                            score["marked"] += 1
                            score["answer2048"] += len(row) <= 2048
                            score["answer4096"] += len(row) <= 4096
        results[kind] = scores
    (root / "results.json").write_text(json.dumps(results, indent=2))
    volume.commit()
    print(root, json.dumps(results), flush=True)
    return results


@app.local_entrypoint()
def main(saved: str):
    print(compare.spawn(saved).object_id)
