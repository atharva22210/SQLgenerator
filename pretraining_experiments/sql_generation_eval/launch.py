"""Run post-training inference on Modal and download the judge input as JSONL."""
import json
from pathlib import Path

import modal
from training.config import DATA_DIR, image, volume

app = modal.App('sqlgen-held-out-generation')


@app.function(image=image, gpu='A10', memory=16384, timeout=86400, volumes={'/vol': volume})
def generate(checkpoint: str, count: int, seed: int, max_new_tokens: int):
    """Load the shared volume's final checkpoint and return serializable rows."""
    from transformers import AutoTokenizer
    from training.config import _tokenizer_path
    from .evaluate import evaluate

    volume.reload()
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(), local_files_only=True)
    return evaluate(Path(checkpoint), DATA_DIR, tokenizer, count, seed, max_new_tokens, 'cuda')


@app.local_entrypoint()
def main(checkpoint: str, output: str, count: int = 60, seed: int = 12345, max_new_tokens: int = 256):
    """checkpoint is a Modal volume path; output is a new local JSONL file."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate.remote(checkpoint, count, seed, max_new_tokens)
    with path.open('x', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Saved {len(rows)} SQL generations to {path.resolve()}')
