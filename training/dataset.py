"""Identify the completed Modal corpus without copying multi-GB token files."""
import hashlib
import json
from pathlib import Path

from data.grouped_corpus import resolve_grouped_sql


def dataset_identity(data_dir: Path, sql_dir: Path | None = None) -> dict:
    """Validate manifests/file lengths and return a small, serializable identity.

    This identifies the published build, not a fresh checksum over every token.
    The loader separately verifies the actual tokenizer fingerprint. Pinning this
    dictionary rejects a changed flat-layout manifest between suite workers.
    """
    directory, sql = resolve_grouped_sql(data_dir, sql_dir)
    english = json.loads((data_dir / "metadata.json").read_text())
    for key in ("tokenizer_sha256", "tokenizer_vocab_size", "eos_token_id", "token_dtype"):
        if english[key] != sql[key]:
            raise ValueError(f"English and SQL disagree on {key}")
    counts = {**english['token_counts'], **sql['token_counts']}
    for key in ("english_train", "english_val", "sql_train", "sql_val"):
        path = (data_dir if key.startswith("english") else directory) / f"{key}.bin"
        if counts[key] <= 0 or path.stat().st_size != counts[key] * 4:
            raise ValueError(f"Token file does not match completion metadata: {path}")
    manifest = json.dumps({"english": english, "sql": sql}, sort_keys=True).encode()
    return {"manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "corpus_id": sql['corpus_id'], "sql_dir": str(directory),
            "tokenizer_sha256": sql['tokenizer_sha256'],
            "vocab_size": sql['tokenizer_vocab_size'], "token_counts": counts}
