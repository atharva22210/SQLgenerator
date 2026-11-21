"""Resolve a completed versioned SQL corpus on the pretraining volume."""

import json
from pathlib import Path


GROUPED_SQL_FORMAT_VERSION = 2


def resolve_grouped_sql(data_dir: Path, explicit: Path | None = None) -> tuple[Path, dict]:
    """Return a compatible directory and manifest after checking both binaries.

    A build can publish examples before its binary files are ready. Only a
    compatible completion manifest makes it eligible for automatic selection.
    """
    root = data_dir / "grouped_sql"
    flat = data_dir / "sql_metadata.json"
    if explicit is not None and (explicit / "sql_metadata.json").is_file():
        directory = explicit
        metadata = json.loads((directory / "sql_metadata.json").read_text())
    elif flat.is_file() and (explicit is None or explicit == data_dir or not explicit.exists()):
        directory = data_dir
        metadata = json.loads(flat.read_text())
        # Older checkpoints pin the version directory by corpus ID.
        if explicit not in (None, data_dir) and explicit.name != metadata["corpus_id"]:
            raise ValueError(f"Saved corpus {explicit} differs from {flat}")
    elif explicit is not None:
        directory = explicit
        metadata = json.loads((directory / "metadata.json").read_text())
    else:
        manifests = sorted(root.glob("*/metadata.json"),
                           key=lambda path: (path.stat().st_mtime_ns, path.parent.name), reverse=True)
        for manifest in manifests:
            metadata = json.loads(manifest.read_text())
            if metadata.get("format_version") == GROUPED_SQL_FORMAT_VERSION:
                directory = manifest.parent
                break
        else:
            raise FileNotFoundError(f"No completed Grouped SQL corpus under {root}; wait for its metadata.json")
    if (metadata.get("format_version") != GROUPED_SQL_FORMAT_VERSION
            or metadata.get("token_dtype") != "uint32"
            or metadata.get("audit", {}).get("overlaps") != dict(db_id=0, repo_name=0, query_hash=0)):
        raise ValueError(f"Invalid Grouped SQL completion metadata or overlap audit at {directory}")
    for split in ("train", "val"):
        key = f"sql_{split}"
        path = directory / f"{key}.bin"
        tokens = metadata["token_counts"][key]
        if tokens <= 0 or not path.is_file() or path.stat().st_size != tokens * 4:
            raise ValueError(f"{path} does not match completed Grouped SQL metadata")
    return directory, metadata
