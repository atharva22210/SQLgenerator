"""Publish complete JSON snapshots without replacing another index."""
import json
import os
from pathlib import Path
import tempfile


def save_json(path: Path, payload: dict) -> None:
    """Atomically write payload to a new path; refuse an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        # Atomic no-clobber publication on the same filesystem: readers
        # see either the complete file or no file. Never an unfinished JSON.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
