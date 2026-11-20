"""Verified catalogue input and the shared schema-only text representation."""
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_catalog(bundle: Path) -> tuple[list[dict], str]:
    """Verify the schema file against the completed bundle's checksum.

    A checksum detects corruption/staleness, not malicious replacement of both
    data and manifest. Load bundles from our trusted storage, not user uploads.
    Only the explicit catalog path is opened; labels/ is never scanned.
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    path = bundle / "catalog/tables.jsonl"
    checksum = sha256(path)
    if not manifest.get("complete") or checksum != manifest["files"]["catalog/tables.jsonl"]["sha256"]:
        raise ValueError("Incomplete or corrupt schema catalogue")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != manifest["counts"]["tables"]:
        raise ValueError("Catalogue count disagrees with manifest")
    return rows, checksum


def searchable_text(doc: dict) -> str:
    """Use schema fields, excluding opaque database IDs and reference answers.

    Repeated template words like 'Columns' and 'none declared' are omitted.
    Domain descriptions help lexical matching but may also add noise; this
    fixed representation is recorded in the index version for reproducibility.
    """
    parts = [doc["table"], doc["domain"], doc["domain_description"]]
    parts += [f"{column['name']} {column['type']} {column.get('comment', '')}" for column in doc["columns"]]
    parts += doc["primary_key"]
    for key in doc["foreign_keys"]:
        parts += key["columns"] + [key["referenced_table"]] + key["referenced_columns"]
    return " ".join(parts)
