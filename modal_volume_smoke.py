"""CPU-only persistence check for the team's existing Modal Volume."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import uuid

import modal


app = modal.App("sqlgen-volume-check")
image = modal.Image.debian_slim(python_version="3.13.15")
volume = modal.Volume.from_name("sqlgen-data")
read_volume = volume.with_mount_options(read_only=True)
MOUNT = Path("/data")


def _marker_path(marker: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", marker):
        raise ValueError("Marker must be a lowercase UUID hex string")
    return MOUNT / "volume_checks" / f"{marker}.json"


def _container_id() -> str:
    return os.environ.get("MODAL_TASK_ID") or socket.gethostname()


@app.function(
    image=image, volumes={MOUNT: volume}, memory=256,
    timeout=60, startup_timeout=60, scaledown_window=2,
)
def write_marker(marker: str) -> dict:
    path = _marker_path(marker)
    container = _container_id()
    payload = {
        "marker": marker,
        "message": "sqlgen shared-volume persistence check",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "writer_container": container,
    }
    content = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects any existing file, even on a marker collision.
    with path.open("xb") as file:
        file.write(content)
    volume.commit()  # Persist only after the file has been closed.
    return {
        "marker": marker,
        "path": str(path.relative_to(MOUNT)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "volume_id": volume.object_id,
        "writer_container": container,
    }


@app.function(
    image=image, volumes={MOUNT: read_volume}, memory=256,
    timeout=60, startup_timeout=60, scaledown_window=2,
)
def read_marker(marker: str, expected_sha256: str, expected_volume_id: str) -> dict:
    path = _marker_path(marker)
    if read_volume.object_id != expected_volume_id:
        raise RuntimeError("This profile/environment resolved a different Volume")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected SHA-256 must contain 64 lowercase hex characters")
    read_volume.reload()  # Refresh this mount before opening any files.
    content = path.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"File checksum mismatch: {actual_sha256}")
    payload = json.loads(content)
    if payload["marker"] != marker:
        raise RuntimeError("File contents do not match the requested marker")
    return {
        "passed": True,
        "marker": marker,
        "path": str(path.relative_to(MOUNT)),
        "sha256": actual_sha256,
        "volume_id": read_volume.object_id,
        "writer_container": payload["writer_container"],
        "reader_container": _container_id(),
    }


@app.local_entrypoint()
def main(
    marker: str = "", expected_sha256: str = "", expected_volume_id: str = "",
) -> None:
    if marker:
        _marker_path(marker)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Read-only verification requires --expected-sha256")
        if not expected_volume_id:
            raise ValueError("Read-only verification requires --expected-volume-id")
        result = read_marker.remote(marker, expected_sha256, expected_volume_id)
    else:
        if expected_sha256 or expected_volume_id:
            raise ValueError("Expected checksum/Volume ID require --marker")
        marker = uuid.uuid4().hex
        print("Writing a new marker and committing it to sqlgen-data...", flush=True)
        written = write_marker.remote(marker)
        print(json.dumps(written, indent=2), flush=True)
        expected_sha256 = written["sha256"]
        expected_volume_id = written["volume_id"]
        print("Reading and verifying it from a separate, read-only container...", flush=True)
        result = read_marker.remote(marker, expected_sha256, expected_volume_id)
        if result["volume_id"] != written["volume_id"]:
            raise RuntimeError("Writer and reader resolved different Volumes")
        if result["reader_container"] == written["writer_container"]:
            raise RuntimeError("Expected separate writer and reader containers")

    print(json.dumps(result, indent=2), flush=True)
    print("Read-only verification command (use your profile for workspace sanika-05):", flush=True)
    print(
        "uv run --locked modal run --env main --profile sanika-05 "
        f"modal_volume_smoke.py --marker {marker} --expected-sha256 {expected_sha256} "
        f"--expected-volume-id {expected_volume_id}",
        flush=True,
    )
