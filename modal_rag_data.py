"""Collect the RAG data bundle using existing data and read-only MySQL access.

No GPUs, training, admin secrets, new databases or SQL answer execution. The
only writes go into a new /vol/rag/<bundle-id>/ directory on post-train-data.
"""
import json
from pathlib import Path
import re
import uuid

import modal

from infra.db_client import ROOT


app = modal.App("sqlgen-rag-data")
volume = modal.Volume.from_name("post-train-data")
image = (
    modal.Image.debian_slim(python_version="3.13.15")
    .uv_sync(str(ROOT), extra_options="--locked --no-default-groups", frozen=False,
             uv_version="0.13.7",
             env={"UV_DEFAULT_INDEX": "https://pypi.org/simple", "UV_INDEX_URL": "https://pypi.org/simple"})
    .add_local_python_source("infra", "rag")
)


@app.function(image=image, volumes={"/vol": volume},
              secrets=[modal.Secret.from_name("mysql-eval-secret")],
              cpu=1, memory=2048, timeout=1800, startup_timeout=600,
              max_containers=1, scaledown_window=2, region="ap-southeast")
def prepare(bundle_id: str) -> dict:
    """Verify live metadata and persist a new bundle, then return its manifest.

    bundle_id is a unique folder name, never a database name. The source
    Parquet files and existing MySQL databases remain unchanged.
    """
    from rag.prepare_data import build_bundle

    if not re.fullmatch(r"gretel-v1-[0-9a-f]{12}", bundle_id):
        raise ValueError("Expected gretel-v1- followed by 12 hexadecimal characters")
    volume.reload()
    manifest = build_bundle(Path("/vol/posttrain_data"), Path("/vol/rag") / bundle_id,
                            check_live=True)
    volume.commit()  # All files are closed; only now advertise the bundle.
    result = {"volume": "post-train-data", "directory": f"rag/{bundle_id}",
              "counts": manifest["counts"], "live_verification": manifest["live_verification"]}
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(bundle_id: str = ""):
    """Spawn one CPU collection job and print the ID used to follow its result."""
    bundle_id = bundle_id or f"gretel-v1-{uuid.uuid4().hex[:12]}"
    call = prepare.spawn(bundle_id)
    print(json.dumps({"bundle_id": bundle_id, "function_call_id": call.object_id,
                      "volume": "post-train-data", "directory": f"rag/{bundle_id}"}), flush=True)
