"""CPU-only Day 4 ingestion into the shared MySQL service and Modal Volume."""

import hashlib
import json
import logging
from pathlib import Path

import modal

from infra.db_client import ROOT


app = modal.App("sqlgen-posttrain-ingest")
volume = modal.Volume.from_name("post-train-data", create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.13.15")
    .uv_sync(
        str(ROOT), extra_options="--locked --no-default-groups", frozen=False,
        uv_version="0.13.7",
        env={"UV_DEFAULT_INDEX": "https://pypi.org/simple", "UV_INDEX_URL": "https://pypi.org/simple"},
    )
    .add_local_python_source("infra", "data")
)
MOUNT = Path("/vol")


@app.function(
    image=image, volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name("mysql-secret"), modal.Secret.from_name("mysql-eval-secret")],
    cpu=2, memory=4096, timeout=86400, startup_timeout=600,
    max_containers=1, scaledown_window=2, region="ap-southeast",
)
def ingest(run_id: str, sft_rows: int, rl_rows: int, test_rows: int, workers: int, smoke: bool,
           publish: bool = False) -> dict:
    from data.ingest_mysql_benchmarks import run

    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    # Only packages are shipped, never the laptop's .env or local credentials.
    output = MOUNT / (f"checks/{run_id}" if smoke else "posttrain_data")
    if publish and not (output / "ready.json").is_file():
        raise ValueError("Finish validation and review ready.json counts before publication")
    return run(output, run_id, sft_rows=sft_rows, rl_rows=rl_rows, test_rows=test_rows,
               workers=workers, publish=publish, commit=volume.commit)


@app.function(
    image=image, volumes={MOUNT: volume.with_mount_options(read_only=True)},
    secrets=[modal.Secret.from_name("mysql-eval-secret")],
    cpu=1, memory=2048, timeout=7200, scaledown_window=2, region="ap-southeast",
)
def verify(smoke_run_id: str = "", check_live: bool = True) -> dict:
    import pyarrow.parquet as pq
    import pymysql
    from infra.db_client import connection_options

    volume.reload()
    output = MOUNT / (f"checks/{smoke_run_id}" if smoke_run_id else "posttrain_data")
    manifest = json.loads((output / "manifest.json").read_text())
    if not manifest["complete"]:
        raise RuntimeError("Ingestion has not completed")
    result = {"volume_id": volume.object_id, "manifest": manifest, "live_queries_checked": 0}
    contexts, identities = {}, set()
    options = connection_options("eval")
    options.update(database=None, autocommit=True, read_timeout=30)
    with pymysql.connect(**options) as reader:
        with reader.cursor() as cur:
            cur.execute("SET SESSION max_execution_time = 5000")
        for stage, metadata in manifest["files"].items():
            path = output / metadata["name"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise AssertionError("Parquet checksum mismatch")
            table = pq.read_table(path)
            if table.num_rows != metadata["rows"]:
                raise AssertionError("Parquet row count mismatch")
            contexts[stage] = set()
            for row in table.to_pylist():
                identity = (row["source_split"], row["source_index"])
                if identity in identities:
                    raise AssertionError("Source row occurs in multiple output splits")
                identities.add(identity)
                contexts[stage].add(row["context_sha256"])
                if check_live and stage != "sft":
                    reader.select_db(row["db_name"])
                    with reader.cursor() as cur:
                        cur.execute("START TRANSACTION READ ONLY")
                        try:
                            cur.execute(row["sql"])
                            if len(cur.fetchall()) != row["result_row_count"]:
                                raise AssertionError("Live result count differs from ingestion")
                        finally:
                            reader.rollback()
                    result["live_queries_checked"] += 1
                    if result["live_queries_checked"] % 250 == 0:
                        print(json.dumps({"live_queries_checked": result["live_queries_checked"]}), flush=True)
    for left, right in (("sft", "rl"), ("sft", "test"), ("rl", "test")):
        if contexts[left] & contexts[right]:
            raise AssertionError("Context overlap between splits")
    return {**result, "passed": True}


@app.local_entrypoint()
def main(run_id: str = "", smoke: bool = False, verify_only: bool = False,
         sft_rows: int = 45000, rl_rows: int = 2000, test_rows: int = 500, workers: int = 4,
         publish_only: bool = False, submit_only: bool = False):
    import re
    import secrets

    if verify_only:
        if smoke and not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise ValueError("Smoke verification requires its 12-character run-id")
        if submit_only:
            call = verify.spawn(run_id if smoke else "")
            print(json.dumps({"function_call_id": call.object_id, "operation": "verify"}), flush=True)
            return
        result = verify.remote(run_id if smoke else "")
    else:
        if publish_only and not run_id:
            raise ValueError("Publication requires the validated run-id")
        run_id = run_id or secrets.token_hex(6)
        if not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise ValueError("run-id must be 12 lowercase hex characters")
        print(f"Run ID: {run_id}; volume: post-train-data; smoke: {smoke}", flush=True)
        if submit_only:
            call = ingest.spawn(run_id, sft_rows, rl_rows, test_rows, workers, smoke, publish_only)
            print(json.dumps({"function_call_id": call.object_id, "run_id": run_id,
                              "operation": "publish" if publish_only else "validate"}), flush=True)
            return
        manifest = ingest.remote(run_id, sft_rows, rl_rows, test_rows, workers, smoke, publish_only)
        print(json.dumps(manifest, indent=2), flush=True)
        result = verify.remote(run_id if smoke else "") if publish_only else manifest
    print(json.dumps(result, indent=2), flush=True)
