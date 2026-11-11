"""Short Modal L4 checks. All GPU code runs remotely, not on the laptop."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import modal


ROOT = Path(__file__).resolve().parent
app = modal.App("sqlgen-infra")
base_image = modal.Image.debian_slim(python_version="3.13.15")
sync_options = {
    "uv_version": "0.13.7",
    "frozen": False,
    # Match the lockfile instead of inheriting a builder's package mirror.
    "env": {
        "UV_DEFAULT_INDEX": "https://pypi.org/simple",
        "UV_INDEX_URL": "https://pypi.org/simple",
    },
}
image = (
    base_image.uv_sync(str(ROOT), extra_options="--locked", **sync_options)
    .add_local_file(ROOT / "scripts/gpu_smoke_worker.py", "/root/gpu_smoke_worker.py")
)
tracking_image = base_image.uv_sync(
    str(ROOT), extra_options="--locked --no-group model", **sync_options
)
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])


def _run_workers(gpu_count: int) -> dict:
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as result_dir:
        subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.run",
                "--standalone", "--nnodes=1", f"--nproc-per-node={gpu_count}",
                "/root/gpu_smoke_worker.py", "--expected-gpus", str(gpu_count),
                "--result-dir", result_dir,
            ],
            check=True,
            timeout=180,
        )
        workers = [
            json.loads((Path(result_dir) / f"rank-{rank}.json").read_text())
            for rank in range(gpu_count)
        ]
    return {
        "gpu_count": gpu_count,
        "passed": all(worker["passed"] for worker in workers),
        # Includes interpreter startup; this is not a communication benchmark.
        "elapsed_seconds": time.perf_counter() - start,
        "workers": workers,
    }


@app.function(
    image=image, gpu="L4", cpu=2, memory=4096, timeout=300,
    startup_timeout=180, scaledown_window=2,
)
def single_l4() -> dict:
    return _run_workers(1)


@app.function(
    image=image, gpu="L4:2", cpu=2, memory=4096, timeout=300,
    startup_timeout=180, scaledown_window=2,
)
def dual_l4() -> dict:
    return _run_workers(2)


def _ensure_project(entity: str, project: str) -> dict:
    """Create a private W&B project if absent; preserve existing visibility."""
    import base64
    import os
    import urllib.request

    def graphql(query: str, variables: dict) -> dict:
        credentials = base64.b64encode(f"api:{os.environ['WANDB_API_KEY']}".encode()).decode()
        request = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Basic {credentials}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get("errors"):
            messages = [error["message"] for error in result["errors"]]
            raise RuntimeError(f"W&B project request failed: {messages}")
        return result["data"]

    existing = graphql(
        "query($entity: String!, $project: String!) { "
        "project(entityName: $entity, name: $project) { name access } }",
        {"entity": entity, "project": project},
    )["project"]
    if existing is None:
        existing = graphql(
            "mutation($input: UpsertModelInput!) { "
            "upsertModel(input: $input) { project { name access } } }",
            {"input": {"entityName": entity, "name": project, "access": "PRIVATE"}},
        )["upsertModel"]["project"]
    return {"entity": entity, "project": project, "access": existing["access"]}


@app.function(
    image=tracking_image, secrets=[wandb_secret], memory=1024, timeout=120,
    startup_timeout=120, scaledown_window=2,
)
def prepare_tracking(entity: str, project: str) -> dict:
    return _ensure_project(entity, project)


@app.function(
    image=tracking_image, secrets=[wandb_secret], memory=1024, timeout=120,
    startup_timeout=120, scaledown_window=2,
)
def track_results(results: list[dict], entity: str, project: str) -> str:
    import wandb

    with wandb.init(
        entity=entity, project=project,
        job_type="infrastructure-smoke-test" if results else "tracking-smoke-test", mode="online",
        config={"gpu_type": "L4", "gpu_counts": [r["gpu_count"] for r in results]},
        settings=wandb.Settings(init_timeout=60),
    ) as run:
        for result in results:
            prefix = f"l4_{result['gpu_count']}"
            run.log({
                f"{prefix}/passed": int(result["passed"]),
                f"{prefix}/elapsed_seconds": result["elapsed_seconds"],
            })
        run.summary["tracking_connected"] = True
        run.summary["gpu_checks_run"] = len(results)
        if results:
            run.summary["all_passed"] = all(r["passed"] for r in results)
        table = wandb.Table(columns=["gpu_count", "rank", "gpu_name", "before", "after", "expected"])
        for result in results:
            for worker in result["workers"]:
                table.add_data(result["gpu_count"], *[worker[k] for k in (
                    "rank", "gpu_name", "before", "after", "expected"
                )])
        if results:
            run.log({"workers": table})
        return run.url


def _call(function, *args, timeout: float):
    """Connected local runner; cancellation depends on this client staying alive."""
    call = function.spawn(*args)
    try:
        return call.get(timeout=timeout)
    except TimeoutError:
        call.cancel(terminate_containers=True)
        raise


async def _call_async(function, *args, timeout: float):
    """Enforce a wall-clock wait and cancel the child on timeout or interruption."""
    call = None
    try:
        async with asyncio.timeout(timeout):
            call = await function.spawn.aio(*args)
            print(f"Submitted child call {call.object_id}", flush=True)
            return await call.get.aio()
    except BaseException:
        if call is not None:
            try:
                await asyncio.wait_for(
                    call.cancel.aio(terminate_containers=True), timeout=30
                )
            except Exception as error:
                print(f"Child cancellation could not be confirmed: {error}", flush=True)
        raise


@app.function(
    image=tracking_image, cpu=0.25, memory=1024, timeout=1200,
    startup_timeout=120, scaledown_window=2,
)
async def verify_dual_l4(
    wandb_entity: str = "", wandb_project: str = "sqlgen-infra",
    gpu_wait_seconds: int = 600,
) -> str:
    """Run the two-L4 verification from a CPU controller that can be detached."""
    if not wandb_entity:
        raise ValueError("--wandb-entity is required")
    if not 1 <= gpu_wait_seconds <= 600:
        raise ValueError("--gpu-wait-seconds must be between 1 and 600")

    print("Remote CPU controller: checking W&B access...", flush=True)
    await _call_async(prepare_tracking, wandb_entity, wandb_project, timeout=180)
    print(f"Requesting two L4s; total GPU call wait is {gpu_wait_seconds}s...", flush=True)
    result = await _call_async(dual_l4, timeout=gpu_wait_seconds)
    print(json.dumps({"gpu_result": result}, indent=2), flush=True)
    url = await _call_async(
        track_results, [result], wandb_entity, wandb_project, timeout=180
    )
    report = {"status": "passed", "gpu_result": result, "wandb_url": url}
    print(json.dumps(report, indent=2), flush=True)
    return json.dumps(report)


@app.local_entrypoint()
def main(
    gpus: str = "both", log_wandb: bool = False,
    wandb_entity: str = "", wandb_project: str = "sqlgen-infra",
    gpu_wait_seconds: int = 360,
) -> None:
    if gpus not in {"1", "2", "both", "none"}:
        raise ValueError("--gpus must be 1, 2, both, or none")
    if gpus == "none" and not log_wandb:
        raise ValueError("--gpus none requires --log-wandb to test tracking")
    if gpu_wait_seconds < 1:
        raise ValueError("--gpu-wait-seconds must be positive")
    if log_wandb:
        if not wandb_entity:
            raise ValueError("Supply --wandb-entity when using --log-wandb")
        print("Checking W&B project access on CPU...", flush=True)
        print(json.dumps(_call(prepare_tracking, wandb_entity, wandb_project, timeout=180)), flush=True)
    results = []
    if gpus in {"1", "both"}:
        print("Starting single-L4 CUDA check...", flush=True)
        results.append(_call(single_l4, timeout=gpu_wait_seconds))
    if gpus in {"2", "both"}:
        print("Starting dual-L4 NCCL check...", flush=True)
        results.append(_call(dual_l4, timeout=gpu_wait_seconds))
    print(json.dumps(results, indent=2), flush=True)
    if log_wandb:
        print("Logging results to W&B on CPU...", flush=True)
        url = _call(track_results, results, wandb_entity, wandb_project, timeout=180)
        print(f"W&B run: {url}", flush=True)
