"""Minimal two-GPU NCCL communication check; no W&B or Volume is used."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import modal


ROOT = Path(__file__).resolve().parent
app = modal.App("sqlgen-two-gpu-smoke")
base_image = modal.Image.debian_slim(python_version="3.13.15")
image = (
    base_image.uv_sync(
        str(ROOT),
        extra_options="--locked",
        uv_version="0.13.7",
        frozen=False,
        env={
            "UV_DEFAULT_INDEX": "https://pypi.org/simple",
            "UV_INDEX_URL": "https://pypi.org/simple",
        },
    )
    .add_local_file(ROOT / "scripts/gpu_smoke_worker.py", "/root/gpu_smoke_worker.py")
)


@app.function(
    image=image,
    # Both options are relatively close in cost; every option requests two
    # GPUs on the same physical machine.
    gpu=["A10:2", "L4:2"],
    cpu=2,
    memory=4096,
    timeout=180,
    startup_timeout=120,
    scaledown_window=2,
)
def communicate() -> dict:
    """Start two workers and verify one NCCL all-reduce."""
    with tempfile.TemporaryDirectory() as result_dir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=2",
                "/root/gpu_smoke_worker.py",
                "--expected-gpus",
                "2",
                "--expected-gpu-family",
                "any",
                "--result-dir",
                result_dir,
            ],
            check=True,
            timeout=120,
        )
        workers = [
            json.loads((Path(result_dir) / f"rank-{rank}.json").read_text())
            for rank in range(2)
        ]
    return {"passed": all(worker["passed"] for worker in workers), "workers": workers}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(communicate.remote(), indent=2), flush=True)
