"""Bounded A10 correctness check, submitted without a persistent local client.

Run with ``modal run --detach -m pretraining_experiments.cuda_check::main`` and pass the
team's W&B entity. This uses a fresh checkpoint name, checks actual CUDA
attention before training, and allows at most 15 minutes of function runtime.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import uuid

import modal

from training.config import ROOT, TrainConfig, image, volume, wandb_secret


app = modal.App("sqlgen-packing-correctness")
# The shared base image already ships the experiment package; this check also
# needs the standalone model-test module in its remote Python path.
check_image = (
    image
    .add_local_file(ROOT / "tests/test_model.py", "/root/test_model.py")
)


@app.function(
    image=check_image, volumes={"/vol": volume}, secrets=[wandb_secret],
    gpu="A10", cpu=4, memory=16_384, timeout=900,
    startup_timeout=600, scaledown_window=2,
)
def check_on_gpu(run_name: str, wandb_entity: str) -> dict:
    """Check CUDA loss/gradients, then train two updates and inspect saved state.

    Args:
        run_name: Fresh directory name under the normal checkpoint root.
        wandb_entity: Existing W&B team or user; credentials come from Modal.

    Returns model-test counts and the training/checkpoint result. Exceptions
    stop the smoke run immediately; no longer training run is queued.
    """
    import unittest

    import torch
    from test_model import TransformerTests
    from pretraining_experiments.sequence_packing import WANDB_PROJECT, _checkpoint_path, _run

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The correctness check requires a CUDA bf16 GPU")
    path = _checkpoint_path(run_name)
    volume.reload()
    if path.exists():
        raise FileExistsError(f"Correctness checks require a fresh checkpoint: {path}")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TransformerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Model correctness tests failed or skipped; training was not started")
    print(json.dumps({"model_tests_passed": result.testsRun,
                      "gpu": torch.cuda.get_device_name()}), flush=True)

    # Keep the requested A10 batch/length/chunk settings. Only duration and
    # evaluation frequency change so this also exercises durable checkpoints.
    config = TrainConfig(mode="masked_pack", max_steps=2, warmup_steps=1,
                         eval_every=1, val_batches=1)
    trained = _run(config, wandb_entity, WANDB_PROJECT, run_name)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["update"] != 2 or state["config"] != asdict(config):
        raise RuntimeError("Saved checkpoint does not match the two-update smoke run")
    # These reads happen after training, outside the optimized microbatch path.
    tensors = list(state["model"].values())
    for parameter_state in state["optimizer"]["state"].values():
        tensors.extend(value for value in parameter_state.values() if torch.is_tensor(value))
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("Model or optimizer checkpoint contains nonfinite values")
    return {"passed": True, "model_tests_passed": result.testsRun,
            "checkpoint_finite": True, **trained}


@app.local_entrypoint()
def main(wandb_entity: str = "") -> None:
    """Spawn the bounded check and print its ID; use modal run --detach."""
    if not wandb_entity:
        raise ValueError("Supply --wandb-entity for W&B logging")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"a10-correctness-{stamp}-{uuid.uuid4().hex[:8]}"
    call = check_on_gpu.spawn(name, wandb_entity)
    print(json.dumps({"run": name, "call_id": call.object_id}), flush=True)
