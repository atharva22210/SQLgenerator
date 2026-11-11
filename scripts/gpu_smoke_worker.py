"""Run with torchrun: one worker owns one GPU; two workers sum their tensors."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-family", default="L4")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} CUDA GPUs, found {count}")
    if world_size != args.expected_gpus or not 0 <= local_rank < count:
        raise RuntimeError("Worker count and GPU assignments do not match the allocation")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if args.expected_gpu_family != "any" and args.expected_gpu_family not in properties.name:
        raise RuntimeError(
            f"Expected a {args.expected_gpu_family}, received {properties.name}"
        )
    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} "
        f"visible_gpus={count} device={device} name={properties.name}",
        flush=True,
    )

    # This checks actual CUDA execution, even in the single-GPU case.
    values = torch.arange(1, 5, dtype=torch.float32, device=device)
    torch.testing.assert_close(values * 2, torch.tensor([2., 4., 6., 8.], device=device))

    before = float(rank + 1)
    tensor = torch.tensor([before], device=device)
    try:
        if world_size > 1:
            dist.init_process_group(
                backend="nccl", timeout=timedelta(seconds=60), device_id=device
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(tensor, torch.tensor([expected], device=device))
        report = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "visible_gpus": count,
            "gpu_name": properties.name,
            "gpu_memory_gib": properties.total_memory / 1024**3,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "backend": "nccl" if world_size > 1 else "none",
            "before": before,
            "after": tensor.item(),
            "expected": expected,
            "passed": True,
        }
        args.result_dir.mkdir(parents=True, exist_ok=True)
        (args.result_dir / f"rank-{rank}.json").write_text(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
