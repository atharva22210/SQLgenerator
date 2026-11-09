"""Explicit utilities for single-process, DataParallel, and DDP execution."""

import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def initialize_distributed() -> torch.device:
    """Initialize torchrun-launched DDP and return the process-local device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    return device


def wrap_model(model: nn.Module, device: torch.device, mode: str = "auto") -> nn.Module:
    """Move and wrap a model; use ``torchrun`` with ``mode="ddp"`` for DDP."""
    if mode not in {"auto", "none", "data_parallel", "ddp"}:
        raise ValueError("mode must be auto, none, data_parallel, or ddp")
    model = model.to(device)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    if mode == "ddp" or (mode == "auto" and distributed):
        if not distributed:
            raise RuntimeError("call initialize_distributed before using DDP")
        device_ids = [device.index] if device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)
    if mode == "data_parallel" or (
        mode == "auto" and device.type == "cuda" and torch.cuda.device_count() > 1
    ):
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("DataParallel requires at least two CUDA devices")
        return nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when a parallel wrapper is in use."""
    if isinstance(model, (nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model
