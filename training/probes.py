"""Observations that do not change the training sampler, gradients or RNG states."""
import torch


def batch_fingerprint(batches) -> str:
    """Hash actual CPU token IDs, targets, positions and attention boundaries.

    Comparing hashes across branches verifies their sampled inputs, beyond
    checking only target counts or SQL proportions. Does not consume any RNG.
    """
    import hashlib
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in (batch.input_ids, batch.labels, batch.position_ids,
                       batch.attention_layout.token_indices, batch.attention_layout.cu_seqlens):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fixed_batch_loss(model, batch, config, device) -> float:
    """Evaluate saved [microbatch, sequence] batches after an update.

    The same examples at every observation remove changing batch contents as a
    source of loss fluctuations. This is an in-training diagnostic, not a held-out
    metric or a curvature measurement. Restore mode and RNG even on exceptions.
    """
    training = model.training
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            batches = batch if isinstance(batch, list) else [batch]
            targets = sum(item.predicted_tokens for item in batches)
            total = torch.zeros((), device=device)
            for item in batches:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    _, loss = model(**item.model_inputs(config.mode, device), loss_chunk_size=config.loss_chunk_size)
                total += loss * (item.predicted_tokens / targets)
            return float(total)
    finally:
        model.train(training)


def update_direction(model, before: dict) -> dict:
    """Return actual FP32 parameter differences on CPU, keyed like state_dict.

    Called only at retained milestones. Differences include gradient clipping,
    decay and floating-point rounding, so a later directional probe need not
    guess the update from a gradient or optimizer formula.
    """
    return {name: (parameter.detach() - before[name]).cpu()
            for name, parameter in model.named_parameters()}
