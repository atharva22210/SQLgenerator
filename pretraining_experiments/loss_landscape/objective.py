"""The unchanged two-document probe, evaluated in FP32 with differentiable attention."""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


class ProbeObjective:
    """Cache standalone probe inputs and return a token-weighted scalar loss.

    Saved rows contain one document each. Removing padding preserves all real
    inputs/targets/positions and lets the standard causal attention path run in
    FP32. The MATH backend supports second derivatives, unlike the training
    FlashAttention backward kernel. No model code or training backend changes.
    """

    def __init__(self, model, batches, *, chunk_size=128):
        self.model = model
        self.inputs = []
        self.targets = sum(batch.predicted_tokens for batch in batches)
        self.chunk_size = chunk_size
        device = next(model.parameters()).device
        for batch in batches:
            if len(batch.attention_layout.sequence_lengths) != 1 or batch.input_ids.shape[0] != 1:
                raise ValueError('FP32 probe requires one standalone document per saved row')
            length = batch.non_pad_tokens
            self.inputs.append(({
                'input_ids': batch.input_ids[:, :length].to(device),
                'labels': batch.labels[:, :length].to(device),
                'position_ids': batch.position_ids[:, :length].to(device),
                'valid_target_count': batch.predicted_tokens,
            }, batch.predicted_tokens))

    def __call__(self):
        total = next(self.model.parameters()).new_zeros(())
        with sdpa_kernel(SDPBackend.MATH):
            for inputs, count in self.inputs:
                _, loss = self.model(**inputs, loss_chunk_size=self.chunk_size)
                total = total + loss * (count / self.targets)
        return total
