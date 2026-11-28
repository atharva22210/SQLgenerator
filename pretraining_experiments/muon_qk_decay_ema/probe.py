"""Small causal probe on identical SQL and English token sequences."""
import json
import math
import random

import torch

from model.transformer_35m import apply_rope


def causal_statistics(q, k, positions):
    """Summarize scaled logits for [heads, tokens, width] on sampled query rows.

    Each document is unpadded and evaluated alone. Row i may see keys 0..i.
    Normalizing entropy by log(i+1) makes uniform attention equal to one.
    """
    scores = q[:, positions].float() @ k.float().transpose(-1, -2) / math.sqrt(q.size(-1))
    valid = torch.arange(k.size(1), device=k.device)[None, :] <= positions[:, None]
    allowed = scores[:, valid]
    probs = scores.masked_fill(~valid, -torch.inf).softmax(-1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1) / (positions + 1).log()
    return {'max_score': allowed.amax(1), 'p99_score': allowed.quantile(.99, dim=1),
            'median_entropy': entropy.quantile(.5, dim=1), 'p05_entropy': entropy.quantile(.05, dim=1)}


class AttentionProbe:
    """Persist two fixed token prefixes and per-head measurements by update."""

    def __init__(self, validation, directory):
        self.directory = directory / 'probe'
        self.directory.mkdir(exist_ok=True)
        path = self.directory / 'tokens.json'
        if path.exists():
            self.documents = json.loads(path.read_text())
        else:
            self.documents = {}
            for index, kind in enumerate(('sql', 'english')):
                rng = random.Random(12345 + index)
                tokens = validation.sample(kind, rng, 256, first_slice=True)
                if len(tokens) < 2:
                    raise ValueError('Probe document needs at least two positions')
                self.documents[kind] = list(map(int, tokens))
            path.write_text(json.dumps(self.documents))

    @torch.inference_mode()
    def measure(self, model, update):
        """Return worst-head summary for W&B and retain each head on the Volume."""
        rows = []
        device = model.token_embedding.weight.device
        for kind, tokens in self.documents.items():
            ids = torch.tensor([tokens], device=device)
            x = model.token_embedding(ids)
            n = len(tokens)
            rotations = (model.rope_cosine[:n][None, None], model.rope_sine[:n][None, None])
            positions = torch.linspace(1, n-1, min(16, n-1), device=device).long()
            for layer, block in enumerate(model.blocks):
                attn = block.attention
                h = block.attention_norm(x)
                cfg = model.config
                q = attn.q_proj(h).reshape(n, cfg.n_heads, cfg.head_dim).transpose(0, 1)
                k = attn.k_proj(h).reshape(n, cfg.n_kv_heads, cfg.head_dim).transpose(0, 1)
                repeat = cfg.n_heads // cfg.n_kv_heads
                metrics = {'q_before': q.norm(dim=-1).mean(1),
                           'k_before': k.norm(dim=-1).mean(1).repeat_interleave(repeat)}
                if attn.q_norm is not None:
                    q, k = attn.q_norm(q), attn.k_norm(k)
                metrics.update(q_after=q.norm(dim=-1).mean(1),
                               k_after=k.norm(dim=-1).mean(1).repeat_interleave(repeat))
                q = apply_rope(q[None], *rotations)[0]
                k = apply_rope(k[None], *rotations)[0].repeat_interleave(repeat, dim=0)
                metrics.update(causal_statistics(q, k, positions))
                values = torch.stack(list(metrics.values()), dim=1).tolist()
                for head, numbers in enumerate(values):
                    rows.append({'document': kind, 'layer': layer, 'head': head,
                                 **dict(zip(metrics, numbers))})
                x = block(x, rotations, None, None)
        (self.directory / f'step-{update:06d}.json').write_text(json.dumps(rows, allow_nan=False))
        return {'probe/max_score': max(row['max_score'] for row in rows),
                'probe/min_p05_entropy': min(row['p05_entropy'] for row in rows)}
