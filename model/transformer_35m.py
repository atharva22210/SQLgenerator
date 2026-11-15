"""A readable, decoder-only transformer for next-token prediction.

This file defines the model, not its data loader or training loop. The input is
a batch of token IDs with shape [batch, time]. The output contains one score
per vocabulary token at each position: [batch, time, vocabulary].

Example::

    config = TransformerConfig(vocab_size=len(tokenizer))
    model = Transformer(config)
    logits, loss = model(input_ids, labels=input_ids)

The tokenizer's vocabulary size must match the token IDs in the .bin files.
Full logits have shape [batch, time, vocabulary]; with a large vocabulary they
can use substantial memory. Measure GPU memory before choosing a microbatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TransformerConfig:
    """Choose the model's size and attention layout.

    Args:
        vocab_size: Number of IDs in the saved tokenizer, including added tokens.
        d_model: Width of each token's hidden vector.
        n_layers: Number of attention + feed-forward blocks.
        n_heads: Number of query heads; 8 heads at width 512 means 64 values/head.
        n_kv_heads: Number of shared key/value heads. Fewer than n_heads is GQA.
        ffn_dim: Hidden width inside the SwiGLU feed-forward network.
        max_seq_len: Longest input this model accepts in one forward pass.
        rope_theta: Base used to make RoPE's rotation frequencies.
        norm_eps: Small constant preventing division by zero in RMSNorm.
        use_qk_norm: Normalize queries and keys per head before RoPE if True.
    """

    vocab_size: int
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn_dim: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        """Reject dimensions that cannot be reshaped into attention heads."""
        if min(self.vocab_size, self.d_model, self.n_layers, self.n_heads,
               self.n_kv_heads, self.ffn_dim, self.max_seq_len) <= 0:
            raise ValueError("Vocabulary and model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("Head width must be even so RoPE can rotate pairs")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")

    @property
    def head_dim(self) -> int:
        """Number of values belonging to one attention head."""
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """Scale a vector by its root mean square, then learn one scale per value.

    Args:
        width: Size of the last dimension being normalized.
        eps: Positive number added inside the square root for stability.
    """

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last dimension of ``x``, preserving its input dtype."""
        # Sum squares in float32 even when the model runs in bf16. This makes
        # the denominator more reliable without changing the model's interface.
        x_float = x.float()
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inverse_rms * self.weight).to(dtype=x.dtype)


def apply_rope(x: Tensor, position_ids: Tensor, theta: float) -> Tensor:
    """Rotate pairs in query/key heads according to token positions.

    Args:
        x: Query or key tensor of shape [batch, heads, time, head_dim].
        position_ids: Token positions of shape [batch, time]. These may reset
            to zero at document boundaries in a future packed data loader.
        theta: Base controlling how quickly different pairs rotate.

    RoPE encodes position without adding a position vector to token embeddings.
    Each adjacent pair (a, b) becomes (a*cos-b*sin, a*sin+b*cos).
    """
    head_dim = x.size(-1)
    pair_numbers = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inverse_frequencies = theta ** (-pair_numbers / head_dim)
    angles = position_ids.float()[..., None] * inverse_frequencies
    # The inserted head axis broadcasts one position across all heads.
    cosine = angles.cos()[:, None, :, :]
    sine = angles.sin()[:, None, :, :]
    even, odd = x.float()[..., ::2], x.float()[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine,
                           even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    """Let each position read earlier positions, using grouped-query heads.

    Args:
        config: Head counts, head width, RoPE settings, and QK norm choice.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps) if config.use_qk_norm else None
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps) if config.use_qk_norm else None

    def forward(
        self, x: Tensor, position_ids: Tensor, attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return attention output with the same [batch, time, width] as ``x``.

        Args:
            x: Hidden token vectors, [batch, time, d_model].
            position_ids: Positions used by RoPE, [batch, time].
            attention_mask: Optional [batch, time, time] boolean matrix. True
                means a query may read that key. Causality is added here even
                if the caller's mask accidentally allows future positions.
        """
        batch, time, _ = x.shape
        # Split the last axis into heads, then move heads before time. With
        # the default 8 query and 4 KV heads, each KV head serves 2 Q heads.
        q = self.q_proj(x).view(batch, time, self.config.n_heads, self.config.head_dim)
        k = self.k_proj(x).view(batch, time, self.config.n_kv_heads, self.config.head_dim)
        v = self.v_proj(x).view(batch, time, self.config.n_kv_heads, self.config.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, position_ids, self.config.rope_theta)
        k = apply_rope(k, position_ids, self.config.rope_theta)

        if attention_mask is None:
            # PyTorch can choose a memory-efficient attention kernel here.
            allowed = None
        else:
            causal = torch.ones(time, time, device=x.device, dtype=torch.bool).tril()
            allowed = attention_mask[:, None, :, :] & causal[None, None, :, :]

        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=allowed,
            is_causal=attention_mask is None,
            enable_gqa=self.config.n_heads != self.config.n_kv_heads,
        )
        # Restore [batch, time, d_model] before the residual connection.
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.config.d_model)
        return self.out_proj(attended)


class SwiGLUFeedForward(nn.Module):
    """Transform each position independently using a gated hidden layer.

    Args:
        config: Input/output width and the wider intermediate ``ffn_dim``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Gate one expansion of ``x`` with another, then return to d_model."""
        # SiLU(gate) decides which expanded features can pass through. The
        # multiplication is elementwise, so token positions never mix here.
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """One attention step and one feed-forward step, each with a residual path.

    Args:
        config: Sizes and attention choices shared by all model blocks.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self, x: Tensor, position_ids: Tensor, attention_mask: Tensor | None,
    ) -> Tensor:
        """Refine hidden vectors while keeping their shape unchanged."""
        # A residual adds the old representation back after each sublayer. It
        # gives information and gradients a direct path across many blocks.
        x = x + self.attention(self.attention_norm(x), position_ids, attention_mask)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    """Predict the next token from preceding tokens with a causal transformer.

    Args:
        config: Validated sizes and attention choices for this model.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        # Use the embedding matrix again for output scores (weight tying).
        # With Qwen's ~151k tokens and width 512, a second matrix would add
        # roughly 78 million parameters without changing the output shape.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize learned matrices near zero; leave RMSNorm scales at one."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        loss_chunk_size: int | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return ``(logits, loss)``; chunked loss returns ``(None, loss)``.

        Args:
            input_ids: Integer token IDs, [batch, time].
            labels: Optional target IDs, also [batch, time]. At position t the
                model predicts labels[t+1]; use -100 to ignore a target. For
                packed documents, ignore the first token of each new document
                so the previous document is not trained to predict it.
            position_ids: Optional positions [batch, time] for RoPE. Omit for
                ordinary 0..time-1 positions; reset them for document packing.
            attention_mask: Optional boolean [batch, time, time] mask. True
                permits attention. For isolated documents, set it True only
                when query and key belong to the same document.
            loss_chunk_size: If set, project this many positions at a time and
                checkpoint their cross-entropy. This avoids retaining the full
                [batch, time, vocab] logits during training, at the cost of
                recomputing each output chunk in backward. Requires labels.

        For a sequence [A, B, C], losses compare predictions after A with B,
        and after B with C. The prediction after C has no target here.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch, time = input_ids.shape
        if time == 0 or time > self.config.max_seq_len:
            raise ValueError(f"time must be between 1 and {self.config.max_seq_len}")
        if labels is not None and (labels.shape != input_ids.shape or time < 2):
            raise ValueError("labels must match input_ids and need at least 2 tokens")
        if loss_chunk_size is not None and (loss_chunk_size <= 0 or labels is None):
            raise ValueError("loss_chunk_size must be positive and requires labels")
        if position_ids is None:
            position_ids = torch.arange(time, device=input_ids.device).expand(batch, -1)
        elif position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have shape [batch, time]")
        if attention_mask is not None:
            if attention_mask.shape != (batch, time, time) or attention_mask.dtype != torch.bool:
                raise ValueError("attention_mask must be boolean [batch, time, time]")

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, position_ids, attention_mask)
        hidden = self.final_norm(hidden)
        if loss_chunk_size is not None:
            targets = labels[:, 1:]
            valid_count = targets.ne(-100).sum()
            if valid_count.item() == 0:
                raise ValueError("At least one next-token target must be unmasked")

            def chunk_loss(chunk_hidden: Tensor, chunk_targets: Tensor) -> Tensor:
                """Sum loss before dividing, so chunk boundaries cannot change it."""
                scores = F.linear(chunk_hidden, self.token_embedding.weight)
                return F.cross_entropy(
                    scores.reshape(-1, self.config.vocab_size),
                    chunk_targets.reshape(-1), ignore_index=-100, reduction="sum",
                )

            total_loss = hidden.new_zeros(())
            for start in range(0, time - 1, loss_chunk_size):
                stop = min(start + loss_chunk_size, time - 1)
                # Checkpoint recomputes this vocabulary projection in backward.
                # Only one small [batch, chunk, vocab] matrix is live at a time.
                total_loss = total_loss + checkpoint(
                    chunk_loss, hidden[:, start:stop], targets[:, start:stop],
                    use_reentrant=False,
                )
            return None, total_loss / valid_count
        logits = F.linear(hidden, self.token_embedding.weight)

        loss = None
        if labels is not None:
            # Shift only the loss targets, not the model inputs. Contiguous()
            # makes reshape work after slicing away the unmatched positions.
            predicted = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            targets = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(predicted, targets, ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Greedily append tokens, choosing the highest-scoring ID each time.

        Args:
            input_ids: Starting prompt, [batch, time].
            max_new_tokens: Number of tokens to append. This basic teaching
                method recomputes the context each time; it has no KV cache.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, time] tensor")
        result = input_ids
        for _ in range(max_new_tokens):
            # The model only accepts max_seq_len tokens, so an overlong prompt
            # keeps its most recent context for the next prediction.
            context = result[:, -self.config.max_seq_len:]
            logits, _ = self(context)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            result = torch.cat((result, next_id), dim=1)
        return result
