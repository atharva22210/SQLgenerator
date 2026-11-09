"""A small, configurable autoregressive PyTorch language model."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings; choose tokenization and training data separately."""

    vocab_size: int = 32_000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_sequence_length: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.max_sequence_length,
        ) < 1:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CausalLanguageModel(nn.Module):
    """Token-level decoder model suitable for text-to-SQL fine-tuning."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.normalization = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return logits and optional next-token cross-entropy loss."""
        self._validate_inputs(input_ids, attention_mask, labels)
        _, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = attention_mask == 0 if attention_mask is not None else None
        hidden_states = self.transformer(
            hidden_states, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        logits = self.output(self.normalization(hidden_states))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Greedily append token IDs without imposing a tokenizer or SQL dialect."""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch_size, sequence_length)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_sequence_length :]
            logits, _ = self(context)
            generated = torch.cat((generated, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        return generated

    def _validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        labels: Tensor | None,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, sequence_length)")
        if not 0 < input_ids.shape[1] <= self.config.max_sequence_length:
            raise ValueError("sequence length is outside the configured limit")
        for name, value in (("attention_mask", attention_mask), ("labels", labels)):
            if value is not None and value.shape != input_ids.shape:
                raise ValueError(f"{name} must have the same shape as input_ids")
