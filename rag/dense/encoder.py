"""Frozen MiniLM encoder using the project's existing torch/transformers stack.

Reference pooling: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
This pretrained retrieval baseline does not train our SQL generation model.
"""
from typing import Protocol

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_LENGTH = 256


class Encoder(Protocol):
    """An encoder maps B strings to a float matrix [B, embedding_dimension]."""
    def configuration(self) -> dict: ...

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray: ...


def encoder_configuration() -> dict:
    """Version the weights, token limit and pooling so queries match the index."""
    return {"model_id": MODEL_ID, "revision": REVISION, "max_length": MAX_LENGTH,
            "pooling": "attention_mask_mean_v1"}


def mean_pool(token_embeddings, attention_mask):
    """Reduce [B,T,H] to [B,H], excluding padded positions from the average.

    Padding is batching machinery, not sentence content: appending pads must
    not change a document's vector. The mask still includes special tokens,
    matching this model's published pooling recipe.
    """
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class MiniLMEncoder:
    """Load pinned pretrained weights on device (CPU by default); no training.

    First use downloads weights/tokenizer to the Hugging Face cache. Imports
    are lazy so installing only the BM25 environment never requires torch.
    """
    def __init__(self, *, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=False, use_safetensors=True,
        ).to(device).eval()
        self.model.requires_grad_(False)

    def configuration(self) -> dict:
        """Return the immutable encoding recipe saved with document vectors."""
        return encoder_configuration()

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """Encode B strings as float32 [B,384]; truncate each to 256 tokens.

        batch_size controls inference memory, not document boundaries. Schema
        vectors are computed at build time; a search encodes only its question.
        """
        import torch

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[start:start + batch_size], padding=True,
                                        truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
                tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state
                pooled = mean_pool(hidden, tokens["attention_mask"])
                batches.append(pooled.cpu().float().numpy())
        return np.concatenate(batches, axis=0)
