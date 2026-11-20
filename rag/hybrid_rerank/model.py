"""Frozen cross-encoder: jointly tokenize question and table text, return logits.

Published recipe: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
Unlike dense retrieval, a table cannot be scored independently of the question.
"""
from typing import Protocol

import numpy as np

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_LENGTH = 512


class Reranker(Protocol):
    """Score N (question,table_text) pairs as a finite float array [N]."""
    def configuration(self) -> dict: ...

    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


def reranker_configuration() -> dict:
    """Pinned inference recipe; scores are raw relevance logits, not confidence."""
    return {"model_id": MODEL_ID, "revision": REVISION, "max_length": MAX_LENGTH, "score_type": "raw_logit"}


class MiniLMCrossEncoder:
    """Load pinned frozen model on device; batch_size bounds pair-inference memory.

    First use may download weights/tokenizer. No training or new dependencies
    are needed: use the project's locked torch/transformers environment.
    """
    def __init__(self, *, device: str = "cpu", batch_size: int = 8):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("reranker batch_size must be a positive integer")
        self.device, self.batch_size = device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=False, use_safetensors=True,
        ).to(device).eval()
        self.model.requires_grad_(False)
        if self.model.config.num_labels != 1:
            raise ValueError("Expected one relevance logit per question-table pair")

    def configuration(self) -> dict:
        """Return the recipe stored with the retrieval snapshot."""
        return reranker_configuration()

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Jointly encode question/table pairs; return raw float32 [N] logits.

        Each input pair shares transformer attention. Pair length is at most
        512 tokens including special tokens; long inputs may be truncated.
        Each table gets its own pair, not one input containing all candidates.
        """
        import torch

        if not passages:
            return np.empty(0, dtype=np.float32)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = passages[start:start + self.batch_size]
                tokens = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                        max_length=MAX_LENGTH, return_tensors="pt")
                tokens = {name: value.to(self.device) for name, value in tokens.items()}
                logits = self.model(**tokens).logits[:, 0]
                batches.append(logits.cpu().float().numpy())
        return np.concatenate(batches)
