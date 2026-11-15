"""Sample training batches from the eight uint32 files on the Modal Volume.

The preparation job wrote each document as token IDs followed by EOS. The
files contain no row table, so this loader finds EOS positions once at startup
and uses those positions as a lightweight document index. It never loads all
token IDs into Python objects: ``numpy.memmap`` reads the needed slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import Tensor


SQL_SOURCES = ("nstext2sql", "sql_context", "text_to_sql_v1")


@dataclass
class Batch:
    """One next-token batch; labels use -100 where loss should be ignored.

    ``document_ids`` is only used by masked packing. Its equality matrix is
    the document-isolated attention mask, shape [batch, time, time].
    """

    input_ids: Tensor
    labels: Tensor
    position_ids: Tensor
    document_ids: Tensor
    non_pad_tokens: int
    predicted_tokens: int
    sql_tokens: int

    def model_inputs(self, mode: str, device: torch.device) -> dict:
        """Move the batch and make a mask only for isolated packing."""
        inputs = {
            "input_ids": self.input_ids.to(device),
            "labels": self.labels.to(device),
            "position_ids": self.position_ids.to(device),
        }
        if mode == "masked_pack":
            ids = self.document_ids.to(device)
            # True permits attention; the transformer also adds causality.
            inputs["attention_mask"] = ids[:, :, None] == ids[:, None, :]
        return inputs


class DocumentFile:
    """Read a token stream and index the EOS-delimited documents in it.

    Args:
        path: A complete ``*_train.bin`` or ``*_val.bin`` file.
        eos_id: Token ID appended after every source row by preparation.
    """

    def __init__(self, path: Path, eos_id: int) -> None:
        if not path.is_file() or path.stat().st_size % 4:
            raise ValueError(f"Missing or malformed uint32 token file: {path}")
        self.tokens = np.memmap(path, mode="r", dtype="<u4")
        if len(self.tokens) == 0 or int(self.tokens[-1]) != eos_id:
            raise ValueError(f"{path} must be complete and end with EOS {eos_id}")
        # Scan in chunks so even a large corpus needs only a small temporary
        # comparison array. Each end is inclusive, and the next start is +1.
        ends = []
        for start in range(0, len(self.tokens), 1_000_000):
            chunk = self.tokens[start:start + 1_000_000]
            ends.append(np.flatnonzero(chunk == eos_id).astype(np.int64) + start)
        self.ends = np.concatenate(ends)
        if len(self.ends) == 0:
            raise ValueError(f"No EOS-delimited documents in {path}")

    def sample(self, rng: random.Random, max_length: int, *, first_slice: bool = False) -> list[int]:
        """Pick one document, or a contiguous slice if it exceeds the context.

        Long documents are sliced for a fixed context; validation uses the
        first slice to keep its examples repeatable across batching modes.
        """
        index = rng.randrange(len(self.ends))
        start = 0 if index == 0 else int(self.ends[index - 1]) + 1
        end = int(self.ends[index]) + 1
        if end - start > max_length:
            start = start if first_slice else rng.randrange(start, end - max_length + 1)
            end = start + max_length
        return self.tokens[start:end].astype(np.int64).tolist()

    @property
    def token_count(self) -> int:
        return len(self.tokens)


class Corpus:
    """Keep English and SQL files separate so their sampling mix is explicit."""

    def __init__(self, directory: Path, split: str, eos_id: int) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.english = DocumentFile(directory / f"english_{split}.bin", eos_id)
        self.sql = [DocumentFile(directory / f"{name}_{split}.bin", eos_id) for name in SQL_SOURCES]
        self.sql_weights = [source.token_count for source in self.sql]

    def sample(self, kind: str, rng: random.Random, length: int, *, first_slice: bool = False) -> list[int]:
        if kind == "english":
            source = self.english
        elif kind == "sql":
            # Weight by token count so a tiny SQL source is not repeated as
            # often as a much larger one. The English/SQL ratio stays separate.
            source = rng.choices(self.sql, weights=self.sql_weights, k=1)[0]
        else:
            raise ValueError("kind must be english or sql")
        return source.sample(rng, length, first_slice=first_slice)


def make_batch(
    corpus: Corpus,
    *,
    mode: str,
    batch_size: int,
    seq_len: int,
    sql_fraction: float,
    pad_id: int,
    rng: random.Random,
    validation_kind: str | None = None,
) -> Batch:
    """Build ``pad``, ``naive_pack``, or ``masked_pack`` examples.

    Args:
        corpus: Memory-mapped train or validation streams.
        mode: ``pad`` uses one document per row. ``naive_pack`` fills rows with
            multiple documents under ordinary causal attention. ``masked_pack``
            also resets RoPE positions and isolates attention by document.
        batch_size: Number of rows in one forward pass.
        seq_len: Token slots per row, including any padding.
        sql_fraction: Probability that a sampled *document* is SQL, in [0, 1].
            The default 0.5 aims for an English/SQL document mix, not an exact
            token mix; different document lengths can change the token ratio.
        pad_id: Token ID for unused slots; padding targets are ignored.
        rng: Explicit random generator so experiment seeds can be held fixed.
        validation_kind: Force all sampled documents to English or SQL.

    At a boundary [A, EOS, B], the prediction after EOS must not learn B. In
    masked_pack we set B's target to -100; naive_pack intentionally leaves it
    as the baseline whose cross-document leakage we want to measure.
    """
    if mode not in {"pad", "naive_pack", "masked_pack"}:
        raise ValueError("mode must be pad, naive_pack, or masked_pack")
    if batch_size <= 0 or seq_len < 2 or not 0 <= sql_fraction <= 1:
        raise ValueError("batch_size > 0, seq_len >= 2, and 0 <= sql_fraction <= 1 required")
    if validation_kind not in {None, "english", "sql"}:
        raise ValueError("validation_kind must be english, sql, or None")

    rows, labels, positions, document_ids = [], [], [], []
    non_pad, predicted, sql_tokens = 0, 0, 0
    for _ in range(batch_size):
        ids, row_labels, pos, doc_ids = [], [], [], []
        document_number = 0
        while len(ids) < seq_len:
            kind = validation_kind or ("sql" if rng.random() < sql_fraction else "english")
            document = corpus.sample(
                kind, rng, seq_len - len(ids), first_slice=validation_kind is not None,
            )
            if not document:
                continue
            if kind == "sql":
                sql_tokens += len(document)
            start = len(ids)
            ids.extend(document)
            row_labels.extend(document)
            pos.extend(range(len(document)) if mode == "masked_pack" else range(start, len(ids)))
            doc_ids.extend([document_number] * len(document))
            if mode == "masked_pack" and start > 0:
                # labels[t] is predicted from input[t-1], so masking this
                # first target removes the cross-document transition.
                row_labels[start] = -100
            document_number += 1
            if mode == "pad" or validation_kind is not None:
                break
        real = len(ids)
        non_pad += real
        pad_count = seq_len - real
        ids.extend([pad_id] * pad_count)
        row_labels.extend([-100] * pad_count)
        pos.extend(range(real, seq_len))
        # Give padding a separate segment; padded positions do not enter loss.
        doc_ids.extend([-1] * pad_count)
        predicted += sum(value != -100 for value in row_labels[1:])
        rows.append(ids)
        labels.append(row_labels)
        positions.append(pos)
        document_ids.append(doc_ids)

    if predicted == 0:
        raise ValueError("Batch contains no next-token targets; check the corpus and seq_len")
    return Batch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        position_ids=torch.tensor(positions, dtype=torch.long),
        document_ids=torch.tensor(document_ids, dtype=torch.long),
        non_pad_tokens=non_pad,
        predicted_tokens=predicted,
        sql_tokens=sql_tokens,
    )
