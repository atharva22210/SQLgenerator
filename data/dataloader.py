"""Sample training batches from the uint32 files on the Modal Volume.

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

from model.transformer_35m import AttentionLayout


SQL_SOURCES = ("nstext2sql", "sql_context", "text_to_sql_v1")


@dataclass
class Batch:
    """One next-token batch; labels use -100 where loss should be ignored.

    ``attention_layout`` stores linear-size unpadding indices and cumulative
    sequence lengths. ``document_ids`` remains CPU metadata for inspection;
    training no longer transfers it or builds its quadratic equality matrix.
    """

    input_ids: Tensor
    labels: Tensor
    position_ids: Tensor
    document_ids: Tensor
    non_pad_tokens: int
    predicted_tokens: int
    sql_tokens: int
    attention_layout: AttentionLayout
    mode: str

    def model_inputs(self, mode: str, device: torch.device) -> dict:
        """Transfer tensors once; leave checked counts and lengths on CPU.

        CUDA copies use pinned host storage and nonblocking transfers, so the
        host can enqueue the entire microbatch without waiting for GPU work.
        ``mode`` must match construction: naive packing has one segment per
        row, while masked packing has one per document.
        """
        if mode != self.mode:
            raise ValueError("model_inputs mode must match the batch mode")

        def transfer(tensor: Tensor) -> Tensor:
            if device.type == "cuda":
                return tensor.pin_memory().to(device, non_blocking=True)
            return tensor.to(device)

        layout = self.attention_layout
        return {
            "input_ids": transfer(self.input_ids),
            "labels": transfer(self.labels),
            "position_ids": transfer(self.position_ids),
            "attention_layout": AttentionLayout(
                token_indices=transfer(layout.token_indices),
                cu_seqlens=transfer(layout.cu_seqlens),
                max_seqlen=layout.max_seqlen,
                sequence_lengths=layout.sequence_lengths,
            ),
            "valid_target_count": self.predicted_tokens,
        }


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

    def sample(self, rng: random.Random, max_length: int, *, first_slice: bool = True) -> list[int]:
        """Pick one document, or a contiguous slice if it exceeds the context.

        Training and validation keep the beginning of long documents, where
        SQL prompts/schemas occur. This deliberately excludes overlong tails.
        Set first_slice=False explicitly to sample a random contiguous slice.
        """
        # The overlap check supplies original row indices. Keep ``ends`` intact:
        # deleting an end would incorrectly merge two neighboring documents.
        eligible = getattr(self, "eligible_indices", None)
        if eligible is None:
            index = rng.randrange(len(self.ends))
        else:
            if len(eligible) == 0:
                raise ValueError("No eligible validation documents in this source")
            index = int(rng.choice(eligible))
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

    def __init__(self, directory: Path, split: str, eos_id: int, *,
                 grouped_sql_dir: Path, include_english: bool = True) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if include_english:
            self.english = DocumentFile(directory / f"english_{split}.bin", eos_id)
        self.sql = [DocumentFile(grouped_sql_dir / f"sql_{split}.bin", eos_id)]
        self.sql_weights = [source.token_count for source in self.sql]

    def sample(self, kind: str, rng: random.Random, length: int, *, first_slice: bool = True) -> list[int]:
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
    token_indices, sequence_lengths = [], []
    non_pad, predicted, sql_tokens = 0, 0, 0
    for row in range(batch_size):
        ids, row_labels, pos, doc_ids = [], [], [], []
        document_number = 0
        while len(ids) < seq_len:
            kind = validation_kind or ("sql" if rng.random() < sql_fraction else "english")
            document = corpus.sample(
                kind, rng, seq_len - len(ids), first_slice=True,
            )
            if not document:
                continue
            if kind == "sql":
                sql_tokens += len(document)
            start = len(ids)
            ids.extend(document)
            if mode == "masked_pack":
                sequence_lengths.append(len(document))
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
        token_indices.extend(range(row * seq_len, row * seq_len + real))
        if mode != "masked_pack":
            # Naive packing intentionally lets later documents read earlier
            # ones in the same row. Splitting it per document would silently
            # turn that baseline into masked packing.
            sequence_lengths.append(real)
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
    cumulative = [0]
    for length in sequence_lengths:
        cumulative.append(cumulative[-1] + length)
    # Boundaries and max length are computed from Python lists, never CUDA
    # nonzero()/max().item(), which would synchronize each microbatch.
    layout = AttentionLayout(
        token_indices=torch.tensor(token_indices, dtype=torch.long),
        cu_seqlens=torch.tensor(cumulative, dtype=torch.int32),
        max_seqlen=max(sequence_lengths),
        sequence_lengths=tuple(sequence_lengths),
    )
    return Batch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        position_ids=torch.tensor(positions, dtype=torch.long),
        document_ids=torch.tensor(document_ids, dtype=torch.long),
        non_pad_tokens=non_pad,
        predicted_tokens=predicted,
        sql_tokens=sql_tokens,
        attention_layout=layout,
        mode=mode,
    )
