"""Evaluate audited, held-out SQL documents from the pretraining corpus."""
from pathlib import Path
import random

import torch

from data.dataloader import make_batch
from training.config import _load_data_and_tokenizer


def load_training_and_sql_validation(grouped_sql_dir: Path | None = None, *, reload_volume: bool = True):
    """Open the training mix and the held-out Grouped SQL split for distributed training."""
    return _load_data_and_tokenizer(grouped_sql_dir, reload_volume=reload_volume,
                                    sql_only_validation=True)


def evaluate_sql(model, corpus, config, pad_id, device, *, full=False):
    """Score filtered SQL only: a fixed sample periodically, all rows at the end.

    Args:
        model: Unwrapped Transformer on rank 0, not the DDP wrapper.
        corpus: SQL-only reader whose eligible_indices were filtered first.
        config: TrainConfig; supplies seed, sequence length, sample count and
            vocabulary-loss chunk size.
        pad_id: Saved tokenizer's PAD ID; PAD targets get label -100.
        device: Rank 0's torch.device, used for inputs and loss accumulation.
        full: False samples config.val_batches documents reproducibly; True
            visits each retained document exactly once, without replacement.
    Returns:
        Scalar loss plus document and target counts. Full-set keys end with
        _filtered_all so they cannot be mistaken for the smaller sampled pass.

    full=True visits every eligible document once. Both paths use the same
    standalone prefix/padding layout and whole-document token-mean loss.
    Loss is NOT answer-only accuracy. Return target counts with the loss.
    """
    # Adapter for make_batch: return one already selected document instead of
    # drawing randomly. This reuses its tested padding, label shift, valid-token
    # counts and attention-layout construction in the full validation pass.
    class OneDocument:
        def __init__(self, ids):
            """ids is one complete document or its context-length prefix."""
            self.ids = ids
        def sample(self, *args, **kwargs):
            return self.ids

    def batches():
        """Yield one document per row; no packing across validation examples."""
        rng = random.Random(config.seed + 20_000)
        if full:
            for source in corpus.sql:
                for index in source.eligible_indices:
                    index = int(index)
                    # ends contains INCLUSIVE original EOS positions. Keep it
                    # intact even when filtering removes neighboring examples;
                    # end+1 turns it into Python's exclusive slicing boundary.
                    start = 0 if index == 0 else int(source.ends[index - 1]) + 1
                    # Long examples use only their prefix, matching training's
                    # current loader convention. "All documents" therefore
                    # does not mean scoring every token of overlong examples.
                    end = min(int(source.ends[index]) + 1, start + config.seq_len)
                    yield make_batch(OneDocument(source.tokens[start:end].tolist()), mode='pad',
                                     batch_size=1, seq_len=config.seq_len, sql_fraction=1,
                                     pad_id=pad_id, rng=rng, validation_kind='sql')
        else:
            for _ in range(config.val_batches):
                yield make_batch(corpus, mode='pad', batch_size=1, seq_len=config.seq_len,
                                 sql_fraction=1, pad_id=pad_id, rng=rng, validation_kind='sql')

    # eval() controls layer behavior; no_grad() independently disables graph
    # recording. Save the old mode so the helper works during training and
    # standalone evaluation, including when an exception occurs.
    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    targets = documents = 0
    try:
        with torch.no_grad():
            for batch in batches():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    _, loss = model(**batch.model_inputs('pad', device), loss_chunk_size=config.loss_chunk_size)
                # A mean of document means would overweight short documents.
                # Convert back to a loss sum, then divide by total valid targets
                # once after the pass. Padding never enters that denominator.
                total += loss.detach() * batch.predicted_tokens
                targets += batch.predicted_tokens
                documents += 1
        if targets == 0:
            raise ValueError('No valid SQL evaluation targets')
        suffix = '_filtered_all' if full else ''
        return {f'val/sql_loss{suffix}': float(total / targets),
                f'val/sql_documents{suffix}': documents, f'val/sql_targets{suffix}': targets}
    finally:
        model.train(was_training)
