"""Load trustworthy token files and evaluate filtered SQL documents only.

The data-preparation job already tokenized the corpus. This module does NOT
retokenize it: the tokenizer supplies vocabulary/special-token identity and
text decoding for overlap matching. Corpus/DocumentFile use read-only memmaps.

Filtering stores eligible original document indices in memory, preserving EOS
boundaries. It does not rewrite the .bin files or prove semantic independence:
matching is the conservative normalized question+SQL rule in data/sql_overlap.
Training continues mixing English and SQL; ONLY validation is SQL-only here.
"""
import json
import random

import torch
from transformers import AutoTokenizer

from data.dataloader import Corpus, DocumentFile, SQL_SOURCES, make_batch
from data.sql_overlap import filter_validation
from train_pretrain import DATA_DIR, volume, _tokenizer_path, _validate_corpus_files


def load_training_and_sql_validation():
    """Return (tokenizer, training Corpus, filtered SQL-only Corpus).

    Called by rank 0 after Modal mounts sqlgen-data at /vol. There are no
    arguments because these experiment paths come from train_pretrain.
    Completion metadata checks all file sizes/progress markers before reading.
    The returned validation object intentionally has no English file reader:
    callers must request SQL, as evaluate_sql does.
    """
    volume.reload()
    metadata = json.loads((DATA_DIR / 'metadata.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(), local_files_only=True)
    if (metadata['token_dtype'] != 'uint32' or len(tokenizer) != metadata['tokenizer_vocab_size']
            or tokenizer.eos_token_id != metadata['eos_token_id'] or tokenizer.pad_token_id is None):
        raise ValueError('Tokenizer and corpus identity mismatch')
    _validate_corpus_files(DATA_DIR, metadata)
    # A full training reader is required both for training and for comparing
    # SQL validation tasks against ALL SQL training sources, not just the
    # source from which a particular validation example originated.
    train = Corpus(DATA_DIR, 'train', tokenizer.eos_token_id)
    # __new__ creates the container without Corpus.__init__, which would also
    # scan English validation. We deliberately populate only the SQL fields
    # used by Corpus.sample(kind='sql'). This restricted object must not be
    # passed to a function that expects an English validation reader.
    validation = Corpus.__new__(Corpus)
    validation.sql = [DocumentFile(DATA_DIR / f'{name}_val.bin', tokenizer.eos_token_id) for name in SQL_SOURCES]
    filter_validation(train, validation, tokenizer)
    return tokenizer, train, validation


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
