"""SQL-only validation for the Day 6 suite; training still mixes English/SQL."""
import json
import random

import torch
from transformers import AutoTokenizer

from data.dataloader import Corpus, DocumentFile, SQL_SOURCES, make_batch
from data.sql_overlap import filter_validation
from train_pretrain import DATA_DIR, volume, _tokenizer_path, _validate_corpus_files


def load_training_and_sql_validation():
    """Load training and filtered SQL readers, without indexing English validation."""
    volume.reload()
    metadata = json.loads((DATA_DIR / 'metadata.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(), local_files_only=True)
    if (metadata['token_dtype'] != 'uint32' or len(tokenizer) != metadata['tokenizer_vocab_size']
            or tokenizer.eos_token_id != metadata['eos_token_id'] or tokenizer.pad_token_id is None):
        raise ValueError('Tokenizer and corpus identity mismatch')
    _validate_corpus_files(DATA_DIR, metadata)
    train = Corpus(DATA_DIR, 'train', tokenizer.eos_token_id)
    validation = Corpus.__new__(Corpus)
    validation.sql = [DocumentFile(DATA_DIR / f'{name}_val.bin', tokenizer.eos_token_id) for name in SQL_SOURCES]
    filter_validation(train, validation, tokenizer)
    return tokenizer, train, validation


def evaluate_sql(model, corpus, config, pad_id, device, *, full=False):
    """Score filtered SQL only: a fixed sample periodically, all rows at the end.

    full=True visits every eligible document once. Both paths use the same
    standalone prefix/padding layout and whole-document token-mean loss.
    Loss is NOT answer-only accuracy. Return target counts with the loss.
    """
    class OneDocument:
        def __init__(self, ids):
            self.ids = ids
        def sample(self, *args, **kwargs):
            return self.ids

    def batches():
        rng = random.Random(config.seed + 20_000)
        if full:
            for source in corpus.sql:
                for index in source.eligible_indices:
                    index = int(index)
                    start = 0 if index == 0 else int(source.ends[index - 1]) + 1
                    end = min(int(source.ends[index]) + 1, start + config.seq_len)
                    yield make_batch(OneDocument(source.tokens[start:end].tolist()), mode='pad',
                                     batch_size=1, seq_len=config.seq_len, sql_fraction=1,
                                     pad_id=pad_id, rng=rng, validation_kind='sql')
        else:
            for _ in range(config.val_batches):
                yield make_batch(corpus, mode='pad', batch_size=1, seq_len=config.seq_len,
                                 sql_fraction=1, pad_id=pad_id, rng=rng, validation_kind='sql')

    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    targets = documents = 0
    try:
        with torch.no_grad():
            for batch in batches():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    _, loss = model(**batch.model_inputs('pad', device), loss_chunk_size=config.loss_chunk_size)
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
