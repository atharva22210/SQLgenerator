"""Generate judge-ready SQL completions from a shared trainer final checkpoint."""
from pathlib import Path
import random

import torch

from data.dataloader import DocumentFile
from data.prepare_pretrain_data import tokenizer_fingerprint
from model.transformer_35m import Transformer, TransformerConfig
from training.dataset import dataset_identity


def split_document(text: str):
    """Return the training-format prompt and reference, or None for ineligible rows.

    Require unique field boundaries. In particular, a second opening SQL marker
    inside an answer must never move part of that answer into the model prompt.
    """
    question_marker = '\n\nQuestion:\n'
    answer_marker = '\n<sql>\n'
    if (not text.startswith('Schema:\n') or not text.endswith('\n</sql>')
            or text.count(question_marker) != 1 or text.count(answer_marker) != 1):
        return None
    before, _, reference = text.removesuffix('\n</sql>').partition(answer_marker)
    context, _, question = before.partition(question_marker)
    if not (context.removeprefix('Schema:\n').strip() and question.strip() and reference.strip()):
        return None
    return before + answer_marker, reference


def select_examples(documents, tokenizer, count, seed, prompt_limit):
    """Sample distinct validation documents without replacement within the token limit.

    Each end offset points to EOS. Decode the complete document, split the
    text, then encode the prompt alone so BPE cannot merge across the boundary.
    """
    indices = list(range(len(documents.ends)))
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        start = int(documents.ends[index - 1]) + 1 if index else 0
        ids = documents.tokens[start:int(documents.ends[index])].tolist()
        example = split_document(tokenizer.decode(ids, clean_up_tokenization_spaces=False))
        if example is None:
            continue
        prompt, reference = example
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) <= prompt_limit:
            selected.append((index, prompt, reference, prompt_ids))
        if len(selected) == count:
            return selected
    raise ValueError(f'Requested {count} examples but only {len(selected)} eligible prompts fit')


@torch.inference_mode()
def greedy_completion(model, tokenizer, prompt_ids, max_new_tokens, device):
    """Decode one unpadded prompt, returning generated text and termination metadata."""
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    reason = 'token_limit'
    raw = ''
    for _ in range(max_new_tokens):
        context = model.generate(context, max_new_tokens=1)
        next_id = context[0, -1].item()
        generated.append(next_id)
        raw = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
        # Search decoded text: the closing tag can span several BPE tokens.
        if '</sql>' in raw:
            reason = 'closing_tag'
            break
        if next_id == tokenizer.eos_token_id:
            reason = 'eos'
            break
    candidate = (tokenizer.decode(generated[:-1], clean_up_tokenization_spaces=False)
                 if reason == 'eos' else raw.partition('</sql>')[0])
    return dict(generated_sql=candidate, raw_completion=raw, stop_reason=reason,
                generated_tokens=len(generated))


def evaluate(checkpoint: Path, data_dir: Path, tokenizer, count: int, seed: int,
             max_new_tokens: int, device: str):
    """Return one judge record per held-out prompt using the checkpoint's model and corpus.

    checkpoint is a completed shared-trainer .pt file. data_dir and tokenizer
    must identify its published build. count and seed select validation rows;
    max_new_tokens reserves answer space and bounds greedy decoding on device.
    """
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if state['update'] != state['config']['max_steps']:
        raise ValueError('Expected a final checkpoint at max_steps')
    identity = dataset_identity(data_dir, Path(state['grouped_sql_dir']))
    if state.get('dataset_identity') != identity or tokenizer_fingerprint(tokenizer) != identity['tokenizer_sha256']:
        raise ValueError('Checkpoint dataset or tokenizer identity mismatch')
    config = TransformerConfig(**state['model_config'])
    if not (count > 0 and 0 < max_new_tokens < config.max_seq_len):
        raise ValueError('Choose a positive count and a generation limit smaller than the context')
    documents = DocumentFile(Path(identity['sql_dir']) / 'sql_val.bin', tokenizer.eos_token_id)
    examples = select_examples(documents, tokenizer, count, seed, config.max_seq_len - max_new_tokens)
    model = Transformer(config)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    metadata = dict(checkpoint=str(checkpoint), update=state['update'], training_run_id=state['wandb_id'],
                    corpus_id=identity['corpus_id'], manifest_sha256=identity['manifest_sha256'],
                    selection_seed=seed, max_new_tokens=max_new_tokens)
    del state
    rows = []
    for index, prompt, reference, ids in examples:
        completion = greedy_completion(model, tokenizer, ids, max_new_tokens, device)
        rows.append(dict(**metadata, validation_document=index, prompt=prompt,
                         prompt_tokens=len(ids), reference_sql=reference, **completion))
    return rows
