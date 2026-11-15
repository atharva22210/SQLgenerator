"""Conservatively exclude repeated SQL tasks before validation sampling.

This checks complete documents, before context cropping. Question/answer
matching deliberately ignores schema differences: ambiguous matches are held
out of evaluation rather than allowed to inflate its score. It does not detect
paraphrases or semantically equivalent SQL with different wording.
"""
import hashlib
import re
import unicodedata


def task_key(text: str, source: str) -> bytes:
    """Fingerprint a formatted question and SQL answer; reject unknown formats."""
    prompt, marker, response = text.rpartition('<sql>')
    answer, closing, _ = response.partition('</sql>')
    if source == 'nstext2sql':
        comments = [line.lstrip()[2:].strip() for line in prompt.splitlines()
                    if line.lstrip().startswith('--')]
        question = comments[-1] if comments else ''
    else:
        _, question_marker, question = prompt.rpartition('Question:\n')
        if not question_marker:
            question = ''
    if not marker or not closing or not question.strip() or not answer.strip():
        raise ValueError('Cannot extract complete SQL task')
    def normalize(value):
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value).casefold()).strip()
    return hashlib.sha256((normalize(question) + '\0' + normalize(answer)).encode()).digest()


def filter_validation(train, validation, tokenizer):
    """Restrict validation to unique tasks absent from every SQL training file.

    Arguments are loaded Corpus objects and their saved tokenizer. Original
    token files stay intact; eligible row indices control subsequent sampling.
    Unknown validation formats are excluded and counted, never silently scored.
    """
    import numpy as np
    from data.dataloader import SQL_SOURCES

    def rows(source):
        start = 0
        for index, last in enumerate(source.ends):
            end = int(last) + 1
            yield index, tokenizer.decode(source.tokens[start:end].tolist(),
                                          skip_special_tokens=False), end - start
            start = end

    seen = set()
    uncertain_answers = set()

    def answer_key(text):
        _, marker, response = text.rpartition("<sql>")
        answer, closing, _ = response.partition("</sql>")
        if not marker or not closing or not answer.strip():
            raise ValueError("Training document has no identifiable SQL answer")
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", answer).casefold()).strip()
    for name, source in zip(SQL_SOURCES, train.sql):
        for _, text, _ in rows(source):
            try:
                seen.add(task_key(text, name))
            except ValueError:
                # Some source rows have an empty question. Conservatively
                # exclude validation tasks sharing their SQL answer.
                uncertain_answers.add(answer_key(text))
        print(f"Indexed SQL training tasks from {name}", flush=True)
    validation_seen = set()
    report = {}
    for name, source in zip(SQL_SOURCES, validation.sql):
        eligible, tokens = [], 0
        counts = dict(total=len(source.ends), training_overlap=0, repeated_validation=0, unparsed=0)
        for index, text, length in rows(source):
            try:
                key = task_key(text, name)
            except ValueError:
                counts['unparsed'] += 1
                continue
            if key in seen or answer_key(text) in uncertain_answers:
                counts['training_overlap'] += 1
            elif key in validation_seen:
                counts['repeated_validation'] += 1
            else:
                eligible.append(index)
                tokens += length
                validation_seen.add(key)
        source.eligible_indices = np.asarray(eligible, dtype=np.int64)
        counts['retained'] = len(eligible)
        report[name] = counts
        source.eligible_token_count = tokens
    validation.sql_weights = [source.eligible_token_count for source in validation.sql]
    if not sum(validation.sql_weights):
        raise ValueError('No uncontaminated SQL validation documents remain')
    print(f'SQL validation overlap filter: {report}', flush=True)
    return report
