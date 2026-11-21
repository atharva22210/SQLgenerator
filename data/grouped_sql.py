"""Build a grouped SQL split on disk before tokenization can publish any files."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import batched
import hashlib
import json
from multiprocessing import get_context
import re
import sqlite3

from sqlglot import Tokenizer
from sqlglot.errors import TokenError
from sqlglot.tokens import TokenType


_REPOSITORY_HEADER = re.compile(r"\A(?:[\ufeff \t\r\n]*<(?:reponame|filename|gh_stars)>[^\r\n]*(?:\r\n|\n|\r|$))+")


def repository_sql_text(content: str) -> str:
    """Remove leading StarCoder provenance lines, retaining the SQL body.

    StarCoder can concatenate metadata tags on one line or emit several
    leading lines. Keeping that prefix would make the same first query hash
    differently when copied into a file with another name.
    """
    return _REPOSITORY_HEADER.sub("", content).lstrip("\ufeff")


class SQLFingerprintTokenizer(Tokenizer):
    """Recognize MySQL backticks as well as standard quoted identifiers."""

    IDENTIFIERS = ['"', "`"]


def statement_hashes(sql, tokenizer):
    """Return ordered SHA-256 digests for the lexical statements in SQL.

    Normalize whitespace, comments and unquoted case while preserving quoted
    text. A semicolon inside 'a;b' is part of a token, so it cannot split a
    statement. This is a lexical check across sources, with no SQL execution.
    """
    statements, tokens = [], []
    for token in tokenizer.tokenize(sql):
        if token.token_type == TokenType.SEMICOLON:
            if tokens:
                statements.append(hashlib.sha256(json.dumps(tokens).encode()).digest())
                tokens = []
        else:
            raw = sql[token.start:token.end + 1]
            tokens.append(raw if raw[:1] in {"'", '"', "`", "["} or "STRING" in token.token_type.name
                          else token.text.lower())
    if tokens:
        statements.append(hashlib.sha256(json.dumps(tokens).encode()).digest())
    return statements


def _fingerprint_batch(kind, rows, eos_marker):
    """Format and hash a bounded batch; return one result per input row."""
    tokenizer = SQLFingerprintTokenizer()
    results = []
    for row in rows:
        if kind == "db_id":
            group, schema, question, sql = (row[key] for key in ("db_id", "schema", "question", "sql"))
            if not all(isinstance(value, str) and value.strip() for value in (group, schema, question, sql)):
                results.append("incomplete")
                continue
            # Retain definitions required to make some questions answerable.
            knowledge = (row.get("external_knowledge") or "").strip()
            context = f"\n\nAdditional context:\n{knowledge}" if knowledge else ""
            document = f"Schema:\n{schema}{context}\n\nQuestion:\n{question}\n<sql>\n{sql}\n</sql>"
        else:
            group, sql = row["max_stars_repo_name"], row["content"]
            if not isinstance(group, str) or not group.strip() or not isinstance(sql, str):
                results.append("incomplete")
                continue
            sql = document = repository_sql_text(sql)
        # The reader indexes every EOS as a boundary. Reject embedded
        # markers before deduplication so another clean copy can survive.
        if eos_marker and eos_marker in document:
            results.append("embedded_eos")
            continue
        try:
            hashes = statement_hashes(sql, tokenizer)
        except TokenError:
            hashes = []
        if hashes:
            digest = hashlib.sha256(b"".join(hashes)).digest()
            results.append((group.strip().casefold(), document, digest, set(hashes)))
        else:
            results.append("unlexable")
    return results


def _fingerprinted_rows(kind, rows, eos_marker, executor, workers):
    """Yield results in source order while limiting the number of queued batches."""
    batches = batched(rows, 32)
    if executor is None:
        for batch in batches:
            yield from _fingerprint_batch(kind, batch, eos_marker)
        return
    pending = deque()
    for batch in batches:
        pending.append(executor.submit(_fingerprint_batch, kind, batch, eos_marker))
        # The first occurrence of a duplicate wins. Reading completed jobs in
        # completion order could change the surviving DB or repo for a hash.
        if len(pending) >= 2 * workers:
            yield from pending.popleft().result()
    while pending:
        yield from pending.popleft().result()


def stage_rows(db: sqlite3.Connection, synsql, repository_sql, *, eos_marker: str | None = None,
               workers: int = 1):
    """Stage document and statement hashes from two streams into SQLite.

    workers controls the number of lexical fingerprint processes. The parent
    alone consumes the streams and writes SQLite in source order. For example,
    a slower earlier copy of a query still wins over a faster later duplicate.
    """
    if workers < 1:
        raise ValueError("Fingerprint workers must be positive")
    db.executescript("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, kind TEXT, group_id TEXT, text TEXT,
            digest BLOB UNIQUE, split TEXT
        );
        CREATE TABLE statements (document_id INTEGER, digest BLOB);
    """)
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
            if workers > 1 else nullcontext())
    report = {}
    with pool as executor:
        for kind, rows in (("db_id", synsql), ("repo_name", repository_sql)):
            counts = report[kind] = dict(read=0, duplicates=0, unlexable=0, incomplete=0, embedded_eos=0)
            for result in _fingerprinted_rows(kind, rows, eos_marker, executor, workers):
                counts["read"] += 1
                if isinstance(result, str):
                    counts[result] += 1
                else:
                    group, document, digest, hashes = result
                    cursor = db.execute(
                        "INSERT OR IGNORE INTO documents(kind, group_id, text, digest) VALUES (?, ?, ?, ?)",
                        (kind, group, document, digest),
                    )
                    if cursor.rowcount:
                        db.executemany("INSERT INTO statements VALUES (?, ?)",
                                       ((cursor.lastrowid, item) for item in hashes))
                    else:
                        counts["duplicates"] += 1
                if counts["read"] % 100_000 == 0:
                    db.commit()
                    print(f"Staged {kind}: {counts}", flush=True)
            db.commit()
            print(f"Finished staging {kind}: {counts}", flush=True)
    return report


def split_and_audit(db: sqlite3.Connection, seed: int):
    """Assign roughly 10% of retained groups per kind to validation.

    Choose groups deterministically, then discard validation documents that
    share even one statement with training. Deleting a contaminated document
    preserves the holdout assignment of the rest of its group.
    """
    db.execute("CREATE INDEX document_groups ON documents(kind, group_id)")
    db.execute("CREATE INDEX statement_documents ON statements(document_id)")
    assigned = {}
    for kind in ("db_id", "repo_name"):
        groups = [group for group, in db.execute(
            "SELECT DISTINCT group_id FROM documents WHERE kind = ?", (kind,))]
        if len(groups) < 2:
            raise ValueError(f"Need at least two deduplicated {kind} groups")
        groups.sort(key=lambda group: (hashlib.sha256(f"{seed}:{kind}:{group}".encode()).digest(), group))
        holdout = max(1, round(len(groups) / 10))
        db.execute("UPDATE documents SET split = 'train' WHERE kind = ?", (kind,))
        db.executemany("UPDATE documents SET split = 'val' WHERE kind = ? AND group_id = ?",
                       ((kind, group) for group in groups[:holdout]))
        assigned[kind] = dict(train=len(groups) - holdout, val=holdout)

    db.executescript("""
        CREATE TABLE train_hashes (digest BLOB PRIMARY KEY) WITHOUT ROWID;
        INSERT OR IGNORE INTO train_hashes
        SELECT s.digest FROM statements s JOIN documents d ON d.id = s.document_id
        WHERE d.split = 'train';
    """)
    dropped = db.execute("""
        UPDATE documents SET split = 'dropped'
        WHERE split = 'val' AND EXISTS (
            SELECT 1 FROM statements s JOIN train_hashes t ON t.digest = s.digest
            WHERE s.document_id = documents.id
        )
    """).rowcount
    overlaps = {}
    for kind in assigned:
        overlaps[kind] = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT group_id FROM documents WHERE kind = ? AND split IN ('train', 'val')
                GROUP BY group_id HAVING COUNT(DISTINCT split) > 1
            )
        """, (kind,)).fetchone()[0]
    overlaps["query_hash"] = db.execute("""
        SELECT COUNT(DISTINCT s.digest) FROM statements s
        JOIN documents d ON d.id = s.document_id
        JOIN train_hashes t ON t.digest = s.digest WHERE d.split = 'val'
    """).fetchone()[0]
    if any(overlaps.values()):
        raise ValueError(f"Train/validation overlap: {overlaps}")
    retained = {
        f"{kind}/{split}": {"groups": groups, "documents": documents}
        for kind, split, groups, documents in db.execute("""
            SELECT kind, split, COUNT(DISTINCT group_id), COUNT(*)
            FROM documents WHERE split IN ('train', 'val') GROUP BY kind, split
        """)
    }
    if len(retained) != 4:
        raise ValueError(f"Expected both source types in both splits after filtering: {retained}")
    return dict(assigned_groups=assigned, retained=retained,
                dropped_validation_documents=dropped, overlaps=overlaps)
