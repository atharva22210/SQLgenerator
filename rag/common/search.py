"""Shared request rules and result format for both retrieval methods."""
from collections import defaultdict
from typing import Protocol

MAX_QUERY_CHARS = 2000


class Retriever(Protocol):
    """Interface consumed by evaluation and the local HTTP service."""
    source_sha256: str
    by_db: dict[str, set[int]]

    def configuration(self) -> dict: ...

    def search(self, query: str, *, k: int = 5, db_id: str | None = None,
               allowed_db_ids: set[str] | None = None, explain: bool = False) -> dict: ...


def group_by_database(documents: list[dict]) -> dict[str, set[int]]:
    """Map database IDs to positions in the same document list used for scoring."""
    if not documents or len({d['document_id'] for d in documents}) != len(documents):
        raise ValueError("Require nonempty documents with unique IDs")
    grouped = defaultdict(set)
    for index, doc in enumerate(documents):
        grouped[doc['db_id']].add(index)
    return grouped


def candidate_indices(by_db: dict[str, set[int]], query: str, k: int,
                      db_id: str | None, allowed_db_ids: set[str] | None) -> set[int]:
    """Validate a request and return eligible document positions BEFORE top-k.

    None permits all databases for a trusted local caller; an empty allowlist
    permits none. Filtering after top-k could hide allowed hits behind private
    ones, so both retrievers must select these candidates before ranking.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise ValueError("query must be nonempty text of at most 2000 characters")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        raise ValueError("k must be an integer between 1 and 20")
    permitted = set(by_db) if allowed_db_ids is None else set(allowed_db_ids)
    if not permitted <= by_db.keys() or (db_id is not None and db_id not in permitted):
        raise ValueError("Unknown or unavailable database scope")
    scope = permitted if db_id is None else {db_id}
    return set().union(*(by_db[db] for db in scope))


def make_hit(doc: dict, score: float | None) -> dict:
    """Return schema context; None marks graph bridges without a retrieval score."""
    return {"document_id": doc["document_id"], "db_id": doc["db_id"], "table": doc["table"],
            "score": float(score) if score is not None else None,
            "schema_text": doc["text"], "schema_sql": doc["schema_sql"]}
