"""Small, explainable BM25 index over schema documents, with no new dependency.

Read in this order: tokenize -> BM25Index.__init__ -> search -> save/load.
The index never reads evaluation questions or reference SQL. It changes no
model weights: counters/postings are built from schema text at startup.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata

VERSION = 1
ANALYZER = "unicode_identifier_v1"
MAX_QUERY_CHARS = 2000


def tokenize(text: str) -> list[str]:
    """Use identical normalization for questions and schemas.

    Split customer_id and customerId into customer/id. Preserve Unicode words
    and numbers. We deliberately avoid stemming, synonyms and stopword tuning
    in this first baseline so its behavior remains easy to explain.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_catalog(bundle: Path) -> tuple[list[dict], str]:
    """Verify the schema file against the completed bundle's checksum.

    A checksum detects corruption/staleness, not malicious replacement of both
    data and manifest. Load bundles from our trusted storage, not user uploads.
    Only the explicit catalog path is opened; labels/ is never scanned.
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    path = bundle / "catalog/tables.jsonl"
    checksum = sha256(path)
    if not manifest.get("complete") or checksum != manifest["files"]["catalog/tables.jsonl"]["sha256"]:
        raise ValueError("Incomplete or corrupt schema catalogue")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != manifest["counts"]["tables"]:
        raise ValueError("Catalogue count disagrees with manifest")
    return rows, checksum


def searchable_text(doc: dict) -> str:
    """Use schema fields, excluding opaque database IDs and reference answers.

    Repeated template words like 'Columns' and 'none declared' are omitted.
    Domain descriptions help lexical matching but may also add noise; this
    fixed representation is recorded in the index version for reproducibility.
    """
    parts = [doc["table"], doc["domain"], doc["domain_description"]]
    parts += [f"{column['name']} {column['type']} {column.get('comment', '')}" for column in doc["columns"]]
    parts += doc["primary_key"]
    for key in doc["foreign_keys"]:
        parts += key["columns"] + [key["referenced_table"]] + key["referenced_columns"]
    return " ".join(parts)


class BM25Index:
    """Immutable-by-convention schema snapshot; build once, search repeatedly.

    Args:
        documents: Table records, not question/answer labels.
        source_sha256: Fingerprint of the verified source tables.jsonl.
        k1: Positive saturation parameter; repetitions have diminishing returns.
        b: Length normalization strength in [0,1]; zero ignores document length.
    Scores are ranking signals, NOT probabilities or calibrated confidence.
    """
    def __init__(self, documents: list[dict], source_sha256: str = "", *, k1: float = 1.2, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1")
        if not documents or len({d["document_id"] for d in documents}) != len(documents):
            raise ValueError("Require nonempty documents with unique IDs")
        self.documents, self.source_sha256, self.k1, self.b = documents, source_sha256, k1, b
        self.postings = defaultdict(list)
        self.lengths = []
        self.by_db = defaultdict(set)
        for index, doc in enumerate(documents):
            frequencies = Counter(tokenize(searchable_text(doc)))
            self.lengths.append(sum(frequencies.values()))
            self.by_db[doc["db_id"]].add(index)
            # Inverted index: token -> (document index, frequency) entries.
            # Search visits matching postings, rather than every schema string.
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))
        self.avg_length = sum(self.lengths) / len(documents)
        if self.avg_length == 0:
            raise ValueError("Schema documents contain no searchable tokens")

    def search(self, query: str, *, k: int = 5, db_id: str | None = None,
               allowed_db_ids: set[str] | None = None, explain: bool = False) -> dict:
        """Return ranked schemas and request diagnostics; never execute SQL.

        query: User question, 1..2000 characters. k: maximum results, 1..20.
        db_id: Optional application-selected database; None searches globally.
        allowed_db_ids: Server-controlled permitted databases, applied BEFORE
            ranking. None means this trusted local caller can search all data.
        explain: Include each matched term's contribution for learning/debugging.
        Empty lexical matches return no results; arbitrary zero-score schemas
        are never used to pad the response. Ties use document IDs deterministically.
        """
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ValueError("query must be nonempty text of at most 2000 characters")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("k must be an integer between 1 and 20")
        permitted = set(self.by_db) if allowed_db_ids is None else set(allowed_db_ids)
        if not permitted <= self.by_db.keys() or (db_id is not None and db_id not in permitted):
            raise ValueError("Unknown or unavailable database scope")
        scope = permitted if db_id is None else {db_id}
        candidates = set().union(*(self.by_db[db] for db in scope))
        terms = sorted(set(tokenize(query)))  # Repeating query words does not boost scores.
        scores, contributions = defaultdict(float), defaultdict(dict)
        for term in terms:
            postings = self.postings.get(term, [])
            # Corpus statistics stay GLOBAL even for scoped searches; the same
            # question/doc pair keeps the same score under an authorization filter.
            df = len(postings)
            idf = math.log1p((len(self.documents) - df + 0.5) / (df + 0.5))
            for index, tf in postings:
                if index not in candidates:
                    continue
                length_norm = 1 - self.b + self.b * self.lengths[index] / self.avg_length
                contribution = idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
                scores[index] += contribution
                if explain:
                    contributions[index][term] = contribution
        ranked = sorted(scores, key=lambda i: (-scores[i], self.documents[i]["document_id"]))[:k]
        results = []
        for index in ranked:
            doc = self.documents[index]
            hit = {"document_id": doc["document_id"], "db_id": doc["db_id"], "table": doc["table"],
                   "score": scores[index], "schema_text": doc["text"], "schema_sql": doc["schema_sql"]}
            if explain:
                hit["term_scores"] = contributions[index]
            results.append(hit)
        return {"index_version": VERSION, "source_sha256": self.source_sha256,
                "scope": "global" if db_id is None else "database", "candidate_count": len(candidates),
                "matched_documents": len(scores), "status": "matches" if results else "no_lexical_match",
                "elapsed_ms": (time.perf_counter() - started) * 1000, "results": results}

    def save(self, path: Path) -> None:
        """Atomically persist a JSON snapshot; existing indexes are not overwritten.

        JSON avoids executing deserialization code (unlike pickle). Postings
        rebuild from the snapshot on load; no labels or credentials are stored.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": VERSION, "analyzer": ANALYZER, "source_sha256": self.source_sha256,
                   "k1": self.k1, "b": self.b, "documents": self.documents}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump(payload, file, ensure_ascii=False, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            # Atomic no-clobber publication on the same filesystem: readers
            # see either the complete file or no file. Never an unfinished JSON.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, bundle: Path):
        """Reject incompatible or stale snapshots against the trusted catalogue."""
        payload = json.loads(path.read_text())
        docs, checksum = read_catalog(bundle)
        if (payload["version"] != VERSION or payload["analyzer"] != ANALYZER
                or payload["source_sha256"] != checksum or payload["documents"] != docs):
            raise ValueError("Stale/incompatible index; rebuild into a new index file")
        return cls(docs, checksum, k1=payload["k1"], b=payload["b"])
