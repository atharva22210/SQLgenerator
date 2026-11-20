"""Small, explainable BM25 index over schema documents, with no new dependency.

Read in this order: tokenize -> BM25Index.__init__ -> search -> save/load.
The index never reads evaluation questions or reference SQL. It changes no
model weights: counters/postings are built from schema text at startup.
"""
from collections import Counter, defaultdict
import json

import math
from pathlib import Path
import re
import time
import unicodedata

from rag.common.catalog import read_catalog, searchable_text
from rag.common.storage import save_json
from rag.common.search import candidate_indices, group_by_database, make_hit

VERSION = 1
ANALYZER = "unicode_identifier_v1"


def tokenize(text: str) -> list[str]:
    """Use identical normalization for questions and schemas.

    Split customer_id and customerId into customer/id. Preserve Unicode words
    and numbers. We deliberately avoid stemming, synonyms and stopword tuning
    in this first baseline so its behavior remains easy to explain.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


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
        self.documents, self.source_sha256, self.k1, self.b = documents, source_sha256, k1, b
        self.postings = defaultdict(list)
        self.lengths = []
        self.by_db = group_by_database(documents)
        for index, doc in enumerate(documents):
            frequencies = Counter(tokenize(searchable_text(doc)))
            self.lengths.append(sum(frequencies.values()))
            # Inverted index: token -> (document index, frequency) entries.
            # Search visits matching postings, rather than every schema string.
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))
        self.avg_length = sum(self.lengths) / len(documents)
        if self.avg_length == 0:
            raise ValueError("Schema documents contain no searchable tokens")

    def configuration(self) -> dict:
        """Settings recorded alongside evaluation metrics for reproducibility."""
        return {"method": "bm25", "k1": self.k1, "b": self.b}

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
        candidates = candidate_indices(self.by_db, query, k, db_id, allowed_db_ids)
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
            hit = make_hit(doc, scores[index])
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
        payload = {"version": VERSION, "analyzer": ANALYZER, "source_sha256": self.source_sha256,
                   "k1": self.k1, "b": self.b, "documents": self.documents}
        save_json(path, payload)

    @classmethod
    def load(cls, path: Path, bundle: Path):
        """Reject incompatible or stale snapshots against the trusted catalogue."""
        payload = json.loads(path.read_text())
        docs, checksum = read_catalog(bundle)
        if (payload["version"] != VERSION or payload["analyzer"] != ANALYZER
                or payload["source_sha256"] != checksum or payload["documents"] != docs):
            raise ValueError("Stale/incompatible index; rebuild into a new index file")
        return cls(docs, checksum, k1=payload["k1"], b=payload["b"])
