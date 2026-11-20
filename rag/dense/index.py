"""Exact FAISS cosine retrieval: text -> normalized vectors -> ranked schemas.

FAISS IndexFlatIP scans eligible vectors; it does not cluster or approximate.
The existing JSON snapshot stays the source of vectors; no server is needed.
"""
import json
from pathlib import Path
import time

import numpy as np

from rag.common.catalog import read_catalog, searchable_text
from rag.common.search import candidate_indices, group_by_database, make_hit
from rag.common.storage import save_json
from rag.dense.encoder import Encoder, MiniLMEncoder, encoder_configuration
from rag.dense.faiss_search import FaissIndex

VERSION = 1
REPRESENTATION = "schema_fields_v1"


def normalize_rows(values, *, rows: int) -> np.ndarray:
    """Validate [rows,H] vectors and normalize each row to unit length.

    Dot products of unit vectors equal cosine similarity. Reject zero/NaN
    vectors because they cannot represent a usable cosine direction.
    """
    matrix = np.array(values, dtype=np.float32, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError("Expected embeddings with shape [documents, dimension]")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Embeddings must be finite and nonzero")
    return matrix / norms


class DenseIndex:
    """Index table documents with a frozen encoder, keeping labels out.

    documents: One schema dictionary per table, in vector row order.
    encoder: Same text-to-vector recipe for document and query encoding.
    source_sha256: Verified tables.jsonl fingerprint.
    batch_size: Number of schema texts encoded together during construction.
    embeddings: Optional [N,H] saved vectors; avoids re-encoding on load.
    """
    def __init__(self, documents: list[dict], encoder: Encoder, source_sha256: str = "", *,
                 batch_size: int = 32, embeddings=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.by_db = group_by_database(documents)
        self.documents, self.encoder, self.source_sha256 = documents, encoder, source_sha256
        self.encoder_config = dict(encoder.configuration())
        if embeddings is None:
            texts = [searchable_text(doc) for doc in documents]
            embeddings = encoder.encode(texts, batch_size=batch_size)
        self.embeddings = normalize_rows(embeddings, rows=len(documents))
        # Add in document order: FAISS vector ID i maps to self.documents[i].
        # Unit-length vectors make inner product (IP) equal cosine similarity.
        self.vector_index = FaissIndex(self.embeddings)

    def configuration(self) -> dict:
        """Record method, representation, encoder identity and vector size."""
        return {"method": "dense", "encoder": self.encoder_config,
                "representation": REPRESENTATION, "dimension": self.embeddings.shape[1],
                "vector_search": "faiss_flat_ip"}

    def search(self, query: str, *, k: int = 5, db_id: str | None = None,
               allowed_db_ids: set[str] | None = None, explain: bool = False) -> dict:
        """Rank eligible schemas by cosine; return the shared search response.

        Query/k/scope rules match BM25. Dense always ranks eligible documents,
        even when scores are zero or negative; similarity is not confidence.
        explain reports the scoring method, not invented word contributions.
        """
        started = time.perf_counter()
        candidates = sorted(candidate_indices(self.by_db, query, k, db_id, allowed_db_ids))
        results = []
        if candidates:
            vector = normalize_rows(self.encoder.encode([query]), rows=1)
            if vector.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Query and document embedding dimensions differ")
            # The selector filters inside FAISS, before ranking. IDs are global
            # document positions, so a scoped result needs no extra ID mapping.
            similarities, positions = self.vector_index.search(vector, candidates)
            # Retrieve all eligible scores to preserve document-ID tie-breaking,
            # including ties across the final k boundary. This exact baseline
            # prioritizes parity with NumPy, not approximate-search throughput.
            ranked = sorted(zip(positions, similarities),
                            key=lambda pair: (-float(pair[1]), self.documents[pair[0]]["document_id"]))[:k]
            for index, score in ranked:
                hit = make_hit(self.documents[index], score)
                if explain:
                    hit["explanation"] = {"metric": "cosine_similarity", "encoder": self.encoder_config}
                results.append(hit)
        return {"index_version": VERSION, "source_sha256": self.source_sha256,
                "scope": "global" if db_id is None else "database", "candidate_count": len(candidates),
                "matched_documents": len(candidates), "status": "matches" if results else "no_candidates",
                "elapsed_ms": (time.perf_counter() - started) * 1000, "results": results}

    def save(self, path: Path) -> None:
        """Save vectors and their source/encoder identity as new atomic JSON.

        JSON is readable and avoids pickle; it is larger than binary vector
        storage. Use this format for the small baseline, not a large warehouse.
        """
        save_json(path, self.snapshot())

    def snapshot(self) -> dict:
        """Return a JSON-ready snapshot, also reusable inside a hybrid index."""
        return {"version": VERSION, **self.configuration(),
                "source_sha256": self.source_sha256, "documents": self.documents,
                "embeddings": self.embeddings.tolist()}

    @classmethod
    def load(cls, path: Path, bundle: Path, *, encoder: Encoder | None = None, device: str = "cpu"):
        """Verify catalogue/recipe before loading vectors; never re-embed tables.

        encoder is injectable for offline correctness tests. Normal CLI loads
        construct the pinned MiniLM encoder on device for subsequent queries.
        """
        return cls.from_snapshot(json.loads(path.read_text()), bundle, encoder=encoder, device=device)

    @classmethod
    def from_snapshot(cls, payload: dict, bundle: Path, *, encoder: Encoder | None = None,
                      device: str = "cpu"):
        """Validate a decoded snapshot against bundle; reuse its [N,H] vectors.

        Both dense and hybrid loaders pass through these same catalogue and
        encoder checks so nesting an index does not bypass stale-data rejection.
        """
        docs, checksum = read_catalog(bundle)
        expected_encoder = encoder.configuration() if encoder is not None else encoder_configuration()
        if (payload["version"] != VERSION or payload["method"] != "dense"
                or payload["representation"] != REPRESENTATION or payload["encoder"] != expected_encoder
                or payload.get("vector_search", "faiss_flat_ip") != "faiss_flat_ip"
                or payload["source_sha256"] != checksum or payload["documents"] != docs):
            raise ValueError("Stale/incompatible dense index; rebuild into a new file")
        vectors = normalize_rows(payload["embeddings"], rows=len(docs))
        if vectors.shape[1] != payload["dimension"]:
            raise ValueError("Saved embedding dimension disagrees with metadata")
        return cls(docs, encoder if encoder is not None else MiniLMEncoder(device=device),
                   checksum, embeddings=vectors)
