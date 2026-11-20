"""BM25 + dense retrieval with reciprocal rank fusion (RRF).

Each child searches the same permitted schemas. We combine ranks because raw
BM25 and cosine scores have different scales; no generator or training is used.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

from rag.bm25 import ANALYZER, VERSION as BM25_VERSION, BM25Index
from rag.common.search import candidate_indices, make_hit
from rag.common.storage import save_json
from rag.dense import DenseIndex
from rag.dense.encoder import Encoder

VERSION = 1
# The child search APIs allow k <= 20. Keep the fusion pool independent of the
# final k, so requesting k=1 still allows agreement below a child's first hit.
RETRIEVAL_DEPTH = 20


class HybridIndex:
    """Fuse two indexes built from the identical ordered catalogue snapshot.

    bm25/dense: Existing index objects with the same documents and source hash.
    rrf_constant: Positive integer controlling rank decay; default 60. A document
        at ranks 1 and 3 receives 1/(60+1) + 1/(60+3). A missing rank adds zero.
    All search arguments and schema result fields match the child retrievers.
    """
    def __init__(self, bm25: BM25Index, dense: DenseIndex, *, rrf_constant: int = 60):
        if isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int) or rrf_constant < 1:
            raise ValueError("rrf_constant must be a positive integer")
        if bm25.source_sha256 != dense.source_sha256 or bm25.documents != dense.documents:
            raise ValueError("Hybrid children must use the same ordered catalogue and checksum")
        self.bm25, self.dense, self.rrf_constant = bm25, dense, rrf_constant
        self.documents, self.by_db = bm25.documents, bm25.by_db
        self.source_sha256 = bm25.source_sha256
        self.document_positions = {doc["document_id"]: i for i, doc in enumerate(self.documents)}

    def configuration(self) -> dict:
        """Record fusion settings and both children's settings in evaluations."""
        return {"method": "hybrid", "fusion": "reciprocal_rank_fusion",
                "rrf_constant": self.rrf_constant, "retrieval_depth": RETRIEVAL_DEPTH,
                "bm25": self.bm25.configuration(), "dense": self.dense.configuration()}

    def search(self, query: str, *, k: int = 5, db_id: str | None = None,
               allowed_db_ids: set[str] | None = None, explain: bool = False) -> dict:
        """Fuse the union of each child's top-20 results, then return top-k.

        Both children receive the same db_id and server allowlist BEFORE they
        rank. Scores are rank-based signals, not probabilities. Dense can return
        low-similarity hits without lexical matches; hybrid inherits that behavior.
        explain exposes child ranks, raw scores and their RRF contributions.
        """
        started = time.perf_counter()
        candidates = candidate_indices(self.by_db, query, k, db_id, allowed_db_ids)
        scores, explanations = defaultdict(float), defaultdict(dict)
        for name, child in (("bm25", self.bm25), ("dense", self.dense)):
            response = child.search(query, k=RETRIEVAL_DEPTH, db_id=db_id, allowed_db_ids=allowed_db_ids)
            for rank, hit in enumerate(response["results"], 1):
                identity = hit["document_id"]
                contribution = 1.0 / (self.rrf_constant + rank)
                scores[identity] += contribution
                if explain:
                    explanations[identity][name] = {"rank": rank, "score": hit["score"],
                                                      "contribution": contribution}
        ranked = sorted(scores, key=lambda identity: (-scores[identity], identity))[:k]
        results = []
        for identity in ranked:
            hit = make_hit(self.documents[self.document_positions[identity]], scores[identity])
            if explain:
                hit["explanation"] = {"metric": "reciprocal_rank_fusion", "rrf_constant": self.rrf_constant,
                                      "children": explanations[identity]}
            results.append(hit)
        return {"index_version": VERSION, "source_sha256": self.source_sha256,
                "scope": "global" if db_id is None else "database", "candidate_count": len(candidates),
                "matched_documents": len(scores), "status": "matches" if results else "no_candidates",
                "elapsed_ms": (time.perf_counter() - started) * 1000, "results": results}

    def save(self, path: Path) -> None:
        """Atomically store one self-contained snapshot without overwriting.

        Dense owns the shared documents/vectors; BM25 postings rebuild from those
        documents on load. Nesting avoids two files drifting to different schemas.
        """
        save_json(path, self.snapshot())

    def snapshot(self) -> dict:
        """JSON-ready hybrid snapshot, reusable inside graph retrieval."""
        return {"version": VERSION, "method": "hybrid", "rrf_constant": self.rrf_constant,
                "retrieval_depth": RETRIEVAL_DEPTH,
                "bm25": {"version": BM25_VERSION, "analyzer": ANALYZER,
                         **self.bm25.configuration()}, "dense": self.dense.snapshot()}

    @classmethod
    def load(cls, path: Path, bundle: Path, *, encoder: Encoder | None = None, device: str = "cpu"):
        """Verify both recipes and the catalogue, reusing saved dense vectors.

        path is a hybrid JSON snapshot; bundle is the verified catalogue folder.
        encoder optionally supplies an offline test encoder; device selects where
        the normal MiniLM query encoder runs. Loading does not re-embed tables.
        """
        return cls.from_snapshot(json.loads(path.read_text()), bundle, encoder=encoder, device=device)

    @classmethod
    def from_snapshot(cls, payload: dict, bundle: Path, *, encoder: Encoder | None = None, device: str = "cpu"):
        """Apply the same child and catalogue checks for nested snapshots."""
        bm25 = payload["bm25"]
        if (payload["version"] != VERSION or payload["method"] != "hybrid"
                or payload["retrieval_depth"] != RETRIEVAL_DEPTH or bm25["version"] != BM25_VERSION
                or bm25["analyzer"] != ANALYZER or bm25["method"] != "bm25"):
            raise ValueError("Stale/incompatible hybrid index; rebuild into a new file")
        dense = DenseIndex.from_snapshot(payload["dense"], bundle, encoder=encoder, device=device)
        lexical = BM25Index(dense.documents, dense.source_sha256, k1=bm25["k1"], b=bm25["b"])
        return cls(lexical, dense, rrf_constant=payload["rrf_constant"])
