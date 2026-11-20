"""Rerank a small plain-hybrid candidate pool without adding new tables."""
import json
from pathlib import Path
import time

import numpy as np

from rag.common.catalog import searchable_text
from rag.common.search import candidate_indices
from rag.common.storage import save_json
from rag.dense.encoder import Encoder
from rag.hybrid import HybridIndex
from rag.hybrid_rerank.model import MiniLMCrossEncoder, Reranker, reranker_configuration

VERSION = 1


class HybridRerankIndex:
    """Hybrid candidate retrieval followed by joint question/table scoring.

    hybrid: Verified plain hybrid index. reranker: Frozen pair-scoring model.
    candidate_pool: Number of permitted hybrid candidates sent to reranking
        (1..20; default 8). Final k must not exceed this pool size.
    """
    def __init__(self, hybrid: HybridIndex, reranker: Reranker, *, candidate_pool: int = 8):
        if isinstance(candidate_pool, bool) or not isinstance(candidate_pool, int) or not 1 <= candidate_pool <= 20:
            raise ValueError("candidate_pool must be an integer in [1,20]")
        self.hybrid, self.reranker, self.candidate_pool = hybrid, reranker, candidate_pool
        self.documents, self.by_db, self.source_sha256 = hybrid.documents, hybrid.by_db, hybrid.source_sha256
        self.lookup = {d["document_id"]: d for d in self.documents}

    def configuration(self) -> dict:
        """Record candidate budget and pinned reranker recipe for evaluation."""
        return {"method": "hybrid_rerank", "candidate_pool": self.candidate_pool,
                "reranker": self.reranker.configuration(), "hybrid": self.hybrid.configuration()}

    def search(self, query: str, *, k: int = 4, db_id: str | None = None,
               allowed_db_ids: set[str] | None = None, explain: bool = False) -> dict:
        """Return top-k cross-encoder results from a permission-filtered pool.

        Returned score is a raw cross-encoder logit, replacing the hybrid score.
        Reranking cannot recover a table absent from the hybrid candidate pool.
        Evaluation must score this final order, not the initial candidate order.
        """
        started = time.perf_counter()
        candidate_indices(self.by_db, query, k, db_id, allowed_db_ids)
        if k > self.candidate_pool:
            raise ValueError("k must not exceed the reranking candidate_pool")
        response = self.hybrid.search(query, k=self.candidate_pool, db_id=db_id,
                                      allowed_db_ids=allowed_db_ids, explain=explain)
        candidates = response["results"]
        results = []
        if candidates:
            passages = [searchable_text(self.lookup[hit["document_id"]]) for hit in candidates]
            scores = np.asarray(self.reranker.score(query, passages), dtype=np.float32)
            if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
                raise ValueError("Expected one finite reranker score per candidate")
            ranked = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]["document_id"]))[:k]
            for i in ranked:
                hit = {**candidates[i], "score": float(scores[i]), "hybrid_score": candidates[i]["score"]}
                if explain:
                    hit["hybrid_explanation"] = hit.pop("explanation", {})
                    hit["reranking"] = {"hybrid_rank": i + 1, "metric": "cross_encoder_logit"}
                results.append(hit)
        return {**response, "method": "hybrid_rerank", "results": results,
                "score_type": "cross_encoder_logit", "rerank_candidate_count": len(candidates),
                "elapsed_ms": (time.perf_counter() - started) * 1000}

    def save(self, path: Path) -> None:
        """Save nested hybrid snapshot and reranker identity, not model weights."""
        save_json(path, {"version": VERSION, **self.configuration(), "hybrid": self.hybrid.snapshot()})

    @classmethod
    def load(cls, path: Path, bundle: Path, *, encoder: Encoder | None = None,
             reranker: Reranker | None = None, device: str = "cpu"):
        """Verify catalogue and model recipe before restoring query-time reranking."""
        payload = json.loads(path.read_text())
        expected = reranker.configuration() if reranker is not None else reranker_configuration()
        if (payload["version"] != VERSION or payload["method"] != "hybrid_rerank"
                or payload["reranker"] != expected):
            raise ValueError("Stale/incompatible reranker index; rebuild into a new file")
        hybrid = HybridIndex.from_snapshot(payload["hybrid"], bundle, encoder=encoder, device=device)
        return cls(hybrid, reranker if reranker is not None else MiniLMCrossEncoder(device=device),
                   candidate_pool=payload["candidate_pool"])
