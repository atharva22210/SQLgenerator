"""Hybrid top-k seeds plus query-weighted FK paths; no unweighted variant."""
from itertools import combinations
import json
import math
from pathlib import Path
import time

from rag.common.search import candidate_indices, make_hit
from rag.common.storage import save_json
from rag.dense.encoder import Encoder
from rag.dense.index import normalize_rows
from rag.hybrid import HybridIndex
from rag.hybrid_graph.graph import EDGE_TEXT_VERSION, ForeignKeyGraph, foreign_key_edges
from rag.hybrid_graph.connections import edge_nodes, select_connections

VERSION = 2


class HybridGraphIndex:
    """Expand hybrid results using question-dependent weighted shortest paths.

    hybrid: Existing hybrid index with its verified documents and frozen encoder.
    alpha: Semantic influence in cost=1-alpha*((cosine+1)/2), strictly (0,1).
    max_added_tables: Additional schema budget beyond k seeds (0..50).
    batch_size: Edge-text embedding batch size during construction.
    edge_embeddings: Optional saved [edges,H] vectors to reuse on load.
    """
    def __init__(self, hybrid: HybridIndex, *, alpha: float = .5,
                 max_added_tables: int = 10, batch_size: int = 32, edge_embeddings=None):
        if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("graph alpha must be finite and strictly between 0 and 1")
        if (isinstance(max_added_tables, bool) or not isinstance(max_added_tables, int)
                or not 0 <= max_added_tables <= 50):
            raise ValueError("max_added_tables must be an integer in [0,50]")
        self.hybrid, self.alpha = hybrid, alpha
        self.max_added_tables = max_added_tables
        self.documents, self.by_db, self.source_sha256 = hybrid.documents, hybrid.by_db, hybrid.source_sha256
        self.lookup = {doc["document_id"]: doc for doc in self.documents}
        self.graph = ForeignKeyGraph(self.documents, hybrid.dense.encoder,
                                     dimension=hybrid.dense.embeddings.shape[1], batch_size=batch_size,
                                     embeddings=edge_embeddings)

    def configuration(self) -> dict:
        """Settings for reproducible retrieval-only evaluations and snapshots."""
        return {"method": "hybrid_graph", "path_method": "question_weighted_dijkstra_kruskal",
                "edge_text_version": EDGE_TEXT_VERSION, "alpha": self.alpha,
                "max_added_tables": self.max_added_tables,
                "hybrid": self.hybrid.configuration()}

    def search(self, query: str, *, k: int = 4, db_id: str | None = None,
               allowed_db_ids: set[str] | None = None, explain: bool = False) -> dict:
        """Retrieve k seeds, then connect separate groups using affordable paths.

        k limits seeds; results can contain up to k+max_added_tables schemas.
        Find one minimum-cost path per same-database seed pair with no hop cap.
        Select paths joining distinct groups, then remove cycles and non-seed
        leaves. Retain seeds first and append only final bridge tables. Budget
        limits can leave groups disconnected; no globally optimal join is claimed.
        """
        started = time.perf_counter()
        candidates = candidate_indices(self.by_db, query, k, db_id, allowed_db_ids)
        response = self.hybrid.search(query, k=k, db_id=db_id, allowed_db_ids=allowed_db_ids, explain=explain)
        seeds = response["results"]
        seed_ids = [hit["document_id"] for hit in seeds]
        pairs = [(a, b) for a, b in combinations(sorted(seed_ids), 2)
                 if self.lookup[a]["db_id"] == self.lookup[b]["db_id"]]
        allowed = {self.documents[i]["document_id"] for i in candidates}
        costs, paths = {}, []
        if pairs and self.graph.edges:
            vector = normalize_rows(self.hybrid.dense.encoder.encode([query]), rows=1)
            costs = self.graph.question_costs(vector, allowed, alpha=self.alpha)
            for source, target in pairs:
                path = self.graph.shortest_path(source, target, costs)
                if path is not None:
                    paths.append(path)
        results = [{**hit, "retrieval_role": "seed"} for hit in seeds]
        selection = select_connections(paths, self.graph.edges, costs, set(seed_ids),
                                       max_added_tables=self.max_added_tables)
        connected_nodes = {node for i in selection.edge_ids for node in edge_nodes(self.graph.edges[i])}
        for identity in sorted(connected_nodes - set(seed_ids)):
            # Connectivity has no RRF relevance score. Only retain bridges
            # surviving cleanup; pruning an edge alone would not shrink context.
            hit = make_hit(self.lookup[identity], None)
            hit["retrieval_role"] = "bridge"
            results.append(hit)
        graph_info = {"seed_document_ids": seed_ids, "added_document_ids":
                      [hit["document_id"] for hit in results if hit["retrieval_role"] == "bridge"],
                      "pairs_considered": len(pairs), "paths_found": len(paths),
                      "paths_added": len(selection.paths), "paths_skipped_for_budget": len(selection.skipped_paths),
                      "redundant_paths_skipped": selection.redundant_paths,
                      "pruned_document_ids": list(selection.pruned_document_ids),
                      "connection_edge_count": len(selection.edge_ids),
                      "connection_cost": sum(costs[i] for i in selection.edge_ids),
                      "seed_groups": [list(group) for group in selection.seed_groups],
                      "all_seeds_connected": bool(seed_ids) and len(selection.seed_groups) == 1}
        if explain:
            # These paths describe selection BEFORE cleanup. connections is the
            # authoritative final edge set; some original path edges may be pruned.
            graph_info["paths_stage"] = "before_cycle_and_leaf_pruning"
            graph_info["paths"] = [{"document_ids": list(path.nodes), "cost": path.cost,
                "relationships": [{**self.graph.edges[i], "cost": costs[i],
                                   "similarity": (1 - costs[i]) / self.alpha} for i in path.edge_ids]}
                for path in selection.paths]
            graph_info["skipped_paths"] = [list(path.nodes) for path in selection.skipped_paths]
            graph_info["connections"] = [{**self.graph.edges[i], "cost": costs[i],
                                           "similarity": (1 - costs[i]) / self.alpha}
                                          for i in selection.edge_ids]
        return {**response, "method": "hybrid_graph", "seed_results": seeds, "results": results,
                "graph": graph_info, "context_table_count": len(results),
                "elapsed_ms": (time.perf_counter() - started) * 1000}

    def save(self, path: Path) -> None:
        """Persist both table and relationship vectors atomically in one JSON file."""
        save_json(path, {"version": VERSION, **self.configuration(), "hybrid": self.hybrid.snapshot(),
                         "edges": self.graph.edges, "edge_embeddings": self.graph.embeddings.tolist()})

    @classmethod
    def load(cls, path: Path, bundle: Path, *, encoder: Encoder | None = None, device: str = "cpu"):
        """Verify nested hybrid/catalogue and edge recipe, reusing saved vectors."""
        payload = json.loads(path.read_text())
        if (payload["version"] != VERSION or payload["method"] != "hybrid_graph"
                or payload["path_method"] != "question_weighted_dijkstra_kruskal"
                or payload["edge_text_version"] != EDGE_TEXT_VERSION):
            raise ValueError("Stale/incompatible graph index; rebuild into a new file")
        hybrid = HybridIndex.from_snapshot(payload["hybrid"], bundle, encoder=encoder, device=device)
        if payload["edges"] != foreign_key_edges(hybrid.documents):
            raise ValueError("Saved graph disagrees with catalogue foreign keys")
        return cls(hybrid, alpha=payload["alpha"],
                   max_added_tables=payload["max_added_tables"], edge_embeddings=payload["edge_embeddings"])
