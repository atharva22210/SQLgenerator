"""Declared foreign keys become question-weighted edges between schema documents.

Read in order: foreign_key_edges -> edge_text -> question_costs -> shortest_path.
We preserve FK direction/columns as evidence, but traverse in either direction:
customers <- purchases -> products must be discoverable from customers too.
"""
from dataclasses import dataclass
import heapq

import numpy as np

from rag.dense.encoder import Encoder
from rag.dense.index import normalize_rows

EDGE_TEXT_VERSION = "foreign_key_endpoints_v1"


def foreign_key_edges(documents: list[dict]) -> list[dict]:
    """Build FK records using only table metadata; never infer edges from SQL labels.

    Each target is resolved inside the source database. Unknown targets fail
    rather than accidentally connecting identically named tables across databases.
    Multiple FKs between the same tables retain distinct column-level evidence.
    """
    lookup = {(doc["db_id"], doc["table"]): doc["document_id"] for doc in documents}
    edges = []
    for doc in documents:
        for key in doc["foreign_keys"]:
            target = lookup.get((doc["db_id"], key["referenced_table"]))
            if target is None:
                raise ValueError(f"Unknown foreign-key target in {doc['document_id']}")
            edges.append({"from_document_id": doc["document_id"], "to_document_id": target,
                          "columns": key["columns"], "referenced_columns": key["referenced_columns"]})
    return edges


def edge_text(edge: dict, documents: dict[str, dict]) -> str:
    """Describe one relationship for embedding, including names but no opaque IDs.

    For example: 'Table purchases columns (customer_id) reference table customers
    columns (id).' Purchases and wishlists then have different representations.
    Names provide evidence of meaning, not proof that a join answers a question.
    """
    source = documents[edge["from_document_id"]]["table"]
    target = documents[edge["to_document_id"]]["table"]
    return (f"Table {source} columns ({', '.join(edge['columns'])}) reference "
            f"table {target} columns ({', '.join(edge['referenced_columns'])}).")


@dataclass(frozen=True)
class GraphPath:
    """An ordered table path plus FK evidence IDs and its summed query cost."""
    nodes: tuple[str, ...]
    edge_ids: tuple[int, ...]
    cost: float


class ForeignKeyGraph:
    """Store FK adjacency and normalized edge vectors with shape [edges,dimension].

    documents: Verified table catalogue. encoder: Same encoder as dense retrieval.
    dimension: Dense embedding size. batch_size: Build-time inference batch size.
    embeddings: Optional saved edge vectors; loading must not re-encode edges.
    Empty graphs are valid: undeclared relationships cannot be recovered here.
    """
    def __init__(self, documents: list[dict], encoder: Encoder, *, dimension: int,
                 batch_size: int = 32, embeddings=None):
        self.edges = foreign_key_edges(documents)
        self.adjacency = {doc["document_id"]: [] for doc in documents}
        for index, edge in enumerate(self.edges):
            source, target = edge["from_document_id"], edge["to_document_id"]
            self.adjacency[source].append((target, index))
            self.adjacency[target].append((source, index))
        for neighbors in self.adjacency.values():
            neighbors.sort()
        if not self.edges:
            if embeddings is not None and np.asarray(embeddings).size:
                raise ValueError("Expected no embeddings for an empty FK graph")
            self.embeddings = np.empty((0, dimension), dtype=np.float32)
        else:
            if embeddings is None:
                lookup = {doc["document_id"]: doc for doc in documents}
                embeddings = encoder.encode([edge_text(edge, lookup) for edge in self.edges], batch_size=batch_size)
            self.embeddings = normalize_rows(embeddings, rows=len(self.edges))
            if self.embeddings.shape[1] != dimension:
                raise ValueError("Edge and table embedding dimensions differ")

    def question_costs(self, query_vector: np.ndarray, allowed: set[str], *, alpha: float) -> dict[int, float]:
        """Map eligible edge IDs to positive query-dependent traversal costs.

        query_vector has shape [1,H] and unit length. Convert cosine [-1,1] to
        s in [0,1], then cost=1-alpha*s. With 0<alpha<1 every edge is positive,
        so Dijkstra applies. A longer relevant path can beat a shorter weak one.
        Unauthorized edges are excluded before similarity/path calculation.
        """
        eligible = [i for i, edge in enumerate(self.edges)
                    if edge["from_document_id"] in allowed and edge["to_document_id"] in allowed]
        if not eligible:
            return {}
        similarities = np.clip((self.embeddings[eligible] @ query_vector[0] + 1) / 2, 0, 1)
        return {i: 1 - alpha * float(similarity) for i, similarity in zip(eligible, similarities)}

    def shortest_path(self, source: str, target: str, costs: dict[int, float]) -> GraphPath | None:
        """Return the minimum-summed-cost path, or None if disconnected.

        Dijkstra expands the cheapest partial path using a min-heap. Positive
        costs mean revisiting a table cannot improve its shortest distance, so
        the finite graph terminates naturally without an artificial hop limit.
        Ties use table IDs then edge IDs for deterministic results. Cost entries
        also define which edges may be traversed; missing edges are forbidden.
        """
        queue = [(0.0, (source,), ())]
        best = {source: queue[0]}
        while queue:
            cost, nodes, edge_ids = heapq.heappop(queue)
            current = nodes[-1]
            if best[current] != (cost, nodes, edge_ids):
                continue
            if current == target:
                return GraphPath(nodes, edge_ids, cost)
            for neighbor, edge_id in self.adjacency[current]:
                if edge_id not in costs:
                    continue
                candidate = (cost + costs[edge_id], nodes + (neighbor,), edge_ids + (edge_id,))
                if neighbor not in best or candidate < best[neighbor]:
                    best[neighbor] = candidate
                    heapq.heappush(queue, candidate)
        return None
