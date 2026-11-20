"""Question-dependent paths, expansion budgets, snapshots and shared entry points."""
from contextlib import redirect_stdout
import http.client
from http.server import HTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from rag.bm25 import BM25Index
from rag.common.catalog import sha256
from rag.common.cli import main
from rag.common.evaluation import evaluate
from rag.common.service import make_handler
from rag.common.search import make_hit
from rag.dense import DenseIndex
from rag.hybrid import HybridIndex
from rag.hybrid_graph import HybridGraphIndex
from rag.hybrid_graph.graph import ForeignKeyGraph


def document(table, references=(), db="shop"):
    return {"document_id": f"{db}.{table}", "db_id": db, "table": table, "domain": "retail",
            "domain_description": "", "columns": [], "primary_key": [],
            "foreign_keys": [{"columns": [f"{target}_id"], "referenced_table": target,
                              "referenced_columns": ["id"]} for target in references],
            "text": f"Table {table}", "schema_sql": f"CREATE TABLE {table} (id INT)"}


class FixtureEncoder:
    """Fixed vectors prove query-conditioned routing, not real model quality."""
    def __init__(self):
        self.seen = []

    def configuration(self):
        return {"model_id": "offline-graph-fixture", "revision": "1"}

    def encode(self, texts, *, batch_size=32):
        self.seen.extend(texts)
        vectors = []
        for text in texts:
            if text == "purchased" or text.startswith("Table purchases "):
                vectors.append([1, 0])
            elif text == "wanted" or text.startswith("Table wishlists "):
                vectors.append([0, 1])
            elif text.startswith(("customers ", "products ")):
                vectors.append([1, 1])
            else:
                vectors.append([-1, -1])
        return np.asarray(vectors, dtype=np.float32)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.docs = [document("customers"), document("products"),
                     document("purchases", ("customers", "products")),
                     document("wishlists", ("customers", "products")),
                     document("shipping", ("purchases",))]
        self.encoder = FixtureEncoder()
        self.hybrid = HybridIndex(BM25Index(self.docs), DenseIndex(self.docs, self.encoder))
        self.index = HybridGraphIndex(self.hybrid)

    def bundle(self, root):
        (root / "catalog").mkdir()
        path = root / "catalog/tables.jsonl"
        path.write_text("".join(json.dumps(doc) + "\n" for doc in self.docs))
        checksum = sha256(path)
        (root / "manifest.json").write_text(json.dumps({"complete": True, "counts": {"tables": len(self.docs)},
            "files": {"catalog/tables.jsonl": {"sha256": checksum}}}))
        return checksum

    def test_question_changes_bridge_for_the_same_hybrid_seeds(self):
        purchased = self.index.search("purchased", k=2, explain=True)
        wanted = self.index.search("wanted", k=2, explain=True)
        self.assertEqual(purchased["graph"]["seed_document_ids"], ["shop.customers", "shop.products"])
        self.assertEqual(wanted["graph"]["seed_document_ids"], purchased["graph"]["seed_document_ids"])
        self.assertEqual(purchased["graph"]["added_document_ids"], ["shop.purchases"])
        self.assertEqual(wanted["graph"]["added_document_ids"], ["shop.wishlists"])
        self.assertIsNone(purchased["results"][-1]["score"])
        path = purchased["graph"]["paths"][0]
        self.assertEqual(path["document_ids"], ["shop.customers", "shop.purchases", "shop.products"])
        self.assertAlmostEqual(path["cost"], 1.0)  # Two cost .5 edges.
        self.assertAlmostEqual(sum(edge["cost"] for edge in path["relationships"]), path["cost"])
        self.assertNotIn("shop.shipping", [hit["document_id"] for hit in purchased["results"]])

    def test_zero_expansion_single_seed_and_permissions(self):
        index = HybridGraphIndex(self.hybrid, max_added_tables=0)
        self.assertEqual(index.search("purchased", k=2)["graph"]["added_document_ids"], [])
        self.assertEqual(self.index.search("purchased", k=1)["graph"]["pairs_considered"], 0)
        self.assertEqual(self.index.search("purchased", allowed_db_ids=set())["results"], [])
        with self.assertRaises(ValueError):
            self.index.search("purchased", db_id="secret", allowed_db_ids={"shop"})
        # Even a connecting edge cannot be traversed if an endpoint is outside scope.
        costs = self.index.graph.question_costs(np.array([[1., 0.]]), {"shop.customers", "shop.products"}, alpha=.5)
        self.assertEqual(costs, {})

    def test_empty_graph_and_unknown_reference(self):
        docs = [document("customers"), document("products")]
        graph = ForeignKeyGraph(docs, self.encoder, dimension=2)
        self.assertEqual(graph.embeddings.shape, (0, 2))
        self.assertIsNone(graph.shortest_path("shop.customers", "shop.products", {}))
        with self.assertRaises(ValueError):
            ForeignKeyGraph([document("orders", ("missing",))], self.encoder, dimension=2)
        # Same table name in a different database does not resolve an FK.
        with self.assertRaises(ValueError):
            ForeignKeyGraph([document("orders", ("customers",)), document("customers", db="secret")],
                            self.encoder, dimension=2)

    def test_dijkstra_prefers_lower_cost_and_has_no_hop_limit(self):
        docs = [document("a"), document("b", ("a",)), document("c", ("b",)),
                document("d", ("a", "c"))]
        graph = ForeignKeyGraph(docs, self.encoder, dimension=2)
        # Edges: b-a, c-b, d-a, d-c. Three cheap edges beat one expensive edge.
        costs = {0: .1, 1: .1, 2: .9, 3: .1}
        path = graph.shortest_path("shop.a", "shop.d", costs)
        self.assertEqual(path.nodes, ("shop.a", "shop.b", "shop.c", "shop.d"))
        self.assertAlmostEqual(path.cost, .3)
        # Seven edges must work; there is no longer a default four-hop cutoff.
        docs = [document("n0")] + [document(f"n{i}", (f"n{i-1}",)) for i in range(1, 8)]
        graph = ForeignKeyGraph(docs, self.encoder, dimension=2)
        path = graph.shortest_path("shop.n0", "shop.n7", {i: .5 for i in range(7)})
        self.assertEqual(len(path.edge_ids), 7)

    def test_path_ties_deduplication_and_whole_path_budget(self):
        costs = {i: .5 for i in range(len(self.index.graph.edges))}
        graph = self.index.graph
        path = graph.shortest_path("shop.customers", "shop.products", costs)
        self.assertEqual(path.nodes[1], "shop.purchases")  # Lexicographic tie.
        # products -> purchases -> shipping -> delivery needs two new tables.
        docs = self.docs + [document("delivery", ("shipping",))]
        hybrid = HybridIndex(BM25Index(docs), DenseIndex(docs, self.encoder))
        seeds = [make_hit(docs[1], .1), make_hit(docs[-1], .1)]
        limited = HybridGraphIndex(hybrid, max_added_tables=0)
        with patch.object(hybrid, "search", return_value={"results": seeds}):
            self.assertEqual(len(limited.search("purchased", k=2)["results"]), 2)
            limited = HybridGraphIndex(hybrid, max_added_tables=1)
            result = limited.search("purchased", k=2)
            self.assertEqual(len(result["results"]), 2)
            self.assertEqual(result["graph"]["paths_skipped_for_budget"], 1)
        seeds = [make_hit(self.docs[i], .1) for i in (0, 1, 4)]
        with patch.object(self.hybrid, "search", return_value={"results": seeds}):
            result = HybridGraphIndex(self.hybrid, max_added_tables=1).search("purchased", k=3)
        self.assertEqual(result["graph"]["added_document_ids"], ["shop.purchases"])
        self.assertEqual(result["graph"]["paths_added"], 2)
        self.assertEqual(result["graph"]["redundant_paths_skipped"], 1)
        self.assertTrue(result["graph"]["all_seeds_connected"])
        self.assertEqual(len(result["results"]), 4)

    def test_seeds_in_different_databases_are_never_connected(self):
        docs = self.docs + [document("customers", db="secret")]
        hybrid = HybridIndex(BM25Index(docs), DenseIndex(docs, self.encoder))
        index = HybridGraphIndex(hybrid)
        seeds = [make_hit(docs[0], .1), make_hit(docs[-1], .1)]
        with patch.object(hybrid, "search", return_value={"results": seeds}):
            result = index.search("purchased", k=2)
        self.assertEqual(result["graph"]["pairs_considered"], 0)
        self.assertEqual(result["graph"]["added_document_ids"], [])

    def test_parameter_validation(self):
        for kwargs in ({"alpha": 0}, {"alpha": 1}, {"alpha": float("nan")},
                       {"max_added_tables": True}, {"max_added_tables": -1}):
            with self.assertRaises(ValueError):
                HybridGraphIndex(self.hybrid, **kwargs)

    def test_redundant_pair_path_does_not_add_its_table_to_context(self):
        docs = [document("a"), document("b"), document("c"), document("x", ("a", "b")),
                document("y", ("b", "c")), document("z", ("a", "c"))]
        hybrid = HybridIndex(BM25Index(docs), DenseIndex(docs, self.encoder))
        index = HybridGraphIndex(hybrid)
        seeds = [make_hit(doc, .1) for doc in docs[:3]]
        costs = dict(enumerate([.5, .5, .6, .6, .9, .9]))
        with patch.object(hybrid, "search", return_value={"results": seeds}), \
             patch.object(index.graph, "question_costs", return_value=costs):
            response = index.search("purchased", k=3, explain=True)
        self.assertEqual(response["graph"]["paths_found"], 3)
        self.assertEqual(response["graph"]["paths_added"], 2)
        self.assertEqual(response["graph"]["added_document_ids"], ["shop.x", "shop.y"])
        self.assertNotIn("shop.z", [hit["document_id"] for hit in response["results"]])
        self.assertEqual(response["graph"]["connection_edge_count"], 4)

    def test_snapshot_roundtrip_reuses_edge_vectors_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checksum = self.bundle(root)
            self.hybrid.bm25.source_sha256 = self.hybrid.dense.source_sha256 = self.hybrid.source_sha256 = checksum
            self.index.source_sha256 = checksum
            path = root / "graph.json"
            self.index.save(path)
            fresh = FixtureEncoder()
            loaded = HybridGraphIndex.load(path, root, encoder=fresh)
            self.assertEqual(fresh.seen, [])
            self.assertEqual(loaded.search("wanted", k=2)["results"], self.index.search("wanted", k=2)["results"])
            with self.assertRaises(FileExistsError):
                self.index.save(path)
            original = path.read_text()
            payload = json.loads(original)
            payload["version"] = 1
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                HybridGraphIndex.load(path, root, encoder=fresh)
            payload = json.loads(original)
            payload["edges"][0]["columns"] = ["changed"]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                HybridGraphIndex.load(path, root, encoder=fresh)
            path.write_text(original)
            (root / "catalog/tables.jsonl").write_text("corrupt")
            with self.assertRaises(ValueError):
                HybridGraphIndex.load(path, root, encoder=fresh)

    def test_cli_and_evaluation_measure_expansion_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.bundle(root)
            (root / "labels").mkdir()
            labels = root / "labels/dev.jsonl"
            labels.write_text(json.dumps({"id": "q", "question": "purchased", "db_id": "shop",
                "gold_document_ids": ["shop.customers", "shop.products", "shop.purchases"]}) + "\n")
            def run(tail):
                output = io.StringIO()
                prefix = ["rag", "--bundle", str(root), "--index", str(root / "graph.json")]
                with patch("sys.argv", prefix + tail), redirect_stdout(output):
                    main(default_backend="hybrid_graph")
                return json.loads(output.getvalue())
            with patch("rag.dense.encoder.MiniLMEncoder", return_value=FixtureEncoder()), \
                 patch("rag.dense.index.MiniLMEncoder", return_value=FixtureEncoder()), \
                 patch("rag.dense.index.encoder_configuration", return_value=FixtureEncoder().configuration()):
                self.assertEqual(run(["build"])["tables"], 5)
                self.assertEqual(len(run(["search", "--query", "purchased"])["seed_results"]), 4)
                self.assertEqual(run(["search", "--query", "purchased", "--k", "2"])["context_table_count"], 3)
                report = run(["evaluate", "--scoped", "--k", "2", "--output", str(root / "report.json")])
                self.assertAlmostEqual(report["all"]["recall"], 2/3)
                self.assertEqual(report["all"]["precision_at_k"], 1)
                self.assertEqual(report["all"]["context_recall"], 1)
                self.assertEqual(report["all"]["context_precision"], 1)
                self.assertEqual(report["all"]["context_table_count"], 3)
                default_report = run(["evaluate", "--output", str(root / "default-report.json")])
                self.assertEqual(default_report["k"], 4)

    def test_authenticated_http_api_returns_expanded_context(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(self.index, "t" * 32, {"shop"}))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            connection.request("POST", "/search", json.dumps({"query": "purchased", "k": 2}),
                {"Content-Type": "application/json", "Authorization": "Bearer " + "t" * 32})
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["graph"]["added_document_ids"], ["shop.purchases"])
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            connection.request("POST", "/search", json.dumps({"query": "purchased"}),
                {"Content-Type": "application/json", "Authorization": "Bearer " + "t" * 32})
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(len(payload["seed_results"]), 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
